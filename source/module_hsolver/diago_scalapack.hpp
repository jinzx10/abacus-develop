#ifndef DIAGO_SCALAPACK_HPP
#define DIAGO_SCALAPACK_HPP

#include <vector>
#include <complex>
#include <cassert>
#include <functional>
#include <algorithm>
#include <numeric>

#include "module_base/blacs_connector.h"
#include "module_base/scalapack_connector.h"
#include "module_basis/module_ao/parallel_2d.h"

inline void pdiagx_check_fatal(int info, const std::vector<int>& ifail, bool is_cplx) {

    // This function takes the output info of pdsyevx/pzheevx and will abort the program
    // if any of the following fatal error occurs:
    //      info < 0: illegal input
    //      info % 2 != 0: eigenvectors failed to converge
    //      (info / 4) % 2 != 0: space limit (not supposed to happen)
    //      (info / 8) % 2 != 0: fails to compute eigenvalues
    //
    // The following info is not considered fatal:
    //      (info / 2) % 2 != 0: eigenvectors unorthogonalized
    //
    // If any fatal error occurs, this function will throw an exception and abort;
    // otherwise it returns without any effect.

    std::string solver = is_cplx ? "pzheevx" : "pdsyevx";
    std::string where = "file " __FILE__ ", in pdiagx(): " + solver + " failed: ";

    if (info < 0) {
        info = -info;
        if (info > 100) {
            throw std::runtime_error(where
                    + "the " + std::to_string(info % 100) + "-th entry in the "
                    + std::to_string(info / 100) + "-th argument is illegal.\n"
            );
        } else {
            throw std::runtime_error(where
                    + "the " + std::to_string(info) + "-th argument is illegal.\n"
            );
        }
    } 

    if (info % 2) {
        std::string ifail_str = std::accumulate(ifail.begin(), ifail.end(), std::string("ifail = "), 
            [](std::string str, int i) { return i != 0 ? str + std::to_string(i) + " " : str; });
        throw std::runtime_error(where
                + "one or more eigenvectors failed to converge:\n" + ifail_str + "\n");
    }

    if (info / 4 % 2) {
        throw std::runtime_error(where + "space limit (should never occur)\n");
    }

    if (info / 8 % 2) {
        throw std::runtime_error(where + "fails to compute eigenvalues");
    }

    // the only non-fatal error: eigenvectors unorthogonalized
    if (info / 2 % 2) {
        return;
    }

    throw std::runtime_error(where + "unknown info");
}


template <typename T>
void pdiagx(int neig, const T* A, const int* desca, double* w, T* z, const int* descz) {
    /**
     * @brief Eigenvalues & eigenvectors for a block-cyclic-distributed real symmetric or
     * complex Hermitian matrix.
     *
     * @param[in]   neig   number of eigenpairs to compute
     * @param[in]   A       local matrix after block-cyclic distribution
     * @param[in]   desca   array descriptor of A
     * @param[out]  w       eigenvalues
     * @param[out]  z       local matrix of eigenvectors after block-cyclic distribution
     * @param[in]   descz   array descriptor of z
     *
     */
    static_assert(std::is_same<T, double>::value ||
                  std::is_same<T, std::complex<double>>::value,
                  "T must be double or std::complex<double>");

    const bool is_cplx = std::is_same<T, std::complex<double>>::value;
    const int one_i = 1;
    const int nglb = desca[2]; // global size of the matrix

    // basic sanity checks
    assert(desca[2] == desca[3]); // matrix A must be globally square
    assert(neig <= nglb);

    assert((z == nullptr) == (descz == nullptr));



    // process grid information
    int ctxt = desca[1];
    int nprow, npcol, myrow, mycol, nprocs;
    Cblacs_gridinfo(ctxt, &nprow, &npcol, &myrow, &mycol);
    nprocs = nprow * npcol;

    // pdsyevx/pzheevx destroys the input matrix, but they do not guarantee
    // a successful calculation even for valid inputs. Therefore, we make a
    // copy of the input matrix in case we need to re-run the calculation.
    // The input matrix `A` will remain intact throughout the function.
    const int ncol_A = numroc_(desca + 3, desca + 5, &mycol, desca + 7, &npcol);
    const size_t nelem_A = static_cast<size_t>(desca[8]) * ncol_A;
    std::vector<T> A_copy(A, A + nelem_A);

    // input parameters
    char jobz = (z == nullptr) ? 'N' : 'V';
    char range = (neig == nglb) ? 'A' : 'I';
    double vl = 0.0, vu = 1.0;
    double abstol = 2.0 * pdlamch_(&ctxt, "S");
    int m = 0, nz = 0;
    double orfac = -1.0;

    // work space
    int lwork = -1, liwork = -1, lrwork = -1;
    std::vector<T> work(3);
    std::vector<double> rwork(3);
    std::vector<int> iwork(1);

    // output informational parameters
    std::vector<int> ifail(nglb);
    std::vector<int> iclustr(2 * nprocs);
    std::vector<double> gap(nprocs);
    int info = 0;

    std::function<void()> eigen;
    if (is_cplx) {
        eigen = [&]() {
            pzheevx_(
                &jobz, &range, "U",
                &nglb,
                reinterpret_cast<std::complex<double>*>(A_copy.data()),
                &one_i, &one_i, desca,
                &vl, &vu, &one_i, &neig,
                &abstol,
                &m, &nz,
                w,
                &orfac,
                reinterpret_cast<std::complex<double>*>(z),
                &one_i, &one_i, descz,
                reinterpret_cast<std::complex<double>*>(work.data()), &lwork,
                rwork.data(), &lrwork,
                iwork.data(), &liwork,
                ifail.data(), iclustr.data(), gap.data(),
                &info
            );
        };
    } else {
        eigen = [&]() {
            pdsyevx_(
                &jobz, &range, "U",
                &nglb,
                reinterpret_cast<double*>(A_copy.data()),
                &one_i, &one_i, desca,
                &vl, &vu, &one_i, &neig,
                &abstol,
                &m, &nz,
                w,
                &orfac,
                reinterpret_cast<double*>(z),
                &one_i, &one_i, descz,
                reinterpret_cast<double*>(work.data()), &lwork,
                iwork.data(), &liwork,
                ifail.data(), iclustr.data(), gap.data(),
                &info
            );
        };
    }

    // work space query
    eigen();

    // allocate workspace
    lwork = std::real(work[0]);
    work.resize(std::max(lwork, 3));
    liwork = iwork[0];
    iwork.resize(liwork);
    if (is_cplx) {
        lrwork = rwork[0];
        rwork.resize(std::max(lrwork, 3));
    }

    // Ideally the next call will compute eigenpairs successfully, in which case
    // `info` will be 0. If not, we will check the error code and decide whether
    // to re-run the calculation with a larger workspace or throw an exception.
    //
    // Most error codes indicate fatal error. In some cases, calculations may
    // finish with eigenvectors unorthogonalized due to insufficient workspace.
    // If that happens, we will increase the size of the workspace and re-run the
    // calculation. This, however, should happen a few times at most and we will
    // abort the program if the calculation fails after enough trials.
    const int n_trial_max = 3;

    for (int i_trial = 0; i_trial < n_trial_max; ++i_trial) {

        // actual calculation
        eigen();

        if (info == 0) {
            return;
        }

        // Abort if the error is fatal; pass without effect otherwise.
        pdiagx_check_fatal(info, ifail, is_cplx);

        // If the error is not fatal, prepare for the re-run
        std::copy(A, A + nelem_A, A_copy.data());
        
        // The only non-fatal non-zero `info` is that eigenvectors are not
        // orthogonalized, in which case we need to increase the size of the
        // workspace based on iclustr and re-run the calculation.
        int max_cluster_size = 1;
        for (size_t i = 0; i < iclustr.size() / 2; ++i) {
            max_cluster_size = std::max(iclustr[2 * i + 1] - iclustr[2 * i] + 1,
                                        max_cluster_size);
        }

        if (is_cplx) {
            lrwork += (max_cluster_size - 1) * nglb;
            rwork.resize(lrwork);
        } else {
            lwork += (max_cluster_size - 1) * nglb;
            work.resize(lwork);
        }

    } // end of while

    throw std::runtime_error("pdiagx fails after " + std::to_string(n_trial_max) + " trials.\n");
}


template <typename T>
int canon_diag(
    int neig,
    T* H, T* S, const int* desc_HS,
    double* E,
    T* C, const int* desc_C,
    double thr = 1e-6
) {
    /**
     * @brief Solves a generalized eigenvalue problem with canonical orthogonalization.
     *
     * Given a (potentially singular) generalized eigenvalue problem
     *
     *                  H * C = S * C * diag(E)
     *
     * this function
     *
     * 1. computes the eigen-decomposition of S = U * diag(D) * U^H
     * 2. selects the subset of eigenvectors, denote X, corresponding to non-tiny eigenvalues
     * 3. rescales X column-wise by 1/sqrt(D)
     * 4. forms subspace Hamiltonian H_sub = X^H * H * X
     * 5. solves the eigenvalue problem H_sub * C_sub = C_sub * E
     * 6. gets the eigenvectors of the original problem C = X * C_sub
     *
     */

    static_assert(std::is_same<T, double>::value ||
                  std::is_same<T, std::complex<double>>::value,
                  "T must be double or std::complex<double>");

    bool is_cplx = std::is_same<T, std::complex<double>>::value;
    const T one_f = 1.0; // floating point 1.0 (real or complex)
    const T zero_f = 0.0; // floating point 0.0 (real or complex)
    const int one_i = 1; // integer 1

    // global size of H/S
    const int nglb = desc_HS[2];
    assert(desc_HS[2] == desc_HS[3]); // H/S must be square

    // block size of block-cyclic distribution
    const int nb = desc_HS[4];
    assert(desc_HS[4] == desc_HS[5]); // block must be square

    // process grid information
    int ctxt = desc_HS[1];
    int nprow, npcol, myrow, mycol;
    Cblacs_gridinfo(ctxt, &nprow, &npcol, &myrow, &mycol);

    // size of the local H/S matrix after block-cyclic distribution
    // number of elements in the local matrix of H/S/U
    const int nrow = numroc_(&nglb, &nb, &myrow, desc_HS + 6, &nprow);
    const int ncol = numroc_(&nglb, &nb, &mycol, desc_HS + 7, &npcol);
    size_t nloc = static_cast<size_t>(nrow) * ncol;

    //====== allocate memory for temporary arrays ======
    // U (eigenvectors of S), X (rescaled from U in-place)  <-- nloc
    // H_sub (X^H * H * X)                                  <-- at most nloc
    // X^H * H, C_sub (eigenvectors of H_sub)               <-- at most nloc
    // D (eigenvalues of S)                                 <-- nglb, always real
    size_t buf_size = 3 * nloc + (nglb + 1) * sizeof(double) / sizeof(T);
    std::vector<T> buffer(buf_size);

    T* U = buffer.data();
    T* H_sub = U + nloc;
    T* XhH = H_sub + nloc;
    T* C_sub = XhH; // share the same memory since they don't coexist
    double* D = reinterpret_cast<double*>(C_sub + nloc);


    //====== 1. eigen-decomposition of S ======
    pdiagx(nglb, S, desc_HS, D, U, desc_HS);

    //====== 2. find the number of tiny eigenvalues below thr ======
    // number of tiny eigenvalues of S below thr
    int ntiny = std::find_if(D, D + nglb, [thr](double x) { return x > thr; }) - D;

    // the "true dimension" of the eigenvalue problem (linear dependency removed)
    int dim = nglb - ntiny;

    //======= 3. transformation matrix of canonical orthogonalization ======
    // block-cyclic distribution of H/S/U
    Parallel_2D p2d_HS;
    p2d_HS.set(nglb, nglb, nb, ctxt);

    // rescale U column-wise by 1/sqrt(D) (excluding tiny eigenvalues)
    for (int icolglb = ntiny; icolglb < nglb; ++icolglb) {
        int icol = p2d_HS.global2local_col(icolglb);
        if (icol != -1) { // if the local matrix contains the specified global column
            // we do an in-place scaling of U
            double inv_sqrt = 1.0 / std::sqrt(D[icolglb]);
            for (int irow = 0; irow < nrow; ++irow) {
                U[irow + icol * nrow] *= inv_sqrt;
            }
        }
    }

    // from now on X = U[:, ntiny:]

    //======= 4. form H_sub = X^H * H * X ======
    // block-cyclic distribution of H_sub/C_sub
    Parallel_2D p2d_sub;
    p2d_sub.set(dim, dim, nb, ctxt);

    // block-cyclic distribution of XhH = X^H * H
    Parallel_2D p2d_XhH;
    p2d_XhH.set(dim, nglb, nb, ctxt);

    // XhH = X^H * H
    ScalapackConnector::gemm(
        is_cplx ? 'C':'T', 'N',
        dim, nglb, nglb,
        one_f,
        U, one_i, ntiny + 1, desc_HS,
        H, one_i, one_i, desc_HS,
        zero_f,
        XhH, one_i, one_i, p2d_XhH.desc
    );

    // H_sub = X^H * H * X
    ScalapackConnector::gemm(
        'N', 'N',
        dim, dim, nglb,
        one_f,
        XhH, one_i, one_i, p2d_XhH.desc,
        U, one_i, ntiny + 1, desc_HS,
        zero_f,
        H_sub, one_i, one_i, p2d_sub.desc
    );

    //======= 5. eigenpairs of H_sub ======
    // NOTE: the documentation of pdsyevx/pzheevx suggests that the array for
    // holding eigenvectors (C_sub) be square, even if only a selected range of
    // eigenpairs is requested. This is checked via its array descriptor; passing
    // the descriptor of a non-square C_sub would result in an error.
    //
    // It might be sufficient to pass a descriptor of a square-sized matrix
    // but allocate a smaller memory for C_sub which is just enough for holding
    // the eigenvectors of interest, but whether this is safe is not clear.
    pdiagx(neig, H_sub, p2d_sub.desc, E, C_sub, p2d_sub.desc /* see NOTE above */);

    //======= transform the eigenvectors back to the original basis ======
    // C = X * C_sub
    ScalapackConnector::gemm(
        'N', 'N',
        nglb, neig, dim,
        one_f,
        U, one_i, ntiny + 1, desc_HS,
        C_sub, one_i, one_i, p2d_sub.desc,
        zero_f,
        C, one_i, one_i, desc_C
    );

    return ntiny;
}

#endif
