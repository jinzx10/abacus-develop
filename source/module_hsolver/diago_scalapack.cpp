//=====================
// AUTHOR : Peize Lin
// DATE : 2021-11-02
// REFACTORING AUTHOR : Daye Zheng
// DATE : 2022-04-14
//=====================

#include "diago_scalapack.h"

#include <cassert>
#include <cstring>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unistd.h>

#include "module_base/global_function.h"
#include "module_base/global_variable.h"
#include "module_base/scalapack_connector.h"
#include "module_hamilt_general/matrixblock.h"

typedef hamilt::MatrixBlock<double> matd;
typedef hamilt::MatrixBlock<std::complex<double>> matcd;

namespace hsolver
{

template <typename T>
void pdiagx(int n_eig, const T* A_loc, const int* desca, double* w, T* z, const int* descz) {
    /**
     * Standard eigen-decomposition for the first `n_eig` eigenpairs.
     *
     */
    static_assert(std::is_same<T, double>::value ||
                  std::is_same<T, std::complex<double>>::value,
                  "T must be double or std::complex<double>");

    const bool is_cplx = std::is_same<T, std::complex<double>>::value;
    const int one_i = 1;
    const int n_glb = desca[2];

    // some sanity checks
    assert(desca[2] == desca[3]); // A must be globally square
    assert(n_eig <= n_glb);

    // process grid information
    int ctxt = desca[1];
    int nprow, npcol, myrow, mycol, nprocs;
    Cblacs_gridinfo(ctxt, &nprow, &npcol, &myrow, &mycol);
    nprocs = nprow * npcol;

    // pdsyevx/pzheevx destroys the input matrix, but they do not guarantee
    // a successful calculation even for valid input. Therefore, we make a
    // copy of the input matrix in case we need to re-run the calculation.
    // The input matrix `A` will remain intact throughout the function.
    const int ncol_A_loc = numroc_(desca + 3, desca + 5, &mycol, desca + 7, &npcol);
    const size_t nelem_A_loc = static_cast<size_t>(desca[8]) * ncol_A_loc;
    std::vector<T> A_loc_copy(A_loc, A_loc + nelem_A_loc);


    // input parameters
    char range = (n_eig == n_glb) ? 'A' : 'I';
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
    std::vector<int> ifail(n_glb);
    std::vector<int> iclustr(2 * nprocs);
    std::vector<double> gap(nprocs);
    int info = 0;

    std::function<void()> eigen;
    if (is_cplx) {
        eigen = [&]() {
            pzheevx_(
                "V", &range, "U",
                &n_glb,
                reinterpret_cast<std::complex<double>*>(A_loc_copy.data()), &one_i, &one_i, desca,
                &vl, &vu, &one_i, &n_eig,
                &abstol,
                &m, &nz,
                w,
                &orfac,
                reinterpret_cast<std::complex<double>*>(z), &one_i, &one_i, descz,
                reinterpret_cast<std::complex<double>*>(work.data()), &lwork,
                rwork.data(), &lrwork, iwork.data(), &liwork,
                ifail.data(), iclustr.data(), gap.data(),
                &info
            );
        };
    } else {
        eigen = [&]() {
            pdsyevx_(
                "V", &range, "U",
                &n_glb,
                reinterpret_cast<double*>(A_loc_copy.data()), &one_i, &one_i, desca,
                &vl, &vu, &one_i, &n_eig,
                &abstol,
                &m, &nz,
                w,
                &orfac,
                reinterpret_cast<double*>(z), &one_i, &one_i, descz,
                reinterpret_cast<double*>(work.data()), &lwork, iwork.data(), &liwork,
                ifail.data(), iclustr.data(), gap.data(),
                &info
            );
        };
    }

    // the first call is a work space query
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

    // Ideally the calculation will be successfully carried out in the next call.
    // In case there is any fatal error, the function will throw an exception
    // and abort the program.
    //
    // In some cases, calculations may finish with eigenvectors unorthogonalized
    // due to insufficient workspace. In this case, we will increase the size of
    // the workspace and re-run the calculation. This, however, should happen
    // a few times at most and we choose to abort the program if the calculation
    // fails after a few trials.
    const int n_trial_max = 3;

    for (int i_trial = 0; i_trial < n_trial_max; ++i_trial) {

        // actual diagonalization
        eigen();

        if (info == 0) {
            return;
        }

        std::string solver = is_cplx ? "pzheevx" : "pdsyevx";
        std::string location = "file " __FILE__ " line "  + std::to_string(__LINE__);

        // The following info aborts the program:
        //      info < 0: illegal input
        //      info % 2 != 0: eigenvectors failed to converge
        //      (info / 4) % 2 != 0: space limit (not supposed to happen in this function)
        //      (info / 8) % 2 != 0: fails to compute eigenvalues
        //
        // The following info triggers a re-run:
        //      (info / 2) % 2 != 0: eigenvectors unorthogonalized

        if (info < 0) {
            info = -info;
            if (info > 100) {
                throw std::runtime_error(location + " the " + std::to_string(info % 100)
                        + "-th entry in the " + std::to_string(info / 100)
                        + "-th argument of " + solver + "is illegal.\n");
            } else {
                throw std::runtime_error(location + " the " + std::to_string(info)
                        + "-th argument of " + solver + "is illegal.\n");
            }
        } 

        if (info % 2) {
            std::string ifail_str = "ifail = ";
            for (auto it = ifail.begin(); it != ifail.end(); ++it) {
                if (*it != 0) {
                    ifail_str += std::to_string(*it) + " ";
                }
            }
            throw std::runtime_error(location + "one or more eigenvectors failed to converge:"
                    + ifail_str);
        }

        if (info / 4 % 2) {
            throw std::runtime_error(location + " space limit (should never occur since the \
                function is called with index-based range selection)");
        }

        if (info / 8 % 2) {
            throw std::runtime_error(location + " fails to compute eigenvalues");
        }

        if (info / 2 % 2) {

            std::copy(A_loc, A_loc + nelem_A_loc, A_loc_copy.data());

            // find the largest cluster
            int cluster_size = 1; 
            for (int i = 0; i < nprocs; ++i) {
                cluster_size = std::max(cluster_size, iclustr[2 * i + 1] - iclustr[2 * i] + 1);
            }

            // increase the size of the workspace
            if (is_cplx) {
                lrwork += (cluster_size - 1) * n_glb;
                rwork.resize(lrwork);
            } else {
                lwork += cluster_size * n_glb;
                work.resize(lwork);
            }

            continue;
        }

        throw std::runtime_error(location + " unknown info");

    } // end of while

    throw std::runtime_error("pdiagx fails after " + std::to_string(n_trial_max) + " trials.\n");
}


template <typename T>
int canon_diag(
    int nrow_loc, int ncol_loc,
    int n_glb, int n_eig,
    T* H, T* S, const int* desc_HS,
    double* E, T* C, const int* desc_C,
    double thr = 1e-5
) {
    /**
     * Solve a generalized eigenvalue problem H*C = S*C*E with canonical orthogonalization.
     *
     * Given a potentially singular generalized eigenvalue problem (i.e., the basis
     * are almost linearly dependent so S might has tiny eigenvalues), this function
     *
     * 1. diagonalizes S
     * 2. selects the subset of eigenvectors, denote X, corresponding to non-tiny eigenvalues
     * 3. rescales X column-wise by diag(1/sqrt(val_S))
     * 4. forms subspace Hamiltonian H_sub = X^H * H * X (^H denotes Hermitian conjugate)
     * 5. solves the eigenvalue problem H_sub * C_sub = C_sub * E
     * 6. gets the eigenvectors of the original problem C = X * C_sub
     *
     */

    static_assert(std::is_same<T, double>::value ||
                  std::is_same<T, std::complex<double>>::value,
                  "T must be double or std::complex<double>");

    bool is_cplx = std::is_same<T, std::complex<double>>::value;

    // number of elements in the local matrix of H/S/vec_S
    size_t nelem_loc = static_cast<size_t>(nrow_loc) * ncol_loc;

    // BLACS context
    int ctxt = desc_HS[1];

    // size of the square block in block-cyclic distribution
    int nb = desc_HS[4];

    //====== allocate memory for temporary arrays ======
    // vec_S (eigenvectors of S, including X)   <-- nelem_loc
    // H_sub                                    <-- at most nelem_loc
    // X^H * H, C_sub (eigenvectors of H_sub)   <-- at most nelem_loc
    // val_S (eigenvalues of S)                 <-- n_glb, always real
    size_t buf_size = 3 * nelem_loc + (n_glb + 1) * sizeof(double) / sizeof(T);
    std::vector<T> buffer(buf_size);

    // X^H * H * X;
    T* vec_S = buffer.data();

    // eigenvectors of S (becomes X after rescaled by 1/sqrt(val_S))
    T* H_sub = vec_S + nelem_loc;

    // X^H * H
    T* XhH = H_sub + nelem_loc;

    // eigenvectors of H_sub; NOTE: C_sub and XhH do not coexist
    T* C_sub = XhH;

    // eigenvalues of S
    double* val_S = reinterpret_cast<double*>(C_sub + nelem_loc);


    //====== 1. full eigen-decomposition of S ======
    pdiagx(n_glb, S, desc_HS, val_S, vec_S, desc_HS);


    //====== 2. find the number of tiny eigenvalues below thr ======
    // number of tiny eigenvalues of S below thr
    int n_tiny = std::find_if(val_S, val_S + n_glb, [thr](double x) { return x > thr; }) - val_S;

    // the "true dimension" of the eigenvalue problem (linear dependency removed)
    int dim = n_glb - n_tiny;

    //======= 3. transformation matrix of canonical orthogonalization ======
    // block-cyclic distribution of H/S/vec_S
    Parallel_2D p2d_HS;
    p2d_HS.set(n_glb, n_glb, nb, ctxt);

    // rescale U column-wise by diag(1/sqrt(o)) (tiny eigenvalues are excluded)
    for (int col_glb = n_tiny; col_glb < n_glb; ++col_glb) {
        if (p2d_HS.global2local_col(col_glb) != -1) { // if the column is in the local matrix
            // we do an in-place scaling of vec_S
            int col_loc = p2d_HS.global2local_col(col_glb);
            double inv_sqrt = 1.0 / std::sqrt(val_S[col_glb]);
            for (int row_loc = 0; row_loc < nrow_loc; ++row_loc) {
                vec_S[row_loc + col_loc * nrow_loc] *= inv_sqrt;
            }
        }
    }

    //====== canonical orthogonalization matrix X = vec_S[:, n_tiny:] ======

    //======= 4. form H_sub = X^H * H * X ======
    // block-cyclic distribution of H_sub/C_sub
    Parallel_2D p2d_sub;
    p2d_sub.set(dim, dim, nb, ctxt);

    // block-cyclic distribution of X^H * H
    Parallel_2D p2d_XhH;
    p2d_XhH.set(dim, n_glb, nb, ctxt);

    const T one_f = 1.0;
    const T zero_f = 0.0;
    const int one_i = 1;
    int icol = 1 + n_tiny; // the first column of U to be used (fortran convention)

    // X^H * H
    ScalapackConnector::gemm(
        is_cplx ? 'C':'T', 'N',
        dim, n_glb, n_glb,
        one_f,
        vec_S, one_i, icol, desc_HS,
        H, one_i, one_i, desc_HS,
        zero_f,
        XhH, one_i, one_i, p2d_XhH.desc
    );

    // H_sub = X^H * H * X
    ScalapackConnector::gemm(
        'N', 'N',
        dim, dim, n_glb,
        one_f,
        XhH, one_i, one_i, p2d_XhH.desc,
        vec_S, one_i, icol, desc_HS,
        zero_f,
        H_sub, one_i, one_i, p2d_sub.desc
    );

    //======= 5. eigen-decomposition of H_sub ======
    // NOTE: the documentation of pdsyevx/pzheevx suggests that the array for
    // holding eigenvectors (C_sub) be square, even if only a selected range of
    // eigenpairs is requested. This is checked by its array descriptor.
    // It might be sufficient to pass a descriptor of a square-sized matrix
    // but allocate a smaller memory for C_sub which is just enough for holding
    // the eigenvectors of interest, but whether this is safe is not clear.
    pdiagx(n_eig, H_sub, p2d_sub.desc, E, C_sub, p2d_sub.desc);

    //======= transform the eigenvectors back ======
    // C = X * C_sub
    ScalapackConnector::gemm(
        'N', 'N',
        n_glb, n_eig, dim,
        one_f,
        vec_S, one_i, icol, desc_HS,
        C_sub, one_i, one_i, p2d_sub.desc,
        zero_f,
        C, one_i, one_i, desc_C
    );

    return n_tiny;
}

    template<>
    void DiagoScalapack<double>::diag(hamilt::Hamilt<double>* phm_in, psi::Psi<double>& psi, Real* eigenvalue_in)
{
    ModuleBase::TITLE("DiagoScalapack", "diag");
    matd h_mat, s_mat;
    phm_in->matrix(h_mat, s_mat);
    assert(h_mat.col == s_mat.col && h_mat.row == s_mat.row && h_mat.desc == s_mat.desc);
    std::vector<double> eigen(GlobalV::NLOCAL, 0.0);

    //this->pdsygvx_diag(h_mat.desc, h_mat.col, h_mat.row, h_mat.p, s_mat.p, eigen.data(), psi);
    canon_diag(
        h_mat.row, h_mat.col,
        GlobalV::NLOCAL, GlobalV::NBANDS,
        h_mat.p, s_mat.p, h_mat.desc,
        eigen.data(), psi.get_pointer(), h_mat.desc
    );
    
        const int inc = 1;
    BlasConnector::copy(GlobalV::NBANDS, eigen.data(), inc, eigenvalue_in, inc);
}
    template<>
    void DiagoScalapack<std::complex<double>>::diag(hamilt::Hamilt<std::complex<double>>* phm_in, psi::Psi<std::complex<double>>& psi, Real* eigenvalue_in)
{
    ModuleBase::TITLE("DiagoScalapack", "diag");
    matcd h_mat, s_mat;
    phm_in->matrix(h_mat, s_mat);
    assert(h_mat.col == s_mat.col && h_mat.row == s_mat.row && h_mat.desc == s_mat.desc);
    std::vector<double> eigen(GlobalV::NLOCAL, 0.0);

    //this->pzhegvx_diag(h_mat.desc, h_mat.col, h_mat.row, h_mat.p, s_mat.p, eigen.data(), psi);
    canon_diag(
        h_mat.row, h_mat.col,
        GlobalV::NLOCAL, GlobalV::NBANDS,
        h_mat.p, s_mat.p, h_mat.desc,
        eigen.data(), psi.get_pointer(), h_mat.desc
    );

    const int inc = 1;
    BlasConnector::copy(GlobalV::NBANDS, eigen.data(), inc, eigenvalue_in, inc);
}

#ifdef __MPI
 template<>
    void DiagoScalapack<double>::diag_pool(hamilt::MatrixBlock<double>& h_mat,
    hamilt::MatrixBlock<double>& s_mat,
    psi::Psi<double>& psi,
    Real* eigenvalue_in,
    MPI_Comm& comm)
{
    ModuleBase::TITLE("DiagoScalapack", "diag_pool");
    assert(h_mat.col == s_mat.col && h_mat.row == s_mat.row && h_mat.desc == s_mat.desc);
    std::vector<double> eigen(GlobalV::NLOCAL, 0.0);
    this->pdsygvx_diag(h_mat.desc, h_mat.col, h_mat.row, h_mat.p, s_mat.p, eigen.data(), psi);
    const int inc = 1;
    BlasConnector::copy(GlobalV::NBANDS, eigen.data(), inc, eigenvalue_in, inc);
}
    template<>
    void DiagoScalapack<std::complex<double>>::diag_pool(hamilt::MatrixBlock<std::complex<double>>& h_mat,
    hamilt::MatrixBlock<std::complex<double>>& s_mat,
    psi::Psi<std::complex<double>>& psi,
    Real* eigenvalue_in,
    MPI_Comm& comm)
{
    ModuleBase::TITLE("DiagoScalapack", "diag_pool");
    assert(h_mat.col == s_mat.col && h_mat.row == s_mat.row && h_mat.desc == s_mat.desc);
    std::vector<double> eigen(GlobalV::NLOCAL, 0.0);
    this->pzhegvx_diag(h_mat.desc, h_mat.col, h_mat.row, h_mat.p, s_mat.p, eigen.data(), psi);
    const int inc = 1;
    BlasConnector::copy(GlobalV::NBANDS, eigen.data(), inc, eigenvalue_in, inc);
}
#endif

    template<typename T>
    std::pair<int, std::vector<int>> DiagoScalapack<T>::pdsygvx_once(const int* const desc,
                                                         const int ncol,
                                                         const int nrow,
                                                         const double *const h_mat,
                                                         const double *const s_mat,
                                                         double *const ekb,
                                                         psi::Psi<double> &wfc_2d) const
{
    ModuleBase::matrix h_tmp(ncol, nrow, false);
    memcpy(h_tmp.c, h_mat, sizeof(double) * ncol * nrow);
    ModuleBase::matrix s_tmp(ncol, nrow, false);
    memcpy(s_tmp.c, s_mat, sizeof(double) * ncol * nrow);

    const char jobz = 'V', range = 'I', uplo = 'U';
    const int itype = 1, il = 1, iu = GlobalV::NBANDS, one = 1;
    int M = 0, NZ = 0, lwork = -1, liwork = -1, info = 0;
    double vl = 0, vu = 0;
    const double abstol = 0, orfac = -1;
    std::vector<double> work(3, 0);
    std::vector<int> iwork(1, 0);
    std::vector<int> ifail(GlobalV::NLOCAL, 0);
    std::vector<int> iclustr(2 * GlobalV::DSIZE);
    std::vector<double> gap(GlobalV::DSIZE);

    pdsygvx_(&itype,
             &jobz,
             &range,
             &uplo,
             &GlobalV::NLOCAL,
             h_tmp.c,
             &one,
             &one,
             desc,
             s_tmp.c,
             &one,
             &one,
             desc,
             &vl,
             &vu,
             &il,
             &iu,
             &abstol,
             &M,
             &NZ,
             ekb,
             &orfac,
             wfc_2d.get_pointer(),
             &one,
             &one,
             desc,
             work.data(),
             &lwork,
             iwork.data(),
             &liwork,
             ifail.data(),
             iclustr.data(),
             gap.data(),
             &info);
    if (info) {
        throw std::runtime_error("info = " + ModuleBase::GlobalFunc::TO_STRING(info) + ".\n"
                                 + ModuleBase::GlobalFunc::TO_STRING(__FILE__) + " line "
                                 + ModuleBase::GlobalFunc::TO_STRING(__LINE__));
}

    //	GlobalV::ofs_running<<"lwork="<<work[0]<<"\t"<<"liwork="<<iwork[0]<<std::endl;
    lwork = work[0];
    work.resize(std::max(lwork,3), 0);
    liwork = iwork[0];
    iwork.resize(liwork, 0);

    pdsygvx_(&itype,
             &jobz,
             &range,
             &uplo,
             &GlobalV::NLOCAL,
             h_tmp.c,
             &one,
             &one,
             desc,
             s_tmp.c,
             &one,
             &one,
             desc,
             &vl,
             &vu,
             &il,
             &iu,
             &abstol,
             &M,
             &NZ,
             ekb,
             &orfac,
             wfc_2d.get_pointer(),
             &one,
             &one,
             desc,
             work.data(),
             &lwork,
             iwork.data(),
             &liwork,
             ifail.data(),
             iclustr.data(),
             gap.data(),
             &info);
    //	GlobalV::ofs_running<<"M="<<M<<"\t"<<"NZ="<<NZ<<std::endl;

    if (info == 0) {
        return std::make_pair(info, std::vector<int>{});
    } else if (info < 0) {
        return std::make_pair(info, std::vector<int>{});
    } else if (info % 2) {
        return std::make_pair(info, ifail);
    } else if (info / 2 % 2) {
        return std::make_pair(info, iclustr);
    } else if (info / 4 % 2) {
        return std::make_pair(info, std::vector<int>{M, NZ});
    } else if (info / 16 % 2) {
        return std::make_pair(info, ifail);
    } else {
        throw std::runtime_error("info = " + ModuleBase::GlobalFunc::TO_STRING(info) + ".\n"
                                 + ModuleBase::GlobalFunc::TO_STRING(__FILE__) + " line "
                                 + ModuleBase::GlobalFunc::TO_STRING(__LINE__));
}
}
    template<typename T>
    std::pair<int, std::vector<int>> DiagoScalapack<T>::pzhegvx_once(const int* const desc,
                                                         const int ncol,
                                                         const int nrow,
                                                         const std::complex<double> *const h_mat,
                                                         const std::complex<double> *const s_mat,
                                                         double *const ekb,
                                                         psi::Psi<std::complex<double>> &wfc_2d) const
{
    ModuleBase::ComplexMatrix h_tmp(ncol, nrow, false);
    memcpy(h_tmp.c, h_mat, sizeof(std::complex<double>) * ncol * nrow);
    ModuleBase::ComplexMatrix s_tmp(ncol, nrow, false);
    memcpy(s_tmp.c, s_mat, sizeof(std::complex<double>) * ncol * nrow);

    const char jobz = 'V', range = 'I', uplo = 'U';
    const int itype = 1, il = 1, iu = GlobalV::NBANDS, one = 1;
    int M = 0, NZ = 0, lwork = -1, lrwork = -1, liwork = -1, info = 0;
    const double abstol = 0, orfac = -1;
    //Note: pzhegvx_ has a bug
    //      We must give vl,vu a value, although we do not use range 'V'
    //      We must give rwork at least a memory of sizeof(double) * 3
    const double vl = 0, vu = 0;
    std::vector<std::complex<double>> work(1, 0);
    std::vector<double> rwork(3, 0);
    std::vector<int> iwork(1, 0);
    std::vector<int> ifail(GlobalV::NLOCAL, 0);
    std::vector<int> iclustr(2 * GlobalV::DSIZE);
    std::vector<double> gap(GlobalV::DSIZE);

    pzhegvx_(&itype,
             &jobz,
             &range,
             &uplo,
             &GlobalV::NLOCAL,
             h_tmp.c,
             &one,
             &one,
             desc,
             s_tmp.c,
             &one,
             &one,
             desc,
             &vl,
             &vu,
             &il,
             &iu,
             &abstol,
             &M,
             &NZ,
             ekb,
             &orfac,
             wfc_2d.get_pointer(),
             &one,
             &one,
             desc,
             work.data(),
             &lwork,
             rwork.data(),
             &lrwork,
             iwork.data(),
             &liwork,
             ifail.data(),
             iclustr.data(),
             gap.data(),
             &info);
    if (info) {
        throw std::runtime_error("info=" + ModuleBase::GlobalFunc::TO_STRING(info) + ". "
                                 + ModuleBase::GlobalFunc::TO_STRING(__FILE__) + " line "
                                 + ModuleBase::GlobalFunc::TO_STRING(__LINE__));
}

    //	GlobalV::ofs_running<<"lwork="<<work[0]<<"\t"<<"lrwork="<<rwork[0]<<"\t"<<"liwork="<<iwork[0]<<std::endl;
    lwork = work[0].real();
    work.resize(lwork, 0);
    lrwork = rwork[0] + this->degeneracy_max * GlobalV::NLOCAL;
    int maxlrwork = std::max(lrwork,3);
    rwork.resize(maxlrwork, 0);
    liwork = iwork[0];
    iwork.resize(liwork, 0);

    pzhegvx_(&itype,
             &jobz,
             &range,
             &uplo,
             &GlobalV::NLOCAL,
             h_tmp.c,
             &one,
             &one,
             desc,
             s_tmp.c,
             &one,
             &one,
             desc,
             &vl,
             &vu,
             &il,
             &iu,
             &abstol,
             &M,
             &NZ,
             ekb,
             &orfac,
             wfc_2d.get_pointer(),
             &one,
             &one,
             desc,
             work.data(),
             &lwork,
             rwork.data(),
             &lrwork,
             iwork.data(),
             &liwork,
             ifail.data(),
             iclustr.data(),
             gap.data(),
             &info);
    //	GlobalV::ofs_running<<"M="<<M<<"\t"<<"NZ="<<NZ<<std::endl;
    
    if (info == 0) {
        return std::make_pair(info, std::vector<int>{});
    } else if (info < 0) {
        return std::make_pair(info, std::vector<int>{});
    } else if (info % 2) {
        return std::make_pair(info, ifail);
    } else if (info / 2 % 2) {
        return std::make_pair(info, iclustr);
    } else if (info / 4 % 2) {
        return std::make_pair(info, std::vector<int>{M, NZ});
    } else if (info / 16 % 2) {
        return std::make_pair(info, ifail);
    } else {
        throw std::runtime_error("info = " + ModuleBase::GlobalFunc::TO_STRING(info) + ".\n"
                                 + ModuleBase::GlobalFunc::TO_STRING(__FILE__) + " line "
                                 + ModuleBase::GlobalFunc::TO_STRING(__LINE__));
}
}
    template<typename T>
    void DiagoScalapack<T>::pdsygvx_diag(const int* const desc,
                             const int ncol,
                             const int nrow,
                             const double *const h_mat,
                             const double *const s_mat,
                             double *const ekb,
                             psi::Psi<double> &wfc_2d)
{
    while (true)
    {
        const std::pair<int, std::vector<int>> info_vec = pdsygvx_once(desc, ncol, nrow, h_mat, s_mat, ekb, wfc_2d);
        post_processing(info_vec.first, info_vec.second);
        if (info_vec.first == 0) {
            break;
}
    }
}

    template<typename T>
    void DiagoScalapack<T> ::pzhegvx_diag(const int* const desc,
                             const int ncol,
                             const int nrow,
                             const std::complex<double> *const h_mat,
                             const std::complex<double> *const s_mat,
                             double *const ekb,
                             psi::Psi<std::complex<double>> &wfc_2d)
{
    while (true)
    {
        const std::pair<int, std::vector<int>> info_vec = pzhegvx_once(desc, ncol, nrow, h_mat, s_mat, ekb, wfc_2d);
        post_processing(info_vec.first, info_vec.second);
        if (info_vec.first == 0) {
            break;
}
    }
}

    template<typename T>
    void DiagoScalapack<T>::post_processing(const int info, const std::vector<int>& vec)
{
    const std::string str_info = "info = " + ModuleBase::GlobalFunc::TO_STRING(info) + ".\n";
    const std::string str_FILE
        = ModuleBase::GlobalFunc::TO_STRING(__FILE__) + " line " + ModuleBase::GlobalFunc::TO_STRING(__LINE__) + ".\n";
    const std::string str_info_FILE = str_info + str_FILE;

    if (info == 0)
    {
        return;
    }
    else if (info < 0)
    {
        const int info_negative = -info;
        const std::string str_index
            = (info_negative > 100)
                  ? ModuleBase::GlobalFunc::TO_STRING(info_negative / 100) + "-th argument "
                        + ModuleBase::GlobalFunc::TO_STRING(info_negative % 100) + "-entry is illegal.\n"
                  : ModuleBase::GlobalFunc::TO_STRING(info_negative) + "-th argument is illegal.\n";
        throw std::runtime_error(str_info_FILE + str_index);
    }
    else if (info % 2)
    {
        std::string str_ifail = "ifail = ";
        for (const int i: vec) {
            str_ifail += ModuleBase::GlobalFunc::TO_STRING(i) + " ";
}
        throw std::runtime_error(str_info_FILE + str_ifail);
    }
    else if (info / 2 % 2)
    {
        int degeneracy_need = 0;
        for (int irank = 0; irank < GlobalV::DSIZE; ++irank) {
            degeneracy_need = std::max(degeneracy_need, vec[2 * irank + 1] - vec[2 * irank]);
}
        const std::string str_need = "degeneracy_need = " + ModuleBase::GlobalFunc::TO_STRING(degeneracy_need) + ".\n";
        const std::string str_saved
            = "degeneracy_saved = " + ModuleBase::GlobalFunc::TO_STRING(this->degeneracy_max) + ".\n";
        if (degeneracy_need <= this->degeneracy_max)
        {
            throw std::runtime_error(str_info_FILE + str_need + str_saved);
        }
        else
        {
            GlobalV::ofs_running << str_need << str_saved;
            this->degeneracy_max = degeneracy_need;
            return;
        }
    }
    else if (info / 4 % 2)
    {
        const std::string str_M = "M = " + ModuleBase::GlobalFunc::TO_STRING(vec[0]) + ".\n";
        const std::string str_NZ = "NZ = " + ModuleBase::GlobalFunc::TO_STRING(vec[1]) + ".\n";
        const std::string str_NBANDS
            = "GlobalV::NBANDS = " + ModuleBase::GlobalFunc::TO_STRING(GlobalV::NBANDS) + ".\n";
        throw std::runtime_error(str_info_FILE + str_M + str_NZ + str_NBANDS);
    }
    else if (info / 16 % 2)
    {
        const std::string str_npos = "not positive definite = " + ModuleBase::GlobalFunc::TO_STRING(vec[0]) + ".\n";
        throw std::runtime_error(str_info_FILE + str_npos);
    }
    else
    {
        throw std::runtime_error(str_info_FILE);
    }
}


} // namespace hsolver
