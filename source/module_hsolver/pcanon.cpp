#include <cstdio>
#include <type_traits>
#include <vector>
#include <cmath>
#include <algorithm>
#include <unistd.h>

#include "module_base/scalapack_connector.h"
#include "module_basis/module_ao/parallel_2d.h"


extern "C" {

    void pdsyevx_(
        const char* jobz, const char* range, const char* uplo,
        const int* n,
        double* a, const int* ia, const int* ja, const int* desca,
        const double* vl, const double* vu, const int* il, const int* iu,
        const double* abstol,
        int* m, int* nz,
        double* w,
        double* orfac,
        double* z, const int* iz, const int* jz, const int* descz,
        double* work, const int* lwork, int* iwork, const int* liwork,
        int* ifail, int* icluster, double* gap,
        int* info
    );

    void pzheevx_(
        const char* jobz, const char* range, const char* uplo,
        const int* n,
        std::complex<double>* a, const int* ia, const int* ja, const int* desca,
        const double* vl, const double* vu, const int* il, const int* iu,
        const double* abstol,
        int* m, int* nz,
        double* w,
        double* orfac,
        std::complex<double>* z, const int* iz, const int* jz, const int* descz,
        std::complex<double>* work, const int* lwork, double* rwork, const int* lrwork, int* iwork, const int* liwork,
        int* ifail, int* icluster, double* gap,
        int* info
    );

    double pdlamch_(int* ictxt, const char* cmach);
}

// pdgemm & pzgemm have the same interface, so we unify them
// and let overload resolution to choose the right one
void pxgemm(
		const char *transa, const char *transb,
		const int *M, const int *N, const int *K,
		const double *alpha,
		const double *A, const int *IA, const int *JA, const int *DESCA,
		const double *B, const int *IB, const int *JB, const int *DESCB,
		const double *beta,
		double *C, const int *IC, const int *JC, const int *DESCC)
{
    pdgemm_(transa, transb, M, N, K, alpha, A, IA, JA, DESCA, B, IB, JB, DESCB, beta, C, IC, JC, DESCC);
}

void pxgemm(
		const char *transa, const char *transb,
		const int *M, const int *N, const int *K,
		const std::complex<double> *alpha,
		const std::complex<double> *A, const int *IA, const int *JA, const int *DESCA,
		const std::complex<double> *B, const int *IB, const int *JB, const int *DESCB,
		const std::complex<double> *beta,
		std::complex<double> *C, const int *IC, const int *JC, const int *DESCC)
{
    pzgemm_(transa, transb, M, N, K, alpha, A, IA, JA, DESCA, B, IB, JB, DESCB, beta, C, IC, JC, DESCC);
}


// NOTE pdsyevx and pzheevx have different interfaces, so we have to write two pdiagx
void pdiagx(int n, int nbands, double* A, int* desca, double* w, double* z, int* descz) {
    // diagonalization returning the first `nbands` eigenpairs

    // get number of procs involved in this diagonalization
    // relevant for the output parameters iclustr and gap
    int blacs_ctxt = desca[1];
    int nprow, npcol, myrow, mycol, nprocs;
    Cblacs_gridinfo(blacs_ctxt, &nprow, &npcol, &myrow, &mycol);
    nprocs = nprow * npcol;

    // input parameters
    char range = (nbands == n) ? 'A' : 'I';
    double vl = 0.0, vu = 1.0;
    double abstol = pdlamch_(&blacs_ctxt, "U");
    int m = 0, nz = 0;
    double orfac = -1.0;

    // work space
    int lwork = -1, liwork = -1;
    std::vector<double> work(3);
    std::vector<int> iwork(1);

    // output informational parameters
    std::vector<int> ifail(n);
    std::vector<int> iclustr(2 * nprocs);
    std::vector<double> gap(nprocs);
    int info = 0;

    const int one = 1;

    // work space query & allocation
    pdsyevx_(
        "V", &range, "U",
        &n,
        A, &one, &one, desca,
        &vl, &vu, &one, &nbands,
        &abstol,
        &m, &nz,
        w,
        &orfac,
        z, &one, &one, descz,
        work.data(), &lwork, iwork.data(), &liwork,
        ifail.data(), iclustr.data(), gap.data(),
        &info
    );

    lwork = work[0];
    work.resize(std::max(lwork, 3), 0);
    liwork = iwork[0];
    iwork.resize(liwork, 0);

    // actual diagonalization
    pdsyevx_(
        "V", &range, "U",
        &n, A, &one, &one, desca,
        &vl, &vu, &one, &nbands,
        &abstol,
        &m, &nz,
        w,
        &orfac,
        z, &one, &one, descz,
        work.data(), &lwork, iwork.data(), &liwork,
        ifail.data(), iclustr.data(), gap.data(),
        &info
    );
}


void pdiagx(int n, int nbands, std::complex<double>* A, int* desca, double* w, std::complex<double>* z, int* descz) {
    // diagonalization returning the first `nbands` eigenpairs

    // get number of procs involved in this diagonalization
    // relevant for the output parameters iclustr and gap
    int blacs_ctxt = desca[1];
    int nprow, npcol, myrow, mycol, nprocs;
    Cblacs_gridinfo(blacs_ctxt, &nprow, &npcol, &myrow, &mycol);
    nprocs = nprow * npcol;

    // input parameters
    char range = (nbands == n) ? 'A' : 'I';
    double vl = 0.0, vu = 1.0;
    double abstol = pdlamch_(&blacs_ctxt, "U");
    int m = 0, nz = 0;
    double orfac = -1.0;

    // work space
    int lwork = -1, lrwork = -1, liwork = -1;
    std::vector<std::complex<double>> work(1);
    std::vector<double> rwork(3);
    std::vector<int> iwork(1);

    // output informational parameters
    std::vector<int> ifail(n);
    std::vector<int> iclustr(2 * nprocs);
    std::vector<double> gap(nprocs);
    int info = 0;

    const int one = 1;

    // work space query & allocation
    pzheevx_(
        "V", &range, "U",
        &n,
        A, &one, &one, desca,
        &vl, &vu, &one, &nbands,
        &abstol,
        &m, &nz,
        w,
        &orfac,
        z, &one, &one, descz,
        work.data(), &lwork, rwork.data(), &lrwork, iwork.data(), &liwork,
        ifail.data(), iclustr.data(), gap.data(),
        &info
    );

    lwork = work[0].real();
    work.resize(lwork, 0);
    lrwork = rwork[0];
    rwork.resize(std::max(lrwork, 3), 0);
    liwork = iwork[0];
    iwork.resize(liwork, 0);

    // actual diagonalization
    pzheevx_(
        "V", &range, "U",
        &n, A, &one, &one, desca,
        &vl, &vu, &one, &nbands,
        &abstol,
        &m, &nz,
        w,
        &orfac,
        z, &one, &one, descz,
        work.data(), &lwork, rwork.data(), &lrwork, iwork.data(), &liwork,
        ifail.data(), iclustr.data(), gap.data(),
        &info
    );
}

template <typename T>
int canon_diag(
    int nrow_loc, int ncol_loc,
    int n_glb, int nbands,
    T* H, T* S, int* desc_HS,
    double* E, T* C, int* desc_C,
    double thr = 1e-6
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

    static_assert(std::is_same<T, double>::value || std::is_same<T, std::complex<double>>::value,
                  "T must be double or std::complex<double>");

    bool is_cplx = std::is_same<T, std::complex<double>>::value;

    // number of elements in the local matrix of H/S/vec_S
    size_t nelem_loc = static_cast<size_t>(nrow_loc) * ncol_loc;

    // BLACS context
    int ctxt = desc_HS[1];

    // size of the square block in block-cyclic distribution
    int nb = desc_HS[4];

    //====== allocate memory for most temporary variables ======
    // S_copy, H_sub                            <-- at most nelem_loc
    // vec_S (eigenvectors of S, including X)   <-- nelem_loc
    // X^H * H, V_sub (eigenvectors of H_sub)   <-- at most nelem_loc
    // val_S (eigenvalues of S)                 <-- n_glb, always real
    size_t buf_size = 3 * nelem_loc + n_glb * sizeof(double) / sizeof(T) + (is_cplx && (n_glb % 2));
    std::vector<T> buffer(buf_size);

    // a copy of S (prevent pdsyevx from destroying the input S)
    T* S_copy = buffer.data();

    // H_sub = X^H * H * X; NOTE: H_sub and S_copy do not coexist
    T* H_sub = S_copy;

    // eigenvectors of S (becomes X after rescaled by 1/sqrt(val_S))
    T* vec_S = H_sub + nelem_loc;

    // X^H * H
    T* XhH = vec_S + nelem_loc;

    // eigenvectors of H_sub; NOTE: C_sub and XhH do not coexist
    T* C_sub = XhH;

    // eigenvalues of S
    double* val_S = reinterpret_cast<double*>(C_sub + nelem_loc);


    // block-cyclic distribution of H/S/vec_S
    Parallel_2D p2d_HS;
    p2d_HS.set(n_glb, n_glb, nb, ctxt);


    //====== 1. eigen-decomposition of S ======
    std::copy(S, S + nelem_loc, S_copy);
    pdiagx(n_glb, n_glb, S_copy, desc_HS, val_S, vec_S, desc_HS);

    //====== 2. find the number of tiny eigenvalues below thr ======
    // number of tiny eigenvalues of S below thr
    int n_tiny = std::find_if(val_S, val_S + n_glb, [thr](double x) { return x > thr; }) - val_S;

    // the "true dimension" of the eigenvalue problem (linear dependency removed)
    int dim = n_glb - n_tiny;

    //======= 3. transformation matrix of canonical orthogonalization ======
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

    // from now on, the canonical transformation matrix X = vec_S[:, n_tiny:]

    // block-cyclic distribution of H_sub/C_sub
    Parallel_2D p2d_sub;
    p2d_sub.set(dim, dim, nb, ctxt);

    // block-cyclic distribution of X^H * H
    Parallel_2D p2d_XhH;
    p2d_XhH.set(dim, n_glb, nb, ctxt);

    //======= 4. form H_sub = X^H * H * X ======
    const T one_f = 1.0;
    const T zero_f = 0.0;
    const int one_i = 1;
    int icol = 1 + n_tiny; // the first column of U to be used (fortran convention)

    // X^H * H
    pxgemm(
        is_cplx ? "C":"T", "N",
        &dim, &n_glb, &n_glb,
        &one_f,
        vec_S, &one_i, &icol, desc_HS,
        H, &one_i, &one_i, desc_HS,
        &zero_f,
        XhH, &one_i, &one_i, p2d_XhH.desc
    );

    // H_sub = X^H * H * X
    pxgemm(
        "N", "N",
        &dim, &dim, &n_glb,
        &one_f,
        XhH, &one_i, &one_i, p2d_XhH.desc,
        vec_S, &one_i, &icol, desc_HS,
        &zero_f,
        H_sub, &one_i, &one_i, p2d_sub.desc
    );

    //======= 5. eigen-decomposition of H_sub ======
    // NOTE: pdsyevx's documentation suggests that the array for holding
    // eigenvectors (C_sub) be square, even if only a selected range of
    // eigenpairs is requested. This is checked by its array descriptor.
    // It might be sufficient to pass a descriptor of a square-sized matrix
    // but allocate a smaller memory for C_sub that's just enough for holding
    // the eigenvectors of interest, but whether this is safe is not clear.
    pdiagx(dim, nbands, H_sub, p2d_sub.desc, E, C_sub, p2d_sub.desc);

    //======= transform the eigenvectors back ======
    // C = X * C_sub
    pxgemm(
        "N", "N",
        &n_glb, &nbands, &dim,
        &one_f,
        vec_S, &one_i, &icol, desc_HS,
        C_sub, &one_i, &one_i, p2d_sub.desc,
        &zero_f,
        C, &one_i, &one_i, desc_C
    );

    return n_tiny;
}


int main() {

    MPI_Init(nullptr, nullptr);

    srand(time(nullptr));

    int id, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &id);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    //====== generate some generalized eigenvalue problem ======
    int rank = 10;
    int n = 12;
    int nbands = 5;
    int nb = 2;

    std::vector<double> H;
    std::vector<double> S, Y;

    if (id == 0) {
        // random symmetric matrix
        H.resize(n * n);
        for (int i = 0; i < n; ++i) {
            for (int j = i; j < n; ++j) {
                H[i + j*n] = H[j + i*n] = 1.0 * rand() / RAND_MAX - 0.5;
            }
        }

        // random rank-deficient overlap matrix S = Y^T * Y
        S.resize(n * n);
        Y.resize(rank * n);
        for (int i = 0; i < rank; ++i) {
            for (int j = 0; j < n; ++j) {
                Y[i + j*rank] = 1.0 * rand() / RAND_MAX - 0.5;
            }
        }

        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                S[i + j*n] = 0.0;
                for (int k = 0; k < rank; ++k) {
                    S[i + j*n] += Y[k + i*rank] * Y[k + j*rank];
                }
            }
        }

        //====== write H and S to H.dat and S.dat ======
        FILE* f = fopen("H.dat", "w");
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                fprintf(f, "% 24.17e ", H[i + j*n]);
            }
            fprintf(f, "\n");
        }
        fclose(f);
    
        f = fopen("S.dat", "w");
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                fprintf(f, "% 24.17e ", S[i + j*n]);
            }
            fprintf(f, "\n");
        }
        fclose(f);
    }
    
    //====== redistribute H and S to all processes ======
    Parallel_2D p2d_HS_glb;
    p2d_HS_glb.init(n, n, n, MPI_COMM_WORLD);

    int blacs_ctxt = p2d_HS_glb.blacs_ctxt;

    Parallel_2D p2d_HS_loc;
    p2d_HS_loc.set(n, n, nb, blacs_ctxt);
    int nrow_loc = p2d_HS_loc.nrow;
    int ncol_loc = p2d_HS_loc.ncol;

    std::vector<double> H_loc, S_loc;
    H_loc.resize(nrow_loc * ncol_loc);
    S_loc.resize(nrow_loc * ncol_loc);

    Cpdgemr2d(n, n, H.data(), 1, 1, p2d_HS_glb.desc, H_loc.data(), 1, 1, p2d_HS_loc.desc, p2d_HS_loc.blacs_ctxt);
    Cpdgemr2d(n, n, S.data(), 1, 1, p2d_HS_glb.desc, S_loc.data(), 1, 1, p2d_HS_loc.desc, p2d_HS_loc.blacs_ctxt);

    //====== solve the generalized eigenvalue problem ======
    std::vector<double> eigval(nbands), eigvec;
    Parallel_2D p2d_vec_loc;
    p2d_vec_loc.set(n, nbands, nb, blacs_ctxt);
    eigvec.resize(p2d_vec_loc.nrow * p2d_vec_loc.ncol);

    canon_diag<double>(
        nrow_loc, ncol_loc, n, nbands,
        H_loc.data(), S_loc.data(), p2d_HS_loc.desc,
        eigval.data(), eigvec.data(), p2d_vec_loc.desc,
        1e-6
    );

    //====== collect to proc-0  ======
    std::vector<double> eigvec_glb;
    Parallel_2D p2d_vec_glb;
    p2d_vec_glb.set(n, nbands, n, blacs_ctxt);
    if (id == 0) {
        eigvec_glb.resize(n * nbands);
    }

    Cpdgemr2d(n, nbands, eigvec.data(), 1, 1, p2d_vec_loc.desc, eigvec_glb.data(), 1, 1, p2d_vec_glb.desc, p2d_vec_glb.blacs_ctxt);


    //====== print the results ======
    if (id == 0) {
        printf("n = %i    nbands = %i\n", n, nbands);
        for (int i = 0; i < nbands; ++i) {
            printf("% 10.2e\n", eigval[i]);
        }
        printf("\n");

        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < nbands; ++j) {
                printf("% 10.2e ", eigvec_glb[i + j*n]);
            }
            printf("\n");
        }
    }

    //=========================================================
    //                  complex version
    //=========================================================
    std::vector<std::complex<double>> Hc;
    std::vector<std::complex<double>> Sc, Yc;

    if (id == 0) {
        // random symmetric matrix
        Hc.resize(n * n);
        for (int i = 0; i < n; ++i) {
            for (int j = i; j < n; ++j) {
                double real = 1.0 * rand() / RAND_MAX - 0.5;
                double imag = (i == j) ? 0.0 : 1.0 * rand() / RAND_MAX - 0.5;
                Hc[i + j*n] = std::complex<double>(real, imag);
                Hc[j + i*n] = std::conj(Hc[i + j*n]);
            }
        }

        // random rank-deficient overlap matrix S = Y^T * Y
        Sc.resize(n * n);
        Yc.resize(rank * n);
        for (int i = 0; i < rank; ++i) {
            for (int j = 0; j < n; ++j) {
                double real = 1.0 * rand() / RAND_MAX - 0.5;
                double imag = 1.0 * rand() / RAND_MAX - 0.5;
                Yc[i + j*rank] = std::complex<double>(real, imag);
            }
        }

        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                Sc[i + j*n] = 0.0;
                for (int k = 0; k < rank; ++k) {
                    Sc[i + j*n] += std::conj(Yc[k + i*rank]) * Yc[k + j*rank];
                }
            }
        }

        //====== write H and S to H.dat and S.dat ======
        FILE* f = fopen("Hc.dat", "w");
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                fprintf(f, "% 24.17e  % 24.17e  ", Hc[i + j*n].real(), Hc[i + j*n].imag());
            }
            fprintf(f, "\n");
        }
        fclose(f);
    
        f = fopen("Sc.dat", "w");
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                fprintf(f, "% 24.17e  % 24.17e  ", Sc[i + j*n].real(), Sc[i + j*n].imag());
            }
            fprintf(f, "\n");
        }
        fclose(f);
    }
    
    //====== redistribute Hc and Sc to all processes ======
    std::vector<std::complex<double>> Hc_loc, Sc_loc;
    Hc_loc.resize(nrow_loc * ncol_loc);
    Sc_loc.resize(nrow_loc * ncol_loc);

    Cpxgemr2d(n, n, Hc.data(), 1, 1, p2d_HS_glb.desc, Hc_loc.data(), 1, 1, p2d_HS_loc.desc, p2d_HS_loc.blacs_ctxt);
    Cpxgemr2d(n, n, Sc.data(), 1, 1, p2d_HS_glb.desc, Sc_loc.data(), 1, 1, p2d_HS_loc.desc, p2d_HS_loc.blacs_ctxt);

    //====== solve the generalized eigenvalue problem ======
    std::vector<std::complex<double>> eigvecc;
    eigvecc.resize(p2d_vec_loc.nrow * p2d_vec_loc.ncol);

    canon_diag(
        nrow_loc, ncol_loc, n, nbands,
        Hc_loc.data(), Sc_loc.data(), p2d_HS_loc.desc,
        eigval.data(), eigvecc.data(), p2d_vec_loc.desc,
        1e-6
    );

    //====== collect to proc-0  ======
    std::vector<std::complex<double>> eigvecc_glb;
    if (id == 0) {
        eigvecc_glb.resize(n * nbands);
    }

    Cpxgemr2d(n, nbands, eigvecc.data(), 1, 1, p2d_vec_loc.desc, eigvecc_glb.data(), 1, 1, p2d_vec_glb.desc, p2d_vec_glb.blacs_ctxt);


    //====== print the results ======
    if (id == 0) {
        printf("n = %i    nbands = %i\n", n, nbands);
        for (int i = 0; i < nbands; ++i) {
            printf("% 10.2e\n", eigval[i]);
        }
        printf("\n");

        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < nbands; ++j) {
                //printf("(% 10.2e,% 10.2e) ", eigvecc_glb[i + j*n].real(), eigvecc_glb[i + j*n].imag());
                printf("% 10.2e ", std::abs(eigvecc_glb[i + j*n]));
            }
            printf("\n");
        }
    }

    MPI_Finalize();
    return 0;
}

