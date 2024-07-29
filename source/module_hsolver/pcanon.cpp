#include <cstdio>
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
        double* work, const int* lwork, int* iwork, int* liwork,
        int* ifail, int* icluster, double* gap,
        int* info
    );

    double pdlamch_(int* ictxt, const char* cmach);
}


// diagonalization returning first nbands eigenpairs
void pdiagx(int n, int nbands, double* A, int* desca, double* w, double* z, int* descz) {

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
    std::vector<int> iwork(3);

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


int pcanon_eig(
    int nrow_loc, int ncol_loc,
    int n_glb, int nbands,
    double* H, double* S, int* desc,
    double* val, double* vec, int* descv,
    double thr = 1e-6
) {
    /**
     * Solve a generalized eigenvalue problem HC = SCE by canonical
     * orthogonalization with eigenvalue filtering.
     *
     * Given a potentially singular generalized eigenvalue problem (i.e., the basis
     * are almost linearly dependent so S has tiny eigenvalues), one can:
     *
     * 1. diagonalize S = U * diag(o) * U^T
     * 2. select the subset of eigenvectors, denote X, corresponding to non-tiny eigenvalues
     * 3. rescale X column-wise by diag(1/sqrt(o))
     * 4. form subspace Hamiltonian H_sub = X^T * H * X
     * 5. solve the eigenvalue problem H_sub C_sub = C_sub E
     * 6. get the eigenvectors of the original problem C = X * C_sub
     *
     */

    //====== eigenvalue decomposition of S ======
    // make a copy of S because pdiagx (pdsyevx) will overwrite it
    std::vector<double> S_copy(S, S + nrow_loc * ncol_loc);

    // eigenvalues and eigenvectors: S = U * diag(o) * U^T
    std::vector<double> U(nrow_loc * ncol_loc);
    std::vector<double> o(n_glb);

    pdiagx(n_glb, n_glb, S_copy.data(), desc, o.data(), U.data(), desc);
    std::vector<double>().swap(S_copy); // release memory

    // number of tiny eigenvalues of S below thr
    int n_tiny = std::find_if(o.begin(), o.end(), [thr](double x) { return x > thr; }) - o.begin();

    // the "true dimension" of the eigenvalue problem (with linear dependency removed)
    int dim = n_glb - n_tiny;

    //======= transformation matrix of canonical orthogonalization ======
    int blacs_ctxt = desc[1];
    int nb = desc[4]; // square block is assumed
    Parallel_2D p2d_U;
    p2d_U.set(n_glb, n_glb, nb, blacs_ctxt);

    // rescale U column-wise by diag(1/sqrt(o)) (tiny eigenvalues are excluded)
    for (int col_glb = n_tiny; col_glb < n_glb; ++col_glb) {
        if (p2d_U.global2local_col(col_glb) != -1) { // the column is in the local matrix
            int col_loc = p2d_U.global2local_col(col_glb);
            double inv_sqrt = 1.0 / std::sqrt(o[col_glb]);
            for (int row_loc = 0; row_loc < nrow_loc; ++row_loc) {
                U[row_loc + col_loc * nrow_loc] *= inv_sqrt;
            }
        }
    }
    std::vector<double>().swap(o);

    // from now on X = U[:, n_tiny:]

    //======= build and diagonalize the transformed Hc = X^T H X ======
    Parallel_2D p2d_sub;
    p2d_sub.set(dim, dim, nb, blacs_ctxt);
    std::vector<double> Hc(p2d_sub.nrow * p2d_sub.ncol);

    Parallel_2D p2d_tmp;
    p2d_tmp.set(dim, n_glb, nb, blacs_ctxt);
    std::vector<double> tmp(p2d_tmp.nrow * p2d_tmp.ncol);

    const double one_d = 1.0;
    const double zero_d = 0.0;
    const int one_i = 1;
    int icol = 1 + n_tiny; // the first column of U to be used (fortran convention)

    // X^T * H
    pdgemm_(
        "T", "N",
        &dim, &n_glb, &n_glb,
        &one_d,
        U.data(), &one_i, &icol, desc,
        H, &one_i, &one_i, desc,
        &zero_d,
        tmp.data(), &one_i, &one_i, p2d_tmp.desc
    );

    // Hc = X^T * H * X
    pdgemm_(
        "N", "N",
        &dim, &dim, &n_glb,
        &one_d,
        tmp.data(), &one_i, &one_i, p2d_tmp.desc,
        U.data(), &one_i, &icol, desc,
        &zero_d,
        Hc.data(), &one_i, &one_i, p2d_sub.desc
    );
    std::vector<double>().swap(tmp);

    // first nbands eigenvectors of Hc
    // NOTE: pdsyevx requires that the array for holding eigenvectors be square,
    // even if only a selected range of eigenpairs is requested.
    // It might be sufficient to allocate a smaller memory and provide a descriptor
    // of the square-sized matrix, but whether this is safe or not is not clear.
    std::vector<double> vtmp(p2d_sub.nrow * p2d_sub.ncol);

    pdiagx(dim, nbands, Hc.data(), p2d_sub.desc, val, vtmp.data(), p2d_sub.desc);
    std::vector<double>().swap(Hc);

    //======= transform the eigenvectors back ======
    // V = X * V_tmp
    pdgemm_(
        "N", "N",
        &n_glb, &nbands, &dim,
        &one_d,
        U.data(), &one_i, &icol, desc,
        vtmp.data(), &one_i, &one_i, p2d_sub.desc,
        &zero_d,
        vec, &one_i, &one_i, descv
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

    pcanon_eig(
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

    MPI_Finalize();
    return 0;
}

