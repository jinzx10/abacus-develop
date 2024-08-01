#include "module_hsolver/diago_scalapack.hpp"

#include <gtest/gtest.h>
#include <unistd.h>
#include <cstdio>

#include "module_base/lapack_connector.h"


/**
 * Unit test for canon_diag (generalized eigenvalue problem solved with
 * canonical orthogonalization + ScaLAPACK pdsyevx/pzheevx).
 *
 * This test starts with a random r-by-r Hermitian matrix H0 and its
 * eigenpairs (E0, V0):
 *
 *                  H0 = V0 * diag(E0) * V0^H
 *
 * Next, a random r-by-n transformation matrix Y is generated to form
 *
 *                  H = Y^H * H0 * Y
 *                  S = Y^H * Y
 *
 * Now, the generalized eigenvalue problem
 *
 *                  H * V = S * V * diag(E)
 *
 * would be singular if n > r. This situation is supposed to be taken
 * care of by canon_diag, i.e., we will verify that the returned eigen-
 * pari (E, V) should satisfy E = E0 and Y * V ~ V0 (up to some phase)
 *
 */

template <typename T>
class DiagoScalapackCanonTest : public ::testing::Test {

protected:
    void SetUp();

    // MPI rank
    int rank;

    // problem size
    int r = 29;
    int n = 31;

    // number of eigenpairs to compute
    int neig = 17;

    // block size for block-cyclic distribution
    int nb = 4;

    // block-cyclic distribution info
    Parallel_2D p2d_V0;
    Parallel_2D p2d_Y;
    Parallel_2D p2d_HS;

    // original eigenpairs
    std::vector<double> E0;
    std::vector<T> V0;

    // transformation matrix
    std::vector<T> Y;

    // rank-deficient generalized eigenvalue problem
    std::vector<T> H;
    std::vector<T> S;

    // numerical tolerance
    double tol_ = 1.0e-12;
};

// fill a double or complex double value with a random number
void randfill(double& val) {
    val = 1.0 * rand() / RAND_MAX - 0.5;
}

void randfill(std::complex<double>& val) {
    double real = 1.0 * rand() / RAND_MAX - 0.5;
    double imag = 1.0 * rand() / RAND_MAX - 0.5;
    val = std::complex<double>(real, imag);
}

double myconj(double val) {
    return val;
}

std::complex<double> myconj(std::complex<double> val) {
    return std::conj(val);
}

template <typename T>
void DiagoScalapackCanonTest<T>::SetUp() {

    T one_f = 1.0;
    T zero_f = 0.0;
    bool is_cplx = std::is_same<T, std::complex<double>>::value;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    p2d_V0.init(r, r, nb, MPI_COMM_WORLD);
    int ctxt = p2d_V0.blacs_ctxt;

    p2d_Y.set(r, n, nb, ctxt);
    p2d_HS.set(n, n, nb, ctxt);

    E0.resize(r);
    V0.resize(p2d_V0.nloc);
    Y.resize(p2d_Y.nloc);
    H.resize(p2d_HS.nloc);
    S.resize(p2d_HS.nloc);

    // random Hermitian matrix H0
    srand(rank);
    Parallel_2D p2d_H0_glb;
    p2d_H0_glb.set(r, r, r, ctxt);
    std::vector<T> H0(p2d_V0.nloc); // block-cyclic distributed
    std::vector<T> H0_glb;          // rank-0 only
    if (rank == 0) {
        H0_glb.resize(r * r);
        for (int i = 0; i < r; ++i) {
            for (int j = i; j < r; ++j) {
                randfill(H0_glb[i + j*r]);
                if (i == j) {
                    H0_glb[i + j*r]= std::real(H0_glb[i + j*r]);
                }
                H0_glb[j + i*r] = myconj(H0_glb[i + j*r]);
            }
        }
    }

    // distribute H0_glb to H0
    Cpxgemr2d(r, r, H0_glb.data(), 1, 1, p2d_H0_glb.desc, H0.data(), 1, 1, p2d_V0.desc, ctxt);

    // solve H0 * V0 = V0 * diag(E0)
    pdiagx(neig, H0.data(), p2d_V0.desc, E0.data(), V0.data(), p2d_V0.desc);

    // generate random transformation matrix Y
    for (size_t i = 0; i < p2d_Y.nloc; ++i) {
        randfill(Y[i]);
    }

    // form H = Y^H * H0 * Y
    Parallel_2D p2d_YhH0;
    p2d_YhH0.set(n, r, nb, ctxt);
    std::vector<T> YhH0(p2d_YhH0.nloc); // intermediate variable

    ScalapackConnector::gemm(
        is_cplx ? 'C':'T', 'N',
        n, r, r,
        one_f,
        Y.data(), 1, 1, p2d_Y.desc,
        H0.data(), 1, 1, p2d_V0.desc,
        zero_f,
        YhH0.data(), 1, 1, p2d_YhH0.desc
    );

    ScalapackConnector::gemm(
        'N', 'N',
        n, n, r,
        one_f,
        YhH0.data(), 1, 1, p2d_YhH0.desc,
        Y.data(), 1, 1, p2d_Y.desc,
        zero_f,
        H.data(), 1, 1, p2d_HS.desc
    );

    // form S = Y^H * Y
    ScalapackConnector::gemm(
        is_cplx ? 'C':'T', 'N',
        n, n, r,
        one_f,
        Y.data(), 1, 1, p2d_Y.desc,
        Y.data(), 1, 1, p2d_Y.desc,
        zero_f,
        S.data(), 1, 1, p2d_HS.desc
    );
}

using TestedTypes = ::testing::Types<double, std::complex<double>>;
TYPED_TEST_SUITE(DiagoScalapackCanonTest, TestedTypes);

TYPED_TEST(DiagoScalapackCanonTest, RankDeficient) {
    TypeParam one_f = 1.0;
    TypeParam zero_f = 0.0;

    std::vector<double> E(this->neig);
    std::vector<TypeParam> V;
    V.resize(this->p2d_HS.nloc);
    canon_diag(this->neig, this->H.data(), this->S.data(), this->p2d_HS.desc, E.data(), V.data(), this->p2d_HS.desc);

    // check eigenvalues
    for (int i = 0; i < this->neig; ++i) {
        EXPECT_NEAR(E[i], this->E0[i], this->tol_);
    }

    // check Y * V = V0
    std::vector<TypeParam> YV(this->p2d_V0.nloc);
    ScalapackConnector::gemm(
        'N', 'N',
        this->r, this->neig, this->n,
        one_f,
        this->Y.data(), 1, 1, this->p2d_Y.desc,
        V.data(), 1, 1, this->p2d_HS.desc,
        zero_f,
        YV.data(), 1, 1, this->p2d_V0.desc
    );

    for (int i = 0; i < this->p2d_V0.nloc; ++i) {
        EXPECT_NEAR(std::abs(YV[i]), std::abs(this->V0[i]), this->tol_);
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    testing::InitGoogleTest(&argc, argv);
    int result = RUN_ALL_TESTS();
    MPI_Finalize();
    return result;
}
