#include <iostream>
#include <Eigen/SparseCore>
#include <Eigen/Sparse>
#include<Eigen/IterativeLinearSolvers>
#include <igl/opengl/glfw/Viewer.h>
#include <igl/readOFF.h>
#include <chrono> 
#include "simulator.hpp"

using namespace Eigen;
using namespace std::chrono;
using namespace std;
int N = 62;

Simulator::Simulator() {
    MatrixXd A = MatrixXd::Zero(62 * 62, 62 * 62);
    MatrixXd Dsub = MatrixXd::Identity(62, 62);

    Dsub *= 1 + (2 * ((D[0] * D[0] + D[1] * D[1]) * visc * dt) / (D[0] * D[0] * D[1] * D[1]));
    float coeff = -(visc * dt) / (D[0] * D[0]);
    // Since D[0] and D[1] are equal, we can use coeff for both
    Dsub(0, 1) = coeff;
    Dsub(Dsub.rows() - 1, Dsub.cols() - 2) = coeff;
    for (int i = 1; i < Dsub.rows() - 1; i++) {
        Dsub(i, i - 1) = coeff;
        Dsub(i, i + 1) = coeff;
    }
    MatrixXd E = MatrixXd::Identity(62, 62);
    E *= coeff;
    // first
    A.block(0, 0, 62, 62) = Dsub;
    A.block(0, 62, 62, 62) = E;
    // middle
    for (int i = 1; i < 61; i++) {
        A.block(i * 62, i * 62, 62, 62) = Dsub;
        A.block(i * 62, (i - 1) * 62, 62, 62) = E;
        A.block(i * 62, (i + 1) * 62, 62, 62) = E;
    }
    // last
    A.block(61 * 62, 61 * 62, 62, 62) = Dsub;
    A.block(61 * 62, 60 * 62, 62, 62) = E;

    sparseA = A.sparseView();

    MatrixXd A_ks = MatrixXd::Zero(62 * 62, 62 * 62);
    MatrixXd Dsub_ks = MatrixXd::Identity(62, 62);

    Dsub_ks *= 1 + (2 * ((D[0] * D[0] + D[1] * D[1]) * kS * dt) / (D[0] * D[0] * D[1] * D[1]));
    float coeff_ks = -(kS * dt) / (D[0] * D[0]);
    Dsub_ks(0, 1) = coeff_ks;
    Dsub_ks(Dsub_ks.rows() - 1, Dsub_ks.cols() - 2) = coeff_ks;
    for (int i = 1; i < Dsub_ks.rows() - 1; i++) {
        Dsub_ks(i, i - 1) = coeff_ks;
        Dsub_ks(i, i + 1) = coeff_ks;
    }
    MatrixXd E_ks = MatrixXd::Identity(62, 62);
    E_ks *= coeff_ks;
    // first
    A_ks.block(0, 0, 62, 62) = Dsub_ks;
    A_ks.block(0, 62, 62, 62) = E_ks;
    // middle
    for (int i = 1; i < 61; i++) {
        A_ks.block(i * 62, i * 62, 62, 62) = Dsub_ks;
        A_ks.block(i * 62, (i - 1) * 62, 62, 62) = E_ks;
        A_ks.block(i * 62, (i + 1) * 62, 62, 62) = E_ks;
    }
    // last
    A_ks.block(61 * 62, 61 * 62, 62, 62) = Dsub_ks;
    A_ks.block(61 * 62, 60 * 62, 62, 62) = E_ks;

    sparseA_ks = A_ks.sparseView();

    // for project step
    MatrixXd A_pr = MatrixXd::Zero(62 * 62, 62 * 62);
    MatrixXd Dsub_pr = MatrixXd::Identity(62, 62);

    Dsub_pr *= ((D[0] * D[0] + D[1] * D[1]) * float(-2)) / (D[0] * D[0] * D[1] * D[1]);
    float coeff_pr = float(1) / (D[0] * D[0]);
    Dsub_pr(0, 1) = coeff_pr;
    Dsub_pr(Dsub_pr.rows() - 1, Dsub_pr.cols() - 2) = coeff_pr;
    for (int i = 1; i < Dsub_pr.rows() - 1; i++) {
        Dsub_pr(i, i - 1) = coeff_pr;
        Dsub_pr(i, i + 1) = coeff_pr;
    }
    MatrixXd E_pr = MatrixXd::Identity(62, 62);
    E_pr *= coeff_pr;
    A_pr.block(0, 0, 62, 62) = Dsub_pr;
    A_pr.block(0, 62, 62, 62) = E_pr;
    // middle
    for (int i = 1; i < 61; i++) {
        A_pr.block(i * 62, i * 62, 62, 62) = Dsub_pr;
        A_pr.block(i * 62, (i - 1) * 62, 62, 62) = E_pr;
        A_pr.block(i * 62, (i + 1) * 62, 62, 62) = E_pr;
    }
    // last
    A_pr.block(61 * 62, 61 * 62, 62, 62) = Dsub_pr;
    A_pr.block(61 * 62, 60 * 62, 62, 62) = E_pr;

    sparseA_pr = A_pr.sparseView();
}

Simulator::~Simulator() {}

void Simulator::addForce(float Grid0[64][64], float Force[64][64]) {

    for (int i = 0; i < N + 2; i++) {
        for (int j = 0; j < N + 2; j++)
            Grid0[i][j] += dt * Force[i][j];
    }
}

void Simulator::Diffuse(int B, SparseMatrix<double> sparseM, float Grid1[64][64], float Grid0[64][64]) {
    VectorXd b(62 * 62);
    for (int i = 1; i < 63; i++) {
        for (int j = 1; j < 63; j++)
            b[(i - 1) * 62 + j - 1] = Grid0[i][j];
    }
    ConjugateGradient<SparseMatrix<double>, Lower | Upper> cg;
    cg.compute(sparseM);

    VectorXd x(62 * 62);
    x = cg.solve(b);
    for (int i = 1; i < 63; i++) {
        for (int j = 1; j < 63; j++)
            Grid1[i][j] = x[(i - 1) * 62 + j - 1];
    }
    set_boundary(B, Grid1);
}

void Simulator::Dissipate(float S0[64][64], float S1[64][64]) {
    for (int i = 0; i < 64; i++) {
        for (int j = 0; j < 64; j++)
            S1[i][j] = S0[i][j] / (1 + dt * aS);
    }
}

void Simulator::Project(float U[2][64][64], float pdiv[2][64][64]) {
    float tmp[2][64][64];
    for (int i = 0; i < 64; i++) {
        for (int j = 0; j < 64; j++) {
            tmp[0][i][j] = pdiv[0][i][j];
        }
    }
    for (int i = 0; i < 64; i++) {
        for (int j = 0; j < 64; j++) {
            tmp[1][i][j] = pdiv[1][i][j];
        }
    }
    ConjugateGradient<SparseMatrix<double>, Lower | Upper> cg_pr;
    cg_pr.compute(sparseA_pr);

    for (int i = 1; i <= 62; i++) {
        for (int j = 1; j <= 62; j++) {
            pdiv[1][i][j] = 0.5 * (U[0][i + 1][j] - U[0][i - 1][j] + U[1][i][j + 1] - U[1][i][j - 1]);
            pdiv[0][i][j] = 0;
        }
    }
    set_boundary(0, pdiv[0]);
    set_boundary(0, pdiv[1]);
    VectorXd b(62 * 62);
    for (int i = 1; i < 63; i++) {
        for (int j = 1; j < 63; j++)
            b[(i - 1) * 62 + j - 1] = pdiv[1][i][j];
    }

    VectorXd x(62 * 62);
    x = cg_pr.solve(b);

    for (int i = 1; i < 63; i++) {
        for (int j = 1; j < 63; j++)
            pdiv[0][i][j] = x[(i - 1) * 62 + j - 1]; 
    }
    set_boundary(0, pdiv[0]);

    for (int i = 1; i <= 62; i++) { 
        for (int j = 1; j <= 62; j++) {
            U[0][i][j] = tmp[0][i][j] - 0.5 * (pdiv[0][i + 1][j] - pdiv[0][i - 1][j]); 
            U[1][i][j] = tmp[1][i][j] - 0.5 * (pdiv[0][i][j + 1] - pdiv[0][i][j - 1]);
        }
    }

    for (int i = 0; i < 64; i++) {
        for (int j = 0; j < 64; j++) {
            pdiv[0][i][j] = tmp[0][i][j];
        }
    }
    for (int i = 0; i < 64; i++) {
        for (int j = 0; j < 64; j++) {
            pdiv[1][i][j] = tmp[1][i][j];
        }
    }
    set_boundary(1, U[0]);
    set_boundary(2, U[1]);
}

void Simulator::TraceParticle(float X[2], float X0[2], float U[2][64][64]) {
    int i = int(X[0] - 0.5);
    int j = int(X[1] - 0.5);

    float K1i = dt * N * U[0][i][j];
    float iPrev = i - K1i;
    if (iPrev < 0.5) 
        iPrev = 0.5;
    if (iPrev > 62.5)
        iPrev = 62.5;

    float K1j = dt * N * U[1][i][j];
    float jPrev = j - K1j;
    if (jPrev < 0.5)
        jPrev = 0.5;
    if (jPrev > 62.5)
        jPrev = 62.5;

    float K2i = dt * N * U[0][int(iPrev)][int(jPrev)]; 
    float i0 = i - (K1i + K2i) / 2;
    if (i0 < 0)
        i0 = 0;
    if (i0 > 62)
        i0 = 62;
    X0[0] = i0;

    float K2j = dt * N * U[1][int(iPrev)][int(jPrev)]; 
    float j0 = j - (K1j + K2j) / 2;
    if (j0 < 0)
        j0 = 0;
    if (j0 > 62)
        j0 = 62;
    X0[1] = j0;
}

float Simulator::LinInterp(float X0[2], float Grid0[64][64]) {
    int int_i = int(X0[0]);
    int int_j = int(X0[1]);
    float s1 = X0[0] - int_i;
    float t1 = X0[1] - int_j;
    float res = (1 - s1) * ((1 - t1) * Grid0[int_i][int_j] + t1 * Grid0[int_i][int_j + 1])
        + s1 * ((1 - t1) * Grid0[int_i + 1][int_j] + t1 * Grid0[int_i + 1][int_j + 1]);
    return res;
}


void Simulator::Transport(int b, float Grid1[64][64], float Grid0[64][64], float U[2][64][64]) {
    float X[2];
    float X0[2];
    for (int i = 1; i < 63; i++) {
        for (int j = 1; j < 63; j++) {
            X[0] = Origin[0] + (i + 0.5) * D[0];
            X[1] = Origin[1] + (j + 0.5) * D[1];
            TraceParticle(X, X0, U);
            Grid1[i][j] = LinInterp(X0, Grid0);
        }
    }
    set_boundary(b, Grid1);
}

void Simulator::Sstep(float S[64][64], float Sprev[64][64], float U[2][64][64]) {
    addForce(Sprev, Ssource);
    Transport(0, S, Sprev, U);
    Diffuse(0, sparseA_ks, Sprev, S);
    Dissipate(Sprev, S);
}

void Simulator::Vstep(float U[2][64][64], float U0[2][64][64]) {
    for (int i = 0; i < 2; i++) {
        addForce(U0[i], F[i]);
    }
    Transport(1, U[0], U0[0], U0); 
    Transport(2, U[1], U0[1], U0);
    Diffuse(1, sparseA, U0[0], U[0]);
    Diffuse(2, sparseA, U0[1], U[1]);
    Project(U, U0);
}

void Simulator::SwapV(float U1[2][64][64], float U0[2][64][64]) {
    for (int dim = 0; dim < 2; dim++) {
        float V_new[64][64];
        for (int i = 0; i < 64; i++) {
            for (int j = 0; j < 64; j++) {
                V_new[i][j] = U1[dim][i][j];
            }
        }
        for (int i = 0; i < 64; i++) {
            for (int j = 0; j < 64; j++) {
                U1[dim][i][j] = U0[dim][i][j];
            }
        }
        for (int i = 0; i < 64; i++) {
            for (int j = 0; j < 64; j++) {
                U0[dim][i][j] = V_new[i][j];
            }
        }
    }
}

void Simulator::SwapS(float S_new[64][64], float S_prev[64][64]) {
    float S_new2[64][64];
    for (int i = 0; i < 64; i++) {
        for (int j = 0; j < 64; j++) {
            S_new2[i][j] = S_new[i][j];
        }
    }
    for (int i = 0; i < 64; i++) {
        for (int j = 0; j < 64; j++) {
            S_new[i][j] = S_prev[i][j];
        }
    }
    for (int i = 0; i < 64; i++) {
        for (int j = 0; j < 64; j++) {
            S_prev[i][j] = S_new2[i][j];
        }
    }
}

// b = 0 when setting boundaries for density, b = 1 for vertical velocity
// and b = 2 for horizontal velocity
void Simulator::set_boundary(int b, float x[64][64]) {
    for (int i = 1; i <= N; i++) {
        if (b == 1) {
            x[i][0] = x[i][1];
            x[i][N + 1] = x[i][N];
            if (x[1][i] < 0)
                x[0][i] = -1 * x[1][i];
            else
                x[0][i] = x[1][i];

            if (x[N][i] > 0)
                x[N + 1][i] = -1 * x[N][i];
            else
                x[N + 1][i] = x[N][i];
        }
        else if (b == 2) {
            x[0][i] = x[1][i];
            x[N + 1][i] = x[N][i];
            if (x[i][1] < 0)
                x[i][0] = -1 * x[i][1];
            else
                x[i][0] = x[i][1];
            if (x[i][N] > 0)
                x[i][N + 1] = -1 * x[i][N];
            else
                x[i][N + 1] = x[i][N];
        }
        else {
            x[0][i] = x[1][i];
            x[N + 1][i] = x[N][i];
            x[i][0] = x[i][1];
            x[i][N + 1] = x[i][N];
        }
    }

    // corners
    x[0][0] = 0.5 * (x[1][0] + x[0][1]);
    x[N + 1][N + 1] = 0.5 * (x[N][N + 1] + x[N + 1][N]);
    x[0][N + 1] = 0.5 * (x[1][N + 1] + x[0][N]);
    x[N + 1][0] = 0.5 * (x[N][0] + x[N + 1][1]);
}
