#include "algo/block_bicgstab.hpp"
#include "common/cholesky.hpp"

using namespace std;
#include <iomanip>
#include <iostream>
#include <vector>

MatrixXd block_bicgstab_rq(const SparseMatrix<double>& A, const MatrixXd& B,
                           MatrixXd X, int max_iter, double tol,
                           vector<double>& res_norms) {
    MatrixXd R0 = B - A * X;
    MatrixXd R, eta;
    CholeskyQR(R0, R, eta);
    MatrixXd R0_tilde = R0;
    MatrixXd P = R;

    res_norms.clear();

    double b_norm = B.norm();
    if (b_norm == 0.0) b_norm = 1.0;

    res_norms.push_back(eta.norm() / b_norm);

    for (int i = 0; i < max_iter; i++) {
        // --- 計算ループの中身は、密行列版とほぼ同じ ---

        // APを計算
        MatrixXd AP = A * P;

        // alphaの計算
        MatrixXd R0_tilde_T_A_P = R0_tilde.transpose() * AP;
        MatrixXd R0_tilde_R = R0_tilde.transpose() * R;
        MatrixXd alpha = R0_tilde_T_A_P.lu().solve(R0_tilde_R);

        // Tの計算
        MatrixXd T = R - AP * alpha;

        // ATの計算
        MatrixXd AT = A * T;

        // 5. zetaの計算
        MatrixXd Teta = T * eta;
        MatrixXd ATeta = AT * eta;
        double tr_AT_T = (ATeta.array() * Teta.array()).sum();
        double tr_AT_AT = (ATeta.array() * ATeta.array()).sum();
        double zeta = (tr_AT_AT != 0.0) ? (tr_AT_T / tr_AT_AT) : 0.0;

        // Xの更新
        X += (P * alpha + zeta * T) * eta;

        // R tauの計算
        MatrixXd R1;  // 次のR
        MatrixXd tau;
        MatrixXd t = T - zeta * A * T;
        CholeskyQR(t, R1, tau);

        // betaの計算
        MatrixXd R0_tilde_T_AP = R0_tilde.transpose() * AP;
        MatrixXd R0_tilde_T_R1 = R0_tilde.transpose() * R1;
        MatrixXd beta = (R0_tilde_T_AP * zeta).lu().solve(R0_tilde_T_R1);

        P = R1 + (P - zeta * AP) * beta;
        R = R1;
        eta = tau * eta;

        double eta_norm = eta.norm();
        res_norms.push_back(eta_norm / b_norm);
        if (eta_norm / b_norm < tol) {
            break;
        }
    }
    return X;
}
