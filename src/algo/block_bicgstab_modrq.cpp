#include "algo/block_bicgstab.hpp"
#include "common/cholesky.hpp"

using namespace std;
#include <vector>

MatrixXd block_bicgstab_modrq(const SparseMatrix<double>& A, const MatrixXd& B,
                              MatrixXd X, int max_iter, double tol,
                              vector<double>& res_norms) {
    // --- 初期化 ---
    MatrixXd R0 = B - A * X;
    MatrixXd R, eta;
    CholeskyQR(R0, R, eta);
    MatrixXd R0_tilde = R0;
    MatrixXd R0_hat = A.transpose() * R0_tilde;
    MatrixXd P = R;

    res_norms.clear();

    double b_norm = B.norm();
    if (b_norm == 0.0) b_norm = 1.0;

    res_norms.push_back(eta.norm() / b_norm);

    for (int i = 0; i < max_iter; i++) {
        // --- 計算ループの中身は、密行列版とほぼ同じ ---

        // alphaの計算
        MatrixXd R_hat_T_P = R0_hat.transpose() * P;
        MatrixXd R_tilde_T_R = R0_tilde.transpose() * R;
        MatrixXd alpha = R_hat_T_P.lu().solve(R_tilde_T_R);

        // Sの計算
        MatrixXd S = P * alpha;

        // ASの計算
        MatrixXd AS = A * S;

        // Tの計算
        MatrixXd T = R - AS;

        // ATの計算
        MatrixXd AT = A * T;

        // 5. zetaの計算
        MatrixXd Teta = T * eta;
        MatrixXd ATeta = AT * eta;
        double tr_AT_T = (ATeta.array() * Teta.array()).sum();
        double tr_AT_AT = (ATeta.array() * ATeta.array()).sum();
        double zeta = (tr_AT_AT != 0.0) ? (tr_AT_T / tr_AT_AT) : 0.0;

        // Xの更新
        X += (S + zeta * T) * eta;

        // R tauの計算
        MatrixXd R1;  // 次のR
        MatrixXd tau;
        MatrixXd t = T - zeta * A * T;
        CholeskyQR(t, R1, tau);

        // gammaの計算
        MatrixXd R0_tilde_T_R = R0_tilde.transpose() * R;
        MatrixXd R0_tilde_T_R1 = R0_tilde.transpose() * R1;

        MatrixXd gamma = (R_tilde_T_R * zeta).lu().solve(R0_tilde_T_R1);

        P = R1 + (S - zeta * AS) * gamma;
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