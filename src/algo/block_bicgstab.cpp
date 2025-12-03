#include "algo/block_bicgstab.hpp"

using namespace std;
#include <vector>

MatrixXd block_bicgstab(const SparseMatrix<double>& A, const MatrixXd& B,
                        MatrixXd X, int max_iter, double tol,
                        vector<double>& res_norms) {
    MatrixXd R = B - A * X;
    MatrixXd P = R;
    MatrixXd R_hat_zero = R;
    MatrixXd R_zero = R;

    res_norms.clear();

    double b_norm = B.norm();
    if (b_norm == 0.0) b_norm = 1.0;

    res_norms.push_back(R.norm() / b_norm);

    for (int i = 0; i < max_iter; i++) {
        // --- 計算ループの中身は、密行列版とほぼ同じ ---

        // 1. A*P (疎行列 * 密行列 -> 密行列)
        MatrixXd AP = A * P;

        // 2. alphaの計算
        MatrixXd R_hat_T_AP = R_hat_zero.transpose() * AP;
        MatrixXd R_hat_T_R = R_hat_zero.transpose() * R;
        MatrixXd alpha = R_hat_T_AP.lu().solve(R_hat_T_R);

        // 3. T = R - AP * alpha
        MatrixXd T = R - AP * alpha;

        // 4. A*T (疎行列 * 密行列 -> 密行列)
        MatrixXd AT = A * T;

        // 5. zetaの計算
        double tr_AT_T = (AT.array() * T.array()).sum();
        double tr_AT_AT = (AT.array() * AT.array()).sum();
        double zeta = (tr_AT_AT != 0.0) ? (tr_AT_T / tr_AT_AT) : 0.0;

        // 6. X_{k+1} = X_k + P*alpha + zeta*T
        X += P * alpha + zeta * T;

        // 7. R_{k+1} = T - zeta*AT
        R = T - zeta * AT;

        // 8. 収束判定
        double r_norm = R.norm();
        res_norms.push_back(r_norm / b_norm);

        if (r_norm / b_norm < tol) {
            break;
        }

        // 9. betaの計算
        MatrixXd R_hat_T_R_new = R_hat_zero.transpose() * R;
        MatrixXd beta = (R_hat_T_AP * zeta).lu().solve(R_hat_T_R_new);

        // 10. P_{k+1} = R_{k+1} + (P_k - zeta*AP_k)*beta
        P = R + (P - zeta * AP) * beta;
    }
    return X;
}