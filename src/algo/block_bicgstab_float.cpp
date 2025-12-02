#include "algo/block_bicgstab.hpp"

using namespace std;
#include <iomanip>
#include <iostream>
#include <vector>

MatrixXf block_bicgstab_float(const SparseMatrix<float>& A, const MatrixXf& B,
                              MatrixXf X, int max_iter = 1000,
                              double tol = 1e-15) {
    MatrixXf R = B - A * X;
    MatrixXf P = R;
    MatrixXf R_hat_zero = R;
    MatrixXf R_zero = R;

    vector<float> res_norms;

    float b_norm = B.norm();
    if (b_norm == 0.0) b_norm = 1.0;

    res_norms.push_back(R.norm() / b_norm);

    for (int i = 0; i < max_iter; i++) {
        // --- 計算ループの中身は、密行列版とほぼ同じ ---

        // 1. A*P (疎行列 * 密行列 -> 密行列)
        MatrixXf AP = A * P;

        // 2. alphaの計算
        MatrixXf R_hat_T_AP = R_hat_zero.transpose() * AP;
        MatrixXf R_hat_T_R = R_hat_zero.transpose() * R;
        MatrixXf alpha = R_hat_T_AP.lu().solve(R_hat_T_R);

        // 3. T = R - AP * alpha
        MatrixXf T = R - AP * alpha;

        // 4. A*T (疎行列 * 密行列 -> 密行列)
        MatrixXf AT = A * T;

        // 5. zetaの計算
        float tr_AT_T = (AT.array() * T.array()).sum();
        float tr_AT_AT = (AT.array() * AT.array()).sum();
        float zeta = (tr_AT_AT != 0.0) ? (tr_AT_T / tr_AT_AT) : 0.0;

        // 6. X_{k+1} = X_k + P*alpha + zeta*T
        X += P * alpha + zeta * T;

        // 7. R_{k+1} = T - zeta*AT
        R = T - zeta * AT;

        // 8. 収束判定
        float r_norm = R.norm();
        res_norms.push_back(r_norm / b_norm);

        if (r_norm / b_norm < tol) {
            break;
        }

        // 9. betaの計算
        MatrixXf R_hat_T_R_new = R_hat_zero.transpose() * R;
        MatrixXf beta = (R_hat_T_AP * zeta).lu().solve(R_hat_T_R_new);

        // 10. P_{k+1} = R_{k+1} + (P_k - zeta*AP_k)*beta
        P = R + (P - zeta * AP) * beta;
    }

    return X;
}