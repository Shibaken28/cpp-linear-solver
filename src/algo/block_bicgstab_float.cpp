#include "algo/block_bicgstab.hpp"
#include "common/cholesky.hpp"

using namespace std;
#include <iostream>
#include <vector>

MatrixXf block_bicgstab_float(const SparseMatrix<float>& A, const MatrixXf& B,
                              MatrixXf X, int max_iter, float tol,
                              vector<float>& res_norms) {
    MatrixXf R = B - A * X;
    MatrixXf P = R;
    MatrixXf R_hat_zero = R;
    MatrixXf R_zero = R;

    res_norms.clear();

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

MatrixXf block_bicgstab_float_rq(const SparseMatrix<float>& A,
                                 const MatrixXf& B, MatrixXf X, int max_iter,
                                 float tol, vector<float>& res_norms) {
    MatrixXf R0 = B - A * X;
    MatrixXf R, eta;
    if (!CholeskyQR(R0, R, eta)) {
        return X;
    }
    MatrixXf R0_tilde = R0;
    MatrixXf P = R;

    res_norms.clear();

    float b_norm = B.norm();
    if (b_norm == 0.0) b_norm = 1.0;

    res_norms.push_back(eta.norm() / b_norm);

    for (int i = 0; i < max_iter; i++) {
        // --- 計算ループの中身は、密行列版とほぼ同じ ---

        // APを計算
        MatrixXf AP = A * P;

        // alphaの計算
        MatrixXf R0_tilde_T_A_P = R0_tilde.transpose() * AP;
        MatrixXf R0_tilde_R = R0_tilde.transpose() * R;
        MatrixXf alpha = R0_tilde_T_A_P.lu().solve(R0_tilde_R);

        // Tの計算
        MatrixXf T = R - AP * alpha;

        // ATの計算
        MatrixXf AT = A * T;

        // 5. zetaの計算
        MatrixXf Teta = T * eta;
        MatrixXf ATeta = AT * eta;
        float tr_AT_T = (ATeta.array() * Teta.array()).sum();
        float tr_AT_AT = (ATeta.array() * ATeta.array()).sum();
        float zeta = (tr_AT_AT != 0.0) ? (tr_AT_T / tr_AT_AT) : 0.0;

        // Xの更新
        X += (P * alpha + zeta * T) * eta;

        // R tauの計算
        MatrixXf R1;  // 次のR
        MatrixXf tau;
        MatrixXf t = T - zeta * A * T;
        if (not CholeskyQR(t, R1, tau)) {
            break;
        }

        // betaの計算
        MatrixXf R0_tilde_T_AP = R0_tilde.transpose() * AP;
        MatrixXf R0_tilde_T_R1 = R0_tilde.transpose() * R1;
        MatrixXf beta = (R0_tilde_T_AP * zeta).lu().solve(R0_tilde_T_R1);

        P = R1 + (P - zeta * AP) * beta;
        R = R1;
        eta = tau * eta;

        float eta_norm = eta.norm();
        res_norms.push_back(eta_norm / b_norm);
        if (eta_norm / b_norm < tol) {
            break;
        }
        cerr << "iter " << i + 1 << ": " << res_norms.back() << endl;
    }
    cerr << res_norms.back() << endl;
    return X;
}
