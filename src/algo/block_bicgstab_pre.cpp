
#include "algo/block_bicgstab.hpp"
#include "common/cholesky.hpp"

using namespace std;
#include <iomanip>
#include <iostream>
#include <vector>

// 前処理付きBlock BiCGSTAB法

MatrixXd block_bicgstab_preprocessing(const SparseMatrix<double>& A,
                                      const MatrixXd& B, MatrixXd X,
                                      int max_iter = 1000, double tol = 1e-15) {
    MatrixXd R = B - A * X;
    MatrixXd P = R;
    MatrixXd R_hat_zero = R;
    MatrixXd R_zero = R;

    // Aの逆行列に近い行列Kを事前計算
    // ここでは単精度(float)でILU分解を行い、メモリと計算時間を節約
    Eigen::IncompleteLUT<float> preconditioner;

    // パラメータ設定 (fillfactor: 非ゼロ要素の充填率, droptol: 切り捨て閾値)
    preconditioner.setFillfactor(7);
    preconditioner.setDroptol(1e-4);

    Eigen::SparseMatrix<float> A_float = A.cast<float>();

    // その変数を渡す
    preconditioner.compute(A_float);

    if (preconditioner.info() != Eigen::Success) {
        cerr << "ILU分解に失敗しました" << endl;
        return X;
    }
    vector<double> res_norms;

    double b_norm = B.norm();
    if (b_norm == 0.0) b_norm = 1.0;

    res_norms.push_back(R.norm() / b_norm);

    for (int i = 0; i < max_iter; i++) {
        // 前処理のために，K^{-1}Pを計算
        // K^-1の代わりにA^{-1}だと思うと，
        // Az = P と解くと z = A^{-1}P になる
        MatrixXd K_inv_P = preconditioner.solve(P.cast<float>()).cast<double>();

        // 1. A*P
        MatrixXd AKP = A * K_inv_P;

        // 2. alphaの計算
        MatrixXd R_hat_T_AP = R_hat_zero.transpose() * AKP;
        MatrixXd R_hat_T_R = R_hat_zero.transpose() * R;
        MatrixXd alpha = R_hat_T_AP.lu().solve(R_hat_T_R);

        // 3. T = R - AP * alpha
        MatrixXd T = R - AKP * alpha;

        MatrixXd K_inv_T = preconditioner.solve(T.cast<float>()).cast<double>();

        // 4. A*T (疎行列 * 密行列 -> 密行列)
        MatrixXd AKT = A * K_inv_T;

        // 5. zetaの計算
        double tr_AT_T = (AKT.array() * T.array()).sum();
        double tr_AT_AT = (AKT.array() * AKT.array()).sum();
        double zeta = (tr_AT_AT != 0.0) ? (tr_AT_T / tr_AT_AT) : 0.0;

        // 6. X_{k+1} = X_k + P*alpha + zeta*T
        X += K_inv_P * alpha + zeta * K_inv_T;

        // 7. R_{k+1} = T - zeta*AKT
        R = T - zeta * AKT;

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
        P = R + (P - zeta * AKP) * beta;
    }

    // 残差を出す
    auto R1 = B - A * X;
    double rel_res_norm = R1.norm() / B.norm();
    cout << "反復回数: " << res_norms.size() - 1
         << ", ||B-AX||/||B|| = " << scientific << setprecision(6)
         << rel_res_norm << endl;

    return X;
}

// 前処理付きBlock BiCGSTAB法 + QR分解による安定化
MatrixXd block_bicgstab_preprocessing_rq(const SparseMatrix<double>& A,
                                         const MatrixXd& B, MatrixXd X,
                                         int max_iter = 1000,
                                         double tol = 1e-15) {
    MatrixXd R0 = B - A * X;
    MatrixXd R, eta;
    CholeskyQR(R0, R, eta);
    MatrixXd R0_tilde = R0;
    MatrixXd P = R;

    // Aの逆行列に近い行列Kを事前計算
    // ここでは単精度(float)でILU分解を行い、メモリと計算時間を節約
    Eigen::IncompleteLUT<float> preconditioner;

    // パラメータ設定 (fillfactor: 非ゼロ要素の充填率, droptol: 切り捨て閾値)
    preconditioner.setFillfactor(7);
    preconditioner.setDroptol(1e-4);
    Eigen::SparseMatrix<float> A_float = A.cast<float>();
    preconditioner.compute(A_float);

    vector<double> res_norms;

    double b_norm = B.norm();
    if (b_norm == 0.0) b_norm = 1.0;

    res_norms.push_back(eta.norm() / b_norm);

    for (int i = 0; i < max_iter; i++) {
        // APを計算
        MatrixXd K_inv_P = preconditioner.solve(P.cast<float>()).cast<double>();
        MatrixXd AKP = A * K_inv_P;

        // alphaの計算
        MatrixXd R0_tilde_T_A_P = R0_tilde.transpose() * AKP;
        MatrixXd R0_tilde_R = R0_tilde.transpose() * R;
        MatrixXd alpha = R0_tilde_T_A_P.lu().solve(R0_tilde_R);
        // MatrixXd alpha =
        // MatrixXd(R0_tilde_T_A_P).fullPivLu().solve(R0_tilde_R);

        // Tの計算
        MatrixXd T = R - AKP * alpha;

        // ATの計算
        MatrixXd K_inv_T = preconditioner.solve(T.cast<float>()).cast<double>();
        MatrixXd AKT = A * K_inv_T;

        // 5. zetaの計算
        MatrixXd Teta = T * eta;
        MatrixXd ATeta = AKT * eta;
        double tr_AT_T = (ATeta.array() * Teta.array()).sum();
        double tr_AT_AT = (ATeta.array() * ATeta.array()).sum();
        double zeta = (tr_AT_AT != 0.0) ? (tr_AT_T / tr_AT_AT) : 0.0;

        // Xの更新
        X += (K_inv_P * alpha + zeta * K_inv_T) * eta;

        // R tauの計算
        MatrixXd R1;  // 次のR
        MatrixXd tau;
        MatrixXd t = T - zeta * A * K_inv_T;
        CholeskyQR(t, R1, tau);

        // betaの計算
        MatrixXd R0_tilde_T_AP = R0_tilde.transpose() * AKP;
        MatrixXd R0_tilde_T_R1 = R0_tilde.transpose() * R1;
        MatrixXd beta = (R0_tilde_T_AP * zeta).lu().solve(R0_tilde_T_R1);
        // MatrixXd beta = MatrixXd(R0_tilde_T_AP *
        // zeta).fullPivLu().solve(R0_tilde_T_R1);

        P = R1 + (P - zeta * AKP) * beta;
        R = R1;
        eta = tau * eta;

        double eta_norm = eta.norm();
        res_norms.push_back(eta_norm / b_norm);
        if (eta_norm / b_norm < tol) {
            break;
        }
    }

    // --- 検算 ---
    double final_res_norm = (B - A * X).norm();
    cerr << "反復回数: " << res_norms.size()
         << " 真の残差 ||B - AX||: " << scientific << setprecision(6)
         << final_res_norm / b_norm << endl;
    return X;
}
