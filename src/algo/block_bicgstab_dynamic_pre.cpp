#include "algo/bicgstab.hpp"
#include "algo/block_bicgstab.hpp"
#include "common/cholesky.hpp"

using namespace std;
#include <iostream>
#include <vector>

MatrixXd block_bicgstab_dynamic_preprocessing(const SparseMatrix<double>& A,
                                              const MatrixXd& B, MatrixXd X,
                                              int max_iter, double tol,
                                              vector<double>& res_norms,
                                              int inner_max_iter,
                                              float inner_tol) {
    MatrixXd R = B - A * X;
    MatrixXd P = R;
    MatrixXd R_hat_zero = R;
    MatrixXd R_zero = R;

    res_norms.clear();

    double b_norm = B.norm();
    if (b_norm == 0.0) b_norm = 1.0;

    res_norms.push_back(R.norm() / b_norm);

    auto A_float = A.cast<float>();
    MatrixXf Zero_float =
        MatrixXf::Zero(A.cols(), P.cols());  // float型のゼロ行列

    vector<float> dummy_res_norms;
    for (int i = 0; i < max_iter; i++) {
        // --- 計算ループの中身は、密行列版とほぼ同じ ---

        // 前処理のために，K^{-1}Pを計算
        // K^-1の代わりにA^{-1}だと思うと，
        // Az = P と解くと z = A^{-1}P になる
        MatrixXd K_inv_P =
            block_bicgstab_float(A_float, P.cast<float>(), Zero_float,
                                 inner_max_iter, inner_tol, dummy_res_norms)
                .cast<double>();

        // 1. A*P (疎行列 * 密行列 -> 密行列)
        MatrixXd AKP = A * K_inv_P;

        // 2. alphaの計算
        MatrixXd R_hat_T_AP = R_hat_zero.transpose() * AKP;
        MatrixXd R_hat_T_R = R_hat_zero.transpose() * R;
        MatrixXd alpha = R_hat_T_AP.lu().solve(R_hat_T_R);

        // 3. T = R - AP * alpha
        MatrixXd T = R - AKP * alpha;

        MatrixXd K_inv_T =
            block_bicgstab_float(A_float, T.cast<float>(), Zero_float,
                                 inner_max_iter, inner_tol, dummy_res_norms)
                .cast<double>();

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
    return X;
}

MatrixXd block_bicgstab_dynamic_preprocessing_rq(const SparseMatrix<double>& A,
                                                 const MatrixXd& B, MatrixXd X,
                                                 int max_iter, double tol,
                                                 vector<double>& res_norms,
                                                 int inner_max_iter,
                                                 float inner_tol) {
    MatrixXd R0 = B - A * X;
    MatrixXd R, eta;
    if (!CholeskyQR(R0, R, eta)) {
        return X;
    }
    MatrixXd R0_tilde = R0;
    MatrixXd P = R;

    Eigen::SparseMatrix<float> A_float = A.cast<float>();

    res_norms.clear();

    double b_norm = B.norm();
    if (b_norm == 0.0) b_norm = 1.0;

    res_norms.push_back(eta.norm() / b_norm);

    vector<double> dummy_res_norms;
    for (int i = 0; i < max_iter; i++) {
        // APを計算
        /*
        MatrixXd K_inv_P =
            block_bicgstab_float_rq(A_float, P.cast<float>(),
                                    MatrixXf::Zero(A.cols(), P.cols()),
                                    inner_max_iter, inner_tol, dummy_res_norms)
                .cast<double>();
        */
        MatrixXd K_inv_P =
            block_bicgstab_rq(A, P, MatrixXd::Zero(A.cols(), P.cols()),
                              inner_max_iter, inner_tol, dummy_res_norms);
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
        /*
        MatrixXd K_inv_T =
            block_bicgstab_float_rq(A_float, T.cast<float>(),
                                    MatrixXf::Zero(A.cols(), T.cols()),
                                    inner_max_iter, inner_tol, dummy_res_norms)
                .cast<double>();
        */
        MatrixXd K_inv_T =
            block_bicgstab_rq(A, T, MatrixXd::Zero(A.cols(), T.cols()),
                              inner_max_iter, inner_tol, dummy_res_norms);
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
        if (!CholeskyQR(t, R1, tau)) {
            break;
        }

        // betaの計算
        MatrixXd R0_tilde_T_AP = R0_tilde.transpose() * AKP;
        MatrixXd R0_tilde_T_R1 = R0_tilde.transpose() * R1;
        MatrixXd beta = (R0_tilde_T_AP * zeta).lu().solve(R0_tilde_T_R1);

        P = R1 + (P - zeta * AKP) * beta;
        R = R1;
        eta = tau * eta;

        double eta_norm = eta.norm();
        res_norms.push_back(eta_norm / b_norm);
        if (eta_norm / b_norm < tol) {
            break;
        }
        // あまりにも発散している場合(最初のres_normの1e4倍以上)は終了
        if (res_norms.back() > res_norms[0] * 1e4) {
            cerr << "発散を検出したため終了します。" << endl;
            break;
        }
    }

    return X;
}

// 内部で1本ずつ解く版
MatrixXd block_bicgstab_dynamic_preprocessing_rq_single(
    const SparseMatrix<double>& A, const MatrixXd& B, MatrixXd X, int max_iter,
    double tol, vector<double>& res_norms, int inner_max_iter,
    float inner_tol) {
    MatrixXd R0 = B - A * X;
    MatrixXd R, eta;
    if (!CholeskyQR(R0, R, eta)) {
        return X;
    }
    MatrixXd R0_tilde = R0;
    MatrixXd P = R;

    Eigen::SparseMatrix<float> A_float = A.cast<float>();

    res_norms.clear();

    double b_norm = B.norm();
    if (b_norm == 0.0) b_norm = 1.0;

    res_norms.push_back(eta.norm() / b_norm);

    vector<double> dummy_res_norms;
    for (int i = 0; i < max_iter; i++) {
        // APを計算
        // 前処理のために，K^{-1}Pを計算
        // K^-1の代わりにA^{-1}だと思うと，
        // Az = P と解くと z = A^{-1}P になる
        // ここでPを1列ずつ解くことにする
        MatrixXd K_inv_P(A.rows(), P.cols());
        for (int col = 0; col < P.cols(); col++) {
            VectorXd p_col = P.col(col);
            VectorXd k_inv_p_col =
                bicgstab(A, p_col, VectorXd::Zero(A.cols()), inner_max_iter,
                         inner_tol, dummy_res_norms);
            K_inv_P.col(col) = k_inv_p_col;
        }
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
        // ここでTを1列ずつ解くことにする
        MatrixXd K_inv_T(A.rows(), T.cols());
        for (int col = 0; col < T.cols(); col++) {
            VectorXd t_col = T.col(col);
            VectorXd k_inv_t_col =
                bicgstab(A, t_col, VectorXd::Zero(A.cols()), inner_max_iter,
                         inner_tol, dummy_res_norms);
            K_inv_T.col(col) = k_inv_t_col;
        }
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
        if (!CholeskyQR(t, R1, tau)) {
            break;
        }

        // betaの計算
        MatrixXd R0_tilde_T_AP = R0_tilde.transpose() * AKP;
        MatrixXd R0_tilde_T_R1 = R0_tilde.transpose() * R1;
        MatrixXd beta = (R0_tilde_T_AP * zeta).lu().solve(R0_tilde_T_R1);

        P = R1 + (P - zeta * AKP) * beta;
        R = R1;
        eta = tau * eta;

        double eta_norm = eta.norm();
        res_norms.push_back(eta_norm / b_norm);
        if (eta_norm / b_norm < tol) {
            break;
        }
        // あまりにも発散している場合(最初のres_normの1e4倍以上)は終了
        if (res_norms.back() > res_norms[0] * 1e4) {
            cerr << "発散を検出したため終了します。" << endl;
            break;
        }
    }

    return X;
}