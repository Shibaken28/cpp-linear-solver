#include "algo/block_bicgstab.hpp"
#include "common/cholesky.hpp"

using namespace std;
#include <iomanip>
#include <iostream>
#include <vector>

MatrixXd block_bicgstab_dynamic_preprocessing(const SparseMatrix<double>& A,
                                              const MatrixXd& B, MatrixXd X,
                                              int max_iter = 1000,
                                              double tol = 1e-15) {
    MatrixXd R = B - A * X;
    MatrixXd P = R;
    MatrixXd R_hat_zero = R;
    MatrixXd R_zero = R;

    // Aの逆行列に近い行列Kを事前計算

    vector<double> res_norms;

    double b_norm = B.norm();
    if (b_norm == 0.0) b_norm = 1.0;

    res_norms.push_back(R.norm() / b_norm);

    auto A_float = A.cast<float>();
    MatrixXf Zero_float =
        MatrixXf::Zero(A.cols(), P.cols());  // float型のゼロ行列

    for (int i = 0; i < max_iter; i++) {
        // --- 計算ループの中身は、密行列版とほぼ同じ ---

        // 前処理のために，K^{-1}Pを計算
        // K^-1の代わりにA^{-1}だと思うと，
        // Az = P と解くと z = A^{-1}P になる
        MatrixXd K_inv_P =
            block_bicgstab_float(A_float, P.cast<float>(), Zero_float, 20, 1e-5)
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
            block_bicgstab_float(A_float, T.cast<float>(), Zero_float, 20, 1e-5)
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

    // 残差を出す
    auto R1 = B - A * X;
    double rel_res_norm = R1.norm() / B.norm();
    cerr << "反復回数: " << res_norms.size() - 1
         << ", ||B-AX||/||B|| = " << scientific << setprecision(6)
         << rel_res_norm << endl;

    return X;
}
