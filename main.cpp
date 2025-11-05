// ファイル名: main.cpp

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#include "io_utils.hpp"

using namespace std;
using namespace Eigen;

// --- プロトタイプ宣言 (Aの型をSparseMatrixに変更) ---
void block_bicgstab_eigen(const SparseMatrix<double>& A, const MatrixXd& B,
                          MatrixXd& X);

// pdeが解きやすい
int main() {
    auto A = read_sparse_matrix("matrix/1138_bus.mtx");
    int n = A.rows();
    vector<int> test_cols = {1, 3, 5, 7, 9};
    cout << test_cols.size() << endl;
    for (int col : test_cols) {
        // BとXは密な行列なので、MatrixXdのまま
        cout << col << endl;

        MatrixXd B = MatrixXd::Zero(n, col);
        unsigned int fixed_seed = 42;
        std::mt19937 gen(fixed_seed);  // メルセンヌ・ツイスターエンジン
        // 3. 浮動小数点数の分布を設定
        //    0.0から1.0までの一様分布
        std::uniform_real_distribution<double> real_dist(0.0, 1.0);
        for (int i = 0; i < n * col; ++i) {
            B(i) = real_dist(gen);
        }
        MatrixXd X = MatrixXd::Zero(n, col);

        // 疎行列を引数として関数を呼び出す
        block_bicgstab_eigen(A, B, X);
    }

    return 0;
}

void block_bicgstab_eigen(const SparseMatrix<double>& A, const MatrixXd& B,
                          MatrixXd& X) {
    // --- 初期化 ---
    // Aが疎行列、Xが密行列でも、積は正しく計算され、結果は密行列Rになります
    MatrixXd R = B - A * X;
    MatrixXd P = R;
    MatrixXd R_hat_zero = R;
    MatrixXd R_zero = R;

    vector<double> res_norms;

    double tol = 1e-10;
    int max_iter = 1000;
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

    // --- 検算 ---
    double final_res_norm = (B - A * X).norm();
    cerr << "反復回数: " << res_norms.size()
         << " 真の残差 ||B - AX||: " << scientific << setprecision(6)
         << final_res_norm / b_norm << endl;

    // --- 収束履歴の出力 ---
    cout << res_norms.size() << endl;
    for (size_t i = 0; i < res_norms.size(); i++) {
        // cout << scientific << setprecision(6) << res_norms[i] << endl;
        cout << fixed << setprecision(20) << res_norms[i] << endl;
    }
}