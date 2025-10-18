// ファイル名: main.cpp

#include <iomanip>
#include <iostream>
#include <vector>

#include "io_utils.hpp"

// 密行列(MatrixXd)と疎行列(SparseMatrix)の両方をインクルード
#include <Eigen/Dense>
#include <Eigen/Sparse>

using namespace std;
using namespace Eigen;

// --- プロトタイプ宣言 (Aの型をSparseMatrixに変更) ---
void block_bicgstab_eigen(const SparseMatrix<double>& A, const MatrixXd& B,
                          MatrixXd& X);

int main() {
    auto A = read_sparse_matrix("matrix/494_bus.mtx");
    int n = A.rows();
    int col = 3;  // 複数の右辺を同時に解くための列数

    // BとXは密な行列なので、MatrixXdのまま
    MatrixXd B = MatrixXd::Zero(n, col);
    for (int i = 0; i < n * col; ++i) {
        B(i) = (double)(i * i);
    }
    MatrixXd X = MatrixXd::Zero(n, col);

    // 疎行列を引数として関数を呼び出す
    block_bicgstab_eigen(A, B, X);

    cout << "\n--- Solution ---" << endl;
    // cout << "Matrix X (Solution):\n" << X << endl; //
    // 大きいのでコメントアウト

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

    double tol = 1e-10;
    int max_iter = 10000;
    double b_norm = B.norm();
    if (b_norm == 0.0) b_norm = 1.0;

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
        cout << "反復回数 " << i + 1 << ": 残差ノルム |R|/|B| = " << scientific
             << setprecision(6) << r_norm / b_norm << endl;

        if (r_norm / b_norm < tol) {
            cout << i + 1 << "回の反復収束" << endl;
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
    cout << "\n真の残差 ||B - AX||: " << scientific << setprecision(6)
         << final_res_norm / b_norm << endl;
}