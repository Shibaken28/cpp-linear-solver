// ファイル名: main.cpp

#include <Eigen/Cholesky>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#include "io_utils.hpp"

using namespace std;
using namespace Eigen;

constexpr unsigned int fixed_seed = 42;
std::mt19937 gen(fixed_seed);

bool CholeskyQR(const MatrixXd& A, MatrixXd& Q, MatrixXd& R) {
    MatrixXd AtA = A.transpose() * A;
    LLT<MatrixXd> llt(AtA);

    if (llt.info() != Eigen::Success) {
        cerr << "コレスキー分解失敗" << endl;
        return false;
    }
    R = llt.matrixU();
    auto R_inv = R.inverse();
    Q = A * R_inv;
    return true;
}

// --- プロトタイプ宣言 (Aの型をSparseMatrixに変更) ---
MatrixXd block_bicgstab_eigen(const SparseMatrix<double>& A, const MatrixXd& B,
                              MatrixXd X);

MatrixXd generate_dense_matrix(int rows, int cols) {
    MatrixXd mat = MatrixXd::Zero(rows, cols);
    std::uniform_real_distribution<double> real_dist(0.0, 1.0);
    for (int i = 0; i < rows * cols; ++i) {
        mat(i) = real_dist(gen);
    }
    return mat;
}

void generate_problem(int m, SparseMatrix<double>& A, MatrixXd& X,
                      MatrixXd& B) {
    A = read_sparse_matrix("matrix/pde2961.mtx");
    // A = read_sparse_matrix("matrix/simple.mtx");
    int n = A.rows();

    X = generate_dense_matrix(n, m);
    B = A * X;
}

MatrixXf block_bicgstab_eigen_float(const SparseMatrix<float>& A,
                                    const MatrixXf& B, MatrixXf X,
                                    int max_iter = 1000, double tol = 1e-15) {
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

MatrixXd block_bicgstab_eigen_preprocessing2(const SparseMatrix<double>& A,
                                             const MatrixXd& B, MatrixXd X,
                                             int max_iter = 1000,
                                             double tol = 1e-15) {
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

    // 分解の実行 (double -> float にキャストして計算)
    // 一度 float型の疎行列として変数に受ける
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
        // --- 計算ループの中身は、密行列版とほぼ同じ ---

        // 前処理のために，K^{-1}Pを計算
        // K^-1の代わりにA^{-1}だと思うと，
        // Az = P と解くと z = A^{-1}P になる
        MatrixXd K_inv_P = preconditioner.solve(P.cast<float>()).cast<double>();

        // 1. A*P (疎行列 * 密行列 -> 密行列)
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

MatrixXd block_bicgstab_eigen_preprocessing(const SparseMatrix<double>& A,
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

    for (int i = 0; i < max_iter; i++) {
        // --- 計算ループの中身は、密行列版とほぼ同じ ---

        // 前処理のために，K^{-1}Pを計算
        // K^-1の代わりにA^{-1}だと思うと，
        // Az = P と解くと z = A^{-1}P になる
        MatrixXd K_inv_P =
            block_bicgstab_eigen_float(
                A.cast<float>(), P.cast<float>(),
                MatrixXd::Zero(A.cols(), P.cols()).cast<float>(), 20, 1e-5)
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
            block_bicgstab_eigen_float(
                A.cast<float>(), T.cast<float>(),
                MatrixXd::Zero(A.cols(), T.cols()).cast<float>(), 20, 1e-5)
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
    cout << "反復回数: " << res_norms.size() - 1
         << ", ||B-AX||/||B|| = " << scientific << setprecision(6)
         << rel_res_norm << endl;

    return X;
}

MatrixXd block_bicgstab_eigen_double(const SparseMatrix<double>& A,
                                     const MatrixXd& B, MatrixXd X,
                                     int max_iter = 1000, double tol = 1e-15) {
    MatrixXd R = B - A * X;
    MatrixXd P = R;
    MatrixXd R_hat_zero = R;
    MatrixXd R_zero = R;

    vector<double> res_norms;

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

    // 残差を出す
    auto R1 = B - A * X;
    double rel_res_norm = R1.norm() / B.norm();
    cout << "反復回数: " << res_norms.size() - 1
         << ", ||B-AX||/||B|| = " << scientific << setprecision(6)
         << rel_res_norm << endl;

    return X;
}

// pdeが解きやすい
int main() {
    vector<int> test_cols = {1, 3, 5, 10, 20, 30, 40, 50};

    cout << test_cols.size() << endl;
    for (int m : test_cols) {
        // --- 問題の生成 ---
        SparseMatrix<double> A;
        MatrixXd X_true, B;
        generate_problem(m, A, X_true, B);

        // --- 初期解の生成 ---
        MatrixXd X0 = MatrixXd::Zero(A.cols(), m);
        cout << "m = " << m << endl;

        // --- Block BiCGSTAB法の適用 ---
        auto start = chrono::high_resolution_clock::now();
        cout << "Block BiCGSTAB法の結果: " << endl;
        MatrixXd X_sol = block_bicgstab_eigen_double(A, B, X0, 1000, 1e-10);
        auto end = chrono::high_resolution_clock::now();
        chrono::duration<double> elapsed = end - start;
        cout << "計算時間: " << elapsed.count() << " 秒" << endl;
        MatrixXd R = B - A * X_sol;
        cout << endl;

        auto start2 = chrono::high_resolution_clock::now();
        cout << "可変前処理付き Block BiCGSTAB法の結果: " << endl;
        X_sol = block_bicgstab_eigen_preprocessing(A, B, X0, 1000, 1e-10);
        auto end2 = chrono::high_resolution_clock::now();
        chrono::duration<double> elapsed2 = end2 - start2;
        cout << "計算時間: " << elapsed2.count() << " 秒" << endl;
        cout << endl;

        cout << "可変ではない前処理付き Block BiCGSTAB法の結果: " << endl;
        auto start3 = chrono::high_resolution_clock::now();
        X_sol = block_bicgstab_eigen_preprocessing2(A, B, X0, 1000, 1e-10);
        auto end3 = chrono::high_resolution_clock::now();
        chrono::duration<double> elapsed3 = end3 - start3;
        cout << "計算時間: " << elapsed3.count() << " 秒" << endl;

        cout << endl;
    }

    return 0;
}