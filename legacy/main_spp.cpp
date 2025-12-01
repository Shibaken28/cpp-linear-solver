// ファイル名: main.cpp

#include <Eigen/Cholesky>
#include <Eigen/Dense>
#include <Eigen/Sparse>
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

MatrixXd block_bicgstab_eigen_rq(const SparseMatrix<double>& A,
                                 const MatrixXd& B, MatrixXd X);

MatrixXd block_bicgstab_eigen_modrq(const SparseMatrix<double>& A,
                                    const MatrixXd& B, MatrixXd X);

MatrixXd generate_dense_matrix(int rows, int cols) {
    MatrixXd mat = MatrixXd::Zero(rows, cols);
    std::uniform_real_distribution<double> real_dist(0.0, 1.0);
    for (int i = 0; i < rows * cols; ++i) {
        mat(i) = real_dist(gen);
    }
    return mat;
}

void generate_saddle_point_problem(int m, SparseMatrix<double>& A, MatrixXd& B,
                                   MatrixXd& CT) {
    A = read_sparse_matrix("matrix/pde900.mtx");
    // A = read_sparse_matrix("matrix/simple.mtx");
    int n = A.rows();

    // ランダムな密行列BとCを生成
    B = generate_dense_matrix(n, m);
    CT = generate_dense_matrix(m, n);
}

void composite_saddle_point_matrix(const SparseMatrix<double>& A,
                                   const MatrixXd& B, const MatrixXd& CT,
                                   SparseMatrix<double>& S) {
    int n = A.rows();
    int m = B.cols();

    S.resize(n + m, n + m);
    std::vector<Triplet<double>> triplets;
    triplets.reserve(A.nonZeros() + B.nonZeros() + CT.nonZeros());

    // Aの要素を追加
    for (int k = 0; k < A.outerSize(); ++k) {
        for (SparseMatrix<double>::InnerIterator it(A, k); it; ++it) {
            triplets.push_back(Triplet<double>(it.row(), it.col(), it.value()));
        }
    }

    // Bの要素を追加
    for (int i = 0; i < B.rows(); ++i) {
        for (int j = 0; j < B.cols(); ++j) {
            if (B(i, j) != 0.0) {
                triplets.push_back(Triplet<double>(i, j + n, B(i, j)));
            }
        }
    }

    // CTの要素を追加
    for (int i = 0; i < CT.rows(); ++i) {
        for (int j = 0; j < CT.cols(); ++j) {
            if (CT(i, j) != 0.0) {
                triplets.push_back(Triplet<double>(i + n, j, CT(i, j)));
            }
        }
    }

    S.setFromTriplets(triplets.begin(), triplets.end());
}

void generate_saddle_point_problem_answer(const SparseMatrix<double>& S,
                                          MatrixXd& true_X, MatrixXd& B) {
    // SX = B となる真の解true_Xをランダムに生成
    int total_size = S.rows();
    true_X = generate_dense_matrix(total_size, 1);
    B = S * true_X;
}

void solve_alt(const SparseMatrix<double>& A, const MatrixXd& B,
               const MatrixXd& CT, const MatrixXd& f, const MatrixXd& g,
               const SparseMatrix<double>& full_A, const MatrixXd& full_B) {
    // すべての行列のサイズを出力

    int n = A.rows();
    int m = B.cols();
    MatrixXd B_new(n, 1 + m);
    B_new.leftCols(m) = B;
    B_new.rightCols(1) = f;
    // これを解く
    MatrixXd X_new = MatrixXd::Zero(n, 1 + m);
    MatrixXd X = block_bicgstab_eigen_modrq(A, B_new, X_new);

    // X -> [U v]に分割
    MatrixXd U = X.leftCols(m);
    MatrixXd v = X.rightCols(1);
    // S = CT U, t = CT v - g
    MatrixXd S = CT * U;
    MatrixXd t = CT * v - g;
    // Sy = t を解く
    MatrixXd y = S.partialPivLu().solve(t);
    // x = v - U y
    MatrixXd x = v - U * y;

    // 最終的な答え [x; y]
    MatrixXd final_X(n + m, 1);
    final_X.topRows(n) = x;
    final_X.bottomRows(m) = y;
    // 検算
    double final_res_norm = (full_B - full_A * final_X).norm();
    double b_norm = full_B.norm();
    if (b_norm == 0.0) b_norm = 1.0;
    cerr << "真の残差 ||B - A X||: " << scientific << setprecision(6)
         << final_res_norm / b_norm << endl;
}

// pdeが解きやすい
int main() {
    vector<int> test_cols = {1, 3, 5, 10, 20, 30, 40, 50};

    cout << test_cols.size() << endl;
    for (int m : test_cols) {
        // --- 問題の生成 ---
        SparseMatrix<double> A;
        MatrixXd B, CT;
        generate_saddle_point_problem(m, A, B, CT);
        int n = A.rows();
        SparseMatrix<double> full_A;
        composite_saddle_point_matrix(A, B, CT, full_A);
        MatrixXd true_X, full_B;
        generate_saddle_point_problem_answer(full_A, true_X, full_B);
        // full_B を [f g]^Tに分割
        MatrixXd f = full_B.topRows(n);
        MatrixXd g = full_B.bottomRows(m);

        // --- 初期解の設定 ---
        MatrixXd X = MatrixXd::Zero(full_A.rows(), 1);

        cout << "========================================" << endl;
        cout << "naive BiCGSTAB for saddle point problem" << endl;
        cout << "n = " << n << ", m = " << m << endl;
        // 疎行列を引数として関数を呼び出す
        cerr << "BiCGSTAB" << endl;
        block_bicgstab_eigen(full_A, full_B, X);
        cerr << "BiCGSTABrQ" << endl;
        block_bicgstab_eigen_rq(full_A, full_B, X);
        cerr << "BiCGSTABrQ  modified" << endl;
        block_bicgstab_eigen_modrq(full_A, full_B, X);

        cout << "----------------------------------------" << endl;
        cout << "Block-BiCGSTAB for saddle point problem" << endl;
        // B_new = [B f] にする
        solve_alt(A, B, CT, f, g, full_A, full_B);
        cout << "========================================" << endl;
    }

    return 0;
}

MatrixXd block_bicgstab_eigen(const SparseMatrix<double>& A, const MatrixXd& B,
                              MatrixXd X) {
    MatrixXd R = B - A * X;
    MatrixXd P = R;
    MatrixXd R_hat_zero = R;
    MatrixXd R_zero = R;

    vector<double> res_norms;

    double tol = 1e-15;
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
    return X;
}

MatrixXd block_bicgstab_eigen_rq(const SparseMatrix<double>& A,
                                 const MatrixXd& B, MatrixXd X) {
    MatrixXd R0 = B - A * X;
    MatrixXd R, eta;
    CholeskyQR(R0, R, eta);
    MatrixXd R0_tilde = R0;
    MatrixXd P = R;

    vector<double> res_norms;

    double tol = 1e-15;
    int max_iter = 1000;
    double b_norm = B.norm();
    if (b_norm == 0.0) b_norm = 1.0;

    res_norms.push_back(eta.norm() / b_norm);

    for (int i = 0; i < max_iter; i++) {
        // --- 計算ループの中身は、密行列版とほぼ同じ ---

        // APを計算
        MatrixXd AP = A * P;

        // alphaの計算
        MatrixXd R0_tilde_T_A_P = R0_tilde.transpose() * AP;
        MatrixXd R0_tilde_R = R0_tilde.transpose() * R;
        MatrixXd alpha = R0_tilde_T_A_P.lu().solve(R0_tilde_R);

        // Tの計算
        MatrixXd T = R - AP * alpha;

        // ATの計算
        MatrixXd AT = A * T;

        // 5. zetaの計算
        MatrixXd Teta = T * eta;
        MatrixXd ATeta = AT * eta;
        double tr_AT_T = (ATeta.array() * Teta.array()).sum();
        double tr_AT_AT = (ATeta.array() * ATeta.array()).sum();
        double zeta = (tr_AT_AT != 0.0) ? (tr_AT_T / tr_AT_AT) : 0.0;

        // Xの更新
        X += (P * alpha + zeta * T) * eta;

        // R tauの計算
        MatrixXd R1;  // 次のR
        MatrixXd tau;
        MatrixXd t = T - zeta * A * T;
        CholeskyQR(t, R1, tau);

        // betaの計算
        MatrixXd R0_tilde_T_AP = R0_tilde.transpose() * AP;
        MatrixXd R0_tilde_T_R1 = R0_tilde.transpose() * R1;
        MatrixXd beta = (R0_tilde_T_AP * zeta).lu().solve(R0_tilde_T_R1);

        P = R1 + (P - zeta * AP) * beta;
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

MatrixXd block_bicgstab_eigen_modrq(const SparseMatrix<double>& A,
                                    const MatrixXd& B, MatrixXd X) {
    // --- 初期化 ---
    MatrixXd R0 = B - A * X;
    MatrixXd R, eta;
    CholeskyQR(R0, R, eta);
    MatrixXd R0_tilde = R0;
    MatrixXd R0_hat = A.transpose() * R0_tilde;
    MatrixXd P = R;

    vector<double> res_norms;

    double tol = 1e-15;
    int max_iter = 1000;
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

    // --- 検算 ---
    double final_res_norm = (B - A * X).norm();
    cerr << "反復回数: " << res_norms.size()
         << " 真の残差 ||B - AX||: " << scientific << setprecision(6)
         << final_res_norm / b_norm << endl;

    return X;
}
