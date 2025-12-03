// ファイル名: main.cpp

#include <Eigen/Cholesky>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <iostream>

#include "algo/block_bicgstab.hpp"
#include "common/gen.hpp"
#include "common/io_utils.hpp"

using namespace std;
using namespace Eigen;

int main() {
    auto A = read_sparse_matrix("in/matrix/pde2961.mtx");
    vector<int> col_sizes = {1, 5, 10, 20, 50, 100};
    for (auto col : col_sizes) {
        cout << "=== 列数: " << col << " ===" << endl;
        MatrixXd B;
        auto true_X = generate_answer(A, B, col);
        {
            vector<double> res_norms;
            cout << "Block BiCGSTAB法" << endl;
            auto X = block_bicgstab(A, B, MatrixXd::Zero(A.cols(), col), 1000,
                                    1e-10, res_norms);
            cout << "反復回数: " << res_norms.size() - 1 << endl;
            double error = (true_X - X).norm() / B.norm();
            cout << "相対誤差: ||X-X0||/||B|| = " << error << endl;
            cout << endl;
        }
        {
            vector<double> res_norms;
            cout << "Block BiCGSTAB法 + QR分解安定化" << endl;
            auto X = block_bicgstab_rq(A, B, MatrixXd::Zero(A.cols(), col),
                                       1000, 1e-10, res_norms);
            cout << "反復回数: " << res_norms.size() - 1 << endl;
            double error = (true_X - X).norm() / B.norm();
            cout << "相対誤差: ||X-X0||/||B|| = " << error << endl;
            cout << endl;
        }
        {
            vector<double> res_norms;
            cout << "Block BiCGSTAB法 + 不完全コレスキー分解前処理" << endl;
            auto X = block_bicgstab_preprocessing(
                A, B, MatrixXd::Zero(A.cols(), col), 1000, 1e-10, res_norms);
            cout << "反復回数: " << res_norms.size() - 1 << endl;
            double error = (true_X - X).norm() / B.norm();
            cout << "相対誤差: ||X-X0||/||B|| = " << error << endl;
            cout << endl;
        }
        {
            vector<double> res_norms;
            cout << "Block BiCGSTAB法 + 不完全コレスキー分解前処理 + "
                    "QR分解安定化"
                 << endl;
            auto X = block_bicgstab_preprocessing_rq(
                A, B, MatrixXd::Zero(A.cols(), col), 1000, 1e-10, res_norms);
            cout << "反復回数: " << res_norms.size() - 1 << endl;
            double error = (true_X - X).norm() / B.norm();
            cout << "相対誤差: ||X-X0||/||B|| = " << error << endl;
            cout << endl;
        }
        {
            vector<double> res_norms;
            cout << "Block BiCGSTAB法 + 可変前処理" << endl;
            auto X = block_bicgstab_dynamic_preprocessing(
                A, B, MatrixXd::Zero(A.cols(), col), 1000, 1e-10, res_norms,
                100, 1e-3);
            cout << "反復回数: " << res_norms.size() - 1 << endl;
            double error = (true_X - X).norm() / B.norm();
            cout << "相対誤差: ||X-X0||/||B|| = " << error << endl;
            cout << endl;
        }
        {
            vector<double> res_norms;
            cout << "Block BiCGSTAB法 + 可変前処理 + QR分解安定化" << endl;
            auto X = block_bicgstab_dynamic_preprocessing_rq(
                A, B, MatrixXd::Zero(A.cols(), col), 1000, 1e-10, res_norms,
                100, 1e-3);
            cout << "反復回数: " << res_norms.size() - 1 << endl;
            double error = (true_X - X).norm() / B.norm();
            cout << "相対誤差: ||X-X0||/||B|| = " << error << endl;
            cout << endl;
        }
    }
}
