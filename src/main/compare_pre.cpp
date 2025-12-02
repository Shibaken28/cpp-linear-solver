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
    auto A = read_sparse_matrix("in/matrix/1138_bus.mtx");
    vector<int> col_sizes = {1, 5, 10, 30};
    for (auto col : col_sizes) {
        cout << "=== 列数: " << col << " ===" << endl;
        MatrixXd B;
        auto true_X = generate_answer(A, B, col);
        {
            cout << "Block BiCGSTAB法 + 前処理あり + QR分解安定化" << endl;
            auto X = block_bicgstab_preprocessing_rq(
                A, B, MatrixXd::Zero(A.cols(), col), 1000, 1e-10);
            double error = (true_X - X).norm() / B.norm();
            cout << "相対誤差: " << error << endl;
        }
        {
            cout << "Block BiCGSTAB法 + QR分解安定化" << endl;
            auto X = block_bicgstab_rq(A, B, MatrixXd::Zero(A.cols(), col),
                                       1000, 1e-10);
            double error = (true_X - X).norm() / B.norm();
            cout << "相対誤差: " << error << endl;
        }
    }
    return 0;
}
