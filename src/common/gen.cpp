#include "common/gen.hpp"

#include <random>

#include "io_utils.hpp"

using namespace std;
using namespace Eigen;
constexpr unsigned int fixed_seed = 42;
std::mt19937 gen(fixed_seed);

MatrixXd generate_dense_matrix(int rows, int cols) {
    MatrixXd mat = MatrixXd::Zero(rows, cols);
    std::uniform_real_distribution<double> real_dist(0.0, 1.0);
    for (int i = 0; i < rows * cols; ++i) {
        mat(i) = real_dist(gen);
    }
    return mat;
}

MatrixXd generate_answer(const SparseMatrix<double>& A, MatrixXd& B, int col) {
    // AX = B となる真の解true_Xをランダムに生成
    MatrixXd true_X = generate_dense_matrix(A.cols(), col);
    B = A * true_X;
    return true_X;
}