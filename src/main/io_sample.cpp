// ファイル名: main.cpp

#include <Eigen/Cholesky>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <iostream>

#include "common/io_utils.hpp"

using namespace std;
using namespace Eigen;

int main() {
    auto A = read_sparse_matrix("in/matrix/simple.mtx");
    cout << A << endl;
    return 0;
}