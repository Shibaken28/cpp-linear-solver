// ファイル名: io_utils.hpp

#ifndef IO_UTILS_HPP
#define IO_UTILS_HPP

#include <Eigen/Sparse>
#include <string>

// 関数の「宣言」のみを記述。中身({})は書かない。
Eigen::SparseMatrix<double> read_sparse_matrix(const std::string& filename);

#endif  // IO_UTILS_HPP