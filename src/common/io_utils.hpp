#include <Eigen/Sparse>

// 関数の「宣言」のみを記述。中身({})は書かない。
Eigen::SparseMatrix<double> read_sparse_matrix(const std::string& filename);
