#include <Eigen/Cholesky>
#include <Eigen/Dense>
#include <Eigen/Sparse>

using namespace std;
using namespace Eigen;

MatrixXd generate_dense_matrix(int rows, int cols);
MatrixXd generate_answer(const SparseMatrix<double>& A, MatrixXd& B, int col);
