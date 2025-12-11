#include <Eigen/Cholesky>
#include <Eigen/Dense>
#include <Eigen/Sparse>

using namespace std;
using namespace Eigen;

MatrixXd generate_dense_matrix(int rows, int cols);
VectorXd generate_dense_vector(int size);
MatrixXd generate_answer(const SparseMatrix<double>& A, MatrixXd& B, int col);
VectorXd generate_answer(const SparseMatrix<double>& A, VectorXd& b);
