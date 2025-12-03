
#include <Eigen/Cholesky>
#include <Eigen/Dense>
#include <Eigen/Sparse>

using namespace std;
using namespace Eigen;

bool CholeskyQR(const MatrixXd& A, MatrixXd& Q, MatrixXd& R);
bool CholeskyQR(const MatrixXf& A, MatrixXf& Q, MatrixXf& R);
