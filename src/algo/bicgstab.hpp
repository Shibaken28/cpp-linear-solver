
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <vector>

using namespace std;
using namespace Eigen;
VectorXd bicgstab(const SparseMatrix<double>& A, const VectorXd& b, VectorXd x,
                  int max_iter, double tol, vector<double>& res_norms);
