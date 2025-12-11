
#include "algo/bicgstab.hpp"

using namespace std;
#include <vector>

VectorXd bicgstab(const SparseMatrix<double>& A, const VectorXd& b, VectorXd x,
                  int max_iter, double tol, vector<double>& res_norms) {
    VectorXd r0 = b - A * x;
    VectorXd r0_star = r0;  // (r0と直交しない任意のベクトル)
    VectorXd r = r0;
    VectorXd p = r;
    res_norms.clear();

    double b_norm = b.norm();
    if (b_norm == 0.0) b_norm = 1.0;

    res_norms.push_back(r.norm() / b_norm);
    for (int i = 0; i < max_iter; i++) {
        // Ap = A * p
        VectorXd Ap = A * p;
        double alpha =
            r0_star.dot(r) / r0_star.dot(Ap);  // r0_star^T * A * p の計算は省略
        VectorXd t = r - alpha * Ap;
        VectorXd At = A * t;
        double zeta = At.dot(t) / At.dot(At);
        x += alpha * p + zeta * t;
        VectorXd r1 = t - zeta * At;
        double r1_norm = r1.norm();
        res_norms.push_back(r1_norm / b_norm);
        if (r1_norm / b_norm < tol) {
            break;
        }
        double beta = alpha / zeta * r0_star.dot(r1) / r0_star.dot(r);
        p = r1 + beta * (p - zeta * Ap);
        r = r1;
    }
    return x;
}