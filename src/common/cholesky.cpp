#include "common/cholesky.hpp"

#include <iostream>

bool CholeskyQR(const MatrixXd& A, MatrixXd& Q, MatrixXd& R) {
    MatrixXd AtA = A.transpose() * A;
    LLT<MatrixXd> llt(AtA);

    if (llt.info() != Eigen::Success) {
        cerr << "コレスキー分解失敗" << endl;
        return false;
    }
    R = llt.matrixU();
    auto R_inv = R.inverse();
    Q = A * R_inv;
    return true;
}

bool CholeskyQR(const MatrixXf& A, MatrixXf& Q, MatrixXf& R) {
    MatrixXf AtA = A.transpose() * A;
    LLT<MatrixXf> llt(AtA);

    if (llt.info() != Eigen::Success) {
        cerr << "コレスキー分解失敗" << endl;
        return false;
    }
    R = llt.matrixU();
    auto R_inv = R.inverse();
    Q = A * R_inv;
    return true;
}
