
#include <Eigen/Dense>
#include <Eigen/Sparse>

using namespace Eigen;
MatrixXd block_bicgstab(const SparseMatrix<double>& A, const MatrixXd& B,
                        MatrixXd X, int max_iter, double tol);

// 単精度
MatrixXf block_bicgstab_float(const SparseMatrix<float>& A, const MatrixXf& B,
                              MatrixXf X, int max_iter, double tol);

// QR分解による安定化
MatrixXd block_bicgstab_rq(const SparseMatrix<double>& A, const MatrixXd& B,
                           MatrixXd X, int max_iter, double tol);

// QR分解+modifiedによる安定化
MatrixXd block_bicgstab_modrq(const SparseMatrix<double>& A, const MatrixXd& B,
                              MatrixXd X, int max_iter, double tol);

// 不完全コレスキー分解による前処理付き
MatrixXd block_bicgstab_preprocessing(const SparseMatrix<double>& A,
                                      const MatrixXd& B, MatrixXd X,
                                      int max_iter, double tol);
// 不完全コレスキー分解による前処理付き，かつQR分解による安定化
MatrixXd block_bicgstab_preprocessing_rq(const SparseMatrix<double>& A,
                                         const MatrixXd& B, MatrixXd X,
                                         int max_iter, double tol);

// 可変前処理(反復法のネスト)付き
MatrixXd block_bicgstab_dynamic_preprocessing(const SparseMatrix<double>& A,
                                              const MatrixXd& B, MatrixXd X,
                                              int max_iter, double tol);
// 可変前処理(反復法のネスト)付き，かつQR分解による安定化
MatrixXd block_bicgstab_dynamic_preprocessing_rq(const SparseMatrix<double>& A,
                                                 const MatrixXd& B, MatrixXd X,
                                                 int max_iter, double tol);