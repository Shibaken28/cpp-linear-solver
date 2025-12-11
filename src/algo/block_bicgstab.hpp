
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <vector>

using namespace std;
using namespace Eigen;
MatrixXd block_bicgstab(const SparseMatrix<double>& A, const MatrixXd& B,
                        MatrixXd X, int max_iter, double tol,
                        vector<double>& res_norms);

// 単精度
MatrixXf block_bicgstab_float(const SparseMatrix<float>& A, const MatrixXf& B,
                              MatrixXf X, int max_iter, float tol,
                              vector<float>& res_norms);
// 単精度+QR分解安定化
MatrixXf block_bicgstab_float_rq(const SparseMatrix<float>& A,
                                 const MatrixXf& B, MatrixXf X, int max_iter,
                                 float tol, vector<float>& res_norms);

// QR分解による安定化
MatrixXd block_bicgstab_rq(const SparseMatrix<double>& A, const MatrixXd& B,
                           MatrixXd X, int max_iter, double tol,
                           vector<double>& res_norms);

// QR分解+modifiedによる安定化
MatrixXd block_bicgstab_modrq(const SparseMatrix<double>& A, const MatrixXd& B,
                              MatrixXd X, int max_iter, double tol,
                              vector<double>& res_norms);

// 不完全コレスキー分解による前処理付き
MatrixXd block_bicgstab_preprocessing(const SparseMatrix<double>& A,
                                      const MatrixXd& B, MatrixXd X,
                                      int max_iter, double tol,
                                      vector<double>& res_norms);
// 不完全コレスキー分解による前処理付き，かつQR分解による安定化
MatrixXd block_bicgstab_preprocessing_rq(const SparseMatrix<double>& A,
                                         const MatrixXd& B, MatrixXd X,
                                         int max_iter, double tol,
                                         vector<double>& res_norms);

// 可変前処理(反復法のネスト)付き
MatrixXd block_bicgstab_dynamic_preprocessing(
    const SparseMatrix<double>& A, const MatrixXd& B, MatrixXd X, int max_iter,
    double tol, vector<double>& res_norms, int inner_max_iter, float inner_tol);
// 可変前処理(反復法のネスト)付き，かつQR分解による安定化
MatrixXd block_bicgstab_dynamic_preprocessing_rq(
    const SparseMatrix<double>& A, const MatrixXd& B, MatrixXd X, int max_iter,
    double tol, vector<double>& res_norms, int inner_max_iter, float inner_tol);
// 可変前処理(反復法のネスト)付き，内部で1本ずつ解く版，かつQR分解による安定化
MatrixXd block_bicgstab_dynamic_preprocessing_rq_single(
    const SparseMatrix<double>& A, const MatrixXd& B, MatrixXd X, int max_iter,
    double tol, vector<double>& res_norms, int inner_max_iter, float inner_tol);