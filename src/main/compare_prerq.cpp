// ファイル名: main.cpp

#include <Eigen/Cholesky>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <iostream>

#include "algo/block_bicgstab.hpp"
#include "common/gen.hpp"
#include "common/io_utils.hpp"

using namespace std;
using namespace Eigen;

int main(int argc, char* argv[]) {
    // 引数の数をチェック
    // (argv[0]はプログラム名なので、引数が1つあるとargcは2になります)
    if (argc < 2) {
        std::cerr << "エラー: 行列ファイルのパスを指定してください。"
                  << std::endl;
        std::cerr << "使用法: " << argv[0] << " <path/to/matrix.mtx>"
                  << std::endl;
        return 1;  // エラー終了
    }

    // コマンドライン引数の1番目を取得
    std::string path = argv[1];
    auto A = read_sparse_matrix(path);
    vector<int> col_sizes = {1, 5, 10, 20, 50, 100, 150, 200};
    cout << "使用する行列: " << path << endl;
    cout << "行列のサイズ: " << A.rows() << " x " << A.cols() << endl;
    for (auto col : col_sizes) {
        cout << "=== 列数: " << col << " ===" << endl;
        MatrixXd B;
        auto true_X = generate_answer(A, B, col);
        {
            vector<double> res_norms;
            cout << "Block BiCGSTAB法 + QR分解安定化" << endl;
            auto X = block_bicgstab_rq(A, B, MatrixXd::Zero(A.cols(), col),
                                       3000, 1e-10, res_norms);
            cout << "反復回数: " << res_norms.size() - 1 << endl;
            double error = (A * X - B).norm() / B.norm();
            cout << "相対誤差: ||B-AX||/||B|| = " << error << endl;
            cout << endl;
        }
        {
            vector<double> res_norms;
            cout << "Block BiCGSTAB法 + 不完全コレスキー分解前処理" << endl;
            auto X = block_bicgstab_preprocessing(
                A, B, MatrixXd::Zero(A.cols(), col), 3000, 1e-10, res_norms);
            cout << "反復回数: " << res_norms.size() - 1 << endl;
            double error = (A * X - B).norm() / B.norm();
            cout << "相対誤差: ||B-AX||/||B|| = " << error << endl;
            cout << endl;
        }
        {
            vector<double> res_norms;
            cout << "Block BiCGSTAB法 + 不完全コレスキー分解前処理 + "
                    "QR分解安定化"
                 << endl;
            auto X = block_bicgstab_preprocessing_rq(
                A, B, MatrixXd::Zero(A.cols(), col), 3000, 1e-10, res_norms);
            cout << "反復回数: " << res_norms.size() - 1 << endl;
            double error = (A * X - B).norm() / B.norm();
            cout << "相対誤差: ||B-AX||/||B|| = " << error << endl;
            cout << endl;
        }
        {
            vector<double> res_norms;
            cout << "Block BiCGSTAB法 + 可変前処理"
                    "内部反復50回or1e-5誤差"
                 << endl;
            auto X = block_bicgstab_dynamic_preprocessing(
                A, B, MatrixXd::Zero(A.cols(), col), 3000, 1e-10, res_norms, 50,
                1e-5);
            cout << "反復回数: " << res_norms.size() - 1 << endl;
            double error = (A * X - B).norm() / B.norm();
            cout << "相対誤差: ||B-AX||/||B|| = " << error << endl;
            cout << endl;
        }
        {
            vector<double> res_norms;
            cout << "Block BiCGSTAB法 + 可変前処理 + QR分解安定化 "
                    "内部反復10回or1e-5誤差"
                 << endl;
            auto X = block_bicgstab_dynamic_preprocessing_rq(
                A, B, MatrixXd::Zero(A.cols(), col), 3000, 1e-10, res_norms, 10,
                1e-5);
            cout << "反復回数: " << res_norms.size() - 1 << endl;
            double error = (A * X - B).norm() / B.norm();
            cout << "相対誤差: ||B-AX||/||B|| = " << error << endl;
            cout << endl;
        }
        {
            vector<double> res_norms;
            cout << "Block BiCGSTAB法 + 可変前処理 + QR分解安定化 "
                    "内部反復50回or1e-5誤差"
                 << endl;
            auto X = block_bicgstab_dynamic_preprocessing_rq(
                A, B, MatrixXd::Zero(A.cols(), col), 3000, 1e-10, res_norms, 50,
                1e-5);
            cout << "反復回数: " << res_norms.size() - 1 << endl;
            double error = (A * X - B).norm() / B.norm();
            cout << "相対誤差: ||B-AX||/||B|| = " << error << endl;
            cout << endl;
        }
        {
            vector<double> res_norms;
            cout << "Block BiCGSTAB法 + 可変前処理 + QR分解安定化 "
                    "内部反復300回or1e-5誤差"
                 << endl;
            auto X = block_bicgstab_dynamic_preprocessing_rq(
                A, B, MatrixXd::Zero(A.cols(), col), 3000, 1e-10, res_norms,
                300, 1e-5);
            cout << "反復回数: " << res_norms.size() - 1 << endl;
            double error = (A * X - B).norm() / B.norm();
            cout << "相対誤差: ||B-AX||/||B|| = " << error << endl;
            cout << endl;
        }
    }
}

/*
内部でもqr安定化するように


色々試した結果


可変前処理が全然安定せず，発散してしまう
内部の反復回数を増やすと正しい答えは出るがそれは前処理を使わないのと同じ意味になってしまう

列数が増えると収束が早くなるから，前処理でほぼ答えが得られる


前処理で1本ずつ解いてみるとどうなる？
-> うまくいけばglobal krylov


メールをかえす
*/