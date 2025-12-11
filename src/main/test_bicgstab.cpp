// ファイル名: main.cpp

#include <Eigen/Cholesky>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <iostream>

#include "algo/bicgstab.hpp"
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
    cout << "使用する行列: " << path << endl;
    cout << "行列のサイズ: " << A.rows() << " x " << A.cols() << endl;

    // bicgstabが正しく動くかチェック
    constexpr int check_times = 100;
    int ok_count = 0;
    for (int t = 0; t < check_times; ++t) {
        VectorXd b;
        auto true_x = generate_answer(A, b);
        vector<double> res_norms;
        cout << "BiCGSTAB法の動作確認" << endl;
        auto x =
            bicgstab(A, b, VectorXd::Zero(A.cols()), 1000, 1e-10, res_norms);
        cout << "反復回数: " << res_norms.size() - 1 << endl;
        double error = (A * x - b).norm() / b.norm();
        cout << "相対誤差: ||b-ax||/||b|| = " << error << endl;
        if (error < 1e-10) {
            ++ok_count;
        }
        cout << endl;
    }
    cout << "テスト結果: " << ok_count << " / " << check_times << " 回成功"
         << endl;
    return 0;
}
