#include "io_utils.hpp"

#include <fstream>
#include <stdexcept>
#include <vector>

Eigen::SparseMatrix<double> readSparseMatrixFromFile(
    const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Error: Could not open file " + filename);
    }

    int rows, cols, nonZeros;
    file >> rows >> cols >> nonZeros;

    std::vector<Eigen::Triplet<double>> tripletList;
    tripletList.reserve(nonZeros);

    int r, c;
    double val;
    for (int i = 0; i < nonZeros; ++i) {
        file >> r >> c >> val;
        tripletList.push_back(Eigen::Triplet<double>(r - 1, c - 1, val));
    }

    file.close();

    Eigen::SparseMatrix<double> mat(rows, cols);
    mat.setFromTriplets(tripletList.begin(), tripletList.end());

    return mat;
}