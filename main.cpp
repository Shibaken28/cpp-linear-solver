// ファイル名: main.cpp

#include <iostream>

#include "io_utils.hpp"

int main() {
    const std::string filename = "matrix/simple.txt";

    std::cout << "Attempting to read sparse matrix from file..." << std::endl;

    try {
        Eigen::SparseMatrix<double> A = readSparseMatrixFromFile(filename);

        std::cout << "\n--- Success! ---" << std::endl;
        std::cout << "Matrix loaded from file:\n" << A.toDense() << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "An error occurred: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}