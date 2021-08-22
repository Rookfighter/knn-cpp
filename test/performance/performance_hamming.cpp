/* test_performance.cpp
 *
 * Author: Fabian Meyer
 * Created On: 30 Jan 2019
 */

#include <knncpp.h>
#include <string>
#include <fstream>
#include <sstream>
#include <chrono>
#include <iostream>

typedef int32_t Scalar;
typedef knncpp::Index Index;
typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;

static void loadMatrix(const std::string &filename, Matrix &mat)
{
    std::ifstream is(filename);

    size_t lineCount = static_cast<size_t>(std::count(
                std::istreambuf_iterator<char>(is),
                std::istreambuf_iterator<char>(),
                '\n') + 1);

    mat.resize(3, lineCount);

    is.clear();
    is.seekg(0, std::ios::beg);

    size_t row = 0;
    std::string line;
    while(!is.eof() && !is.fail())
    {
        std::getline(is, line, '\n');

        Scalar a, b, c;
        std::istringstream istr(line);
        istr >> a >> b >> c;
        mat.col(row) << a, b, c;

        ++row;
    }
}

template<typename Tree>
void testPerformance(Tree &kdtree, Matrix &mat)
{
    std::cout << "-- build index" << std::endl;
    auto start = std::chrono::steady_clock::now();
    kdtree.build();
    auto end = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout << "-- Took " << duration / 1e6 << "s." << std::endl;

    Matrix dists;
    typename Tree::Matrixi idxs;
    std::cout << "-- query index" << std::endl;
    start = std::chrono::steady_clock::now();
    kdtree.query(mat, 20, idxs, dists);
    end = std::chrono::steady_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout << "-- Took " << duration / 1e6 << "s." << std::endl;
}

int main(int argc, char** argv)
{
    if(argc != 2)
    {
        std::cerr << "specify data file" << std::endl;
        return -1;
    }

    std::string filename = argv[1];
    Matrix mat;
    std::cout << "Loading " << filename << std::endl;
    loadMatrix(filename, mat);

    std::cout << "Matrix (" << mat.rows() << "," << mat.cols() << ")" << std::endl;
    std:: cout << mat.block(0, 0, 3, 10) << std::endl;

    std::cout << "BruteForce" << std::endl;
    knncpp::BruteForce<Scalar, knncpp::HammingDistance<Scalar>> bf(mat);
    bf.setSorted(true);
    bf.setMaxDistance(120);
    bf.setThreads(0);
    bf.setTakeRoot(false);
    testPerformance(bf, mat);

    std::cout << "MultiIndexHashing" << std::endl;
    knncpp::MultiIndexHashing<Scalar> mih(mat);
    mih.setSorted(true);
    mih.setMaxDistance(120);
    mih.setThreads(0);

    testPerformance(mih, mat);

    return 0;
}
