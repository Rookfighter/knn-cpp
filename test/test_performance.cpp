/* test_performance.cpp
 *
 * Author: Fabian Meyer
 * Created On: 30 Jan 2019
 */

#include <kdtree_eigen.h>
#include <string>
#include <fstream>
#include <sstream>
#include <chrono>
#include <iostream>

typedef double Scalar;
typedef kdt::KDTree<Scalar> KDTree;

static void loadMatrix(const std::string &filename, KDTree::Matrix &mat)
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

int main(int argc, char** argv)
{
    if(argc != 2)
    {
        std::cerr << "specify data file" << std::endl;
        return -1;
    }

    std::string filename = argv[1];
    KDTree::Matrix mat;
    std::cout << "Loading " << filename << std::endl;
    loadMatrix(filename, mat);

    std::cout << "Matrix (" << mat.rows() << "," << mat.cols() << ")" << std::endl;
    std:: cout << mat.block(0, 0, 3, 10) << std::endl;

    KDTree kdtree(mat);
    kdtree.setSorted(true);
    kdtree.setBalanced(false);
    kdtree.setCompact(true);
    kdtree.setMaxDistance(0.5);
    kdtree.setThreads(0);

    std::cout << "Building kdtree" << std::endl;
    auto start = std::chrono::steady_clock::now();
    kdtree.build();
    auto end = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout << "-- Took " << duration / 1e6 << "s." << std::endl;

    KDTree::Matrix dists;
    KDTree::MatrixI idxs;
    std::cout << "Querying kdtree" << std::endl;
    start = std::chrono::steady_clock::now();
    kdtree.query(mat, 20, idxs, dists);
    end = std::chrono::steady_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout << "-- Took " << duration / 1e6 << "s." << std::endl;

    return 0;
}
