/* test_performance.cpp
 *
 * Author: Fabian Meyer
 * Created On: 30 Jan 2019
 */

#include <knn/kdtree_eigen.h>
#include <knn/kdtree_flann.h>
#include <string>
#include <fstream>
#include <sstream>
#include <chrono>
#include <iostream>

typedef double Scalar;
typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
typedef typename Matrix::Index Index;
typedef Eigen::Matrix<Index, Eigen::Dynamic, Eigen::Dynamic> MatrixI;
typedef Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic> Matrixi;

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
    std::cout << "Building kdtree" << std::endl;
    auto start = std::chrono::steady_clock::now();
    kdtree.build();
    auto end = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout << "-- Took " << duration / 1e6 << "s." << std::endl;

    typename Tree::Matrix dists;
    typename Tree::MatrixI idxs;
    std::cout << "Querying kdtree" << std::endl;
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

    knn::KDTree<Scalar> kdtree(mat);
    kdtree.setSorted(true);
    kdtree.setBalanced(false);
    kdtree.setCompact(true);
    kdtree.setMaxDistance(0.5);
    kdtree.setThreads(0);
    kdtree.setTakeRoot(false);
    kdtree.setBucketSize(16);

    std::cout << "KDTree" << std::endl;
    testPerformance(kdtree, mat);


    knn::KDTreeFlann<Scalar> kdtree2(mat);
    kdtree2.setIndexParams(flann::KDTreeSingleIndexParams(16));
    kdtree2.setThreads(0);
    kdtree.setMaxDistance(0.5);

    std::cout << "KDTreeFlann" << std::endl;
    testPerformance(kdtree2, mat);

    return 0;
}
