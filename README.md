# knn-cpp

![Cpp11](https://img.shields.io/badge/C%2B%2B-11-blue.svg)
![License](https://img.shields.io/packagist/l/doctrine/orm.svg)
![CMake](https://github.com/Rookfighter/knn-cpp/workflows/CMake/badge.svg)

```knn-cpp``` is a header-only C++ library for k nearest neighbor search
using the ```Eigen3``` library.

It implements various interfaces for KNN search:

* pure ````Eigen3```` parallelized brute force search
* pure ````Eigen3```` kdtree for efficient search with Manhatten, Euclidean and Minkowski distances

## Install

Simply copy the header files into your project or install them using
the CMake build system by typing

```bash
cd path/to/repo
mkdir build
cd build
cmake ..
make install
```

The library requires ```Eigen3``` to be installed on your system.

In Debian based systems you can simply install these dependencies using ```apt-get```.

```bash
apt-get install libeigen3-dev
```

Make sure ```Eigen3``` can be found by your build system.

You can use the CMake Find module in ```cmake/``` to find the installed headers.

## Usage

All search algorithms share a similar interface. Have a look at the files in the
```examples/``` directory.

Here is a basic example on how to build a kdtree and query it.

```cpp
#include <iostream>
#include <knncpp.h>

typedef Eigen::MatrixXd Matrix;
typedef knncpp::Matrixi Matrixi;

int main()
{
    // Define some data points, which should be searched.
    // Each column defines one datapoint.
    Matrix dataPoints(3, 9);
    dataPoints << 1, 2, 3, 1, 2, 3, 1, 2, 3,
                  2, 1, 0, 3, 2, 1, 0, 3, 4,
                  3, 1, 3, 1, 3, 4, 4, 2, 1;

    // Create a KDTreeMinkowski object and set the data points.
    // Data is not copied by default. You can also pass an additional bool flag
    // to create a data copy. The tree is not built yet.
    // You can also use the setData() method to set the data at a later point.
    // The distance type is defined by the second template parameter.
    // Currently ManhattenDistance, EuclideanDistance, ChebyshevDistance and
    // MinkowskiDistance are available.
    knncpp::KDTreeMinkowskiX<double, knncpp::EuclideanDistance<double>> kdtree(dataPoints);

    // Set the bucket size for each leaf node in the tree. The higher the value
    // the less leafs have to be visited to find the nearest neighbors. The
    // lower the value the less distance evaluations have to be computed.
    // Default is 16.
    kdtree.setBucketSize(16);
    // Set if the resulting neighbors should be sorted in ascending order after
    // a successfull search.
    // This consumes some time during the query.
    // Default is true.
    kdtree.setSorted(true);
    // Set if the root should be taken of the distances after a successful search.
    // This consumes some time during the query.
    // Default is false.
    kdtree.setTakeRoot(true);
    // Set the maximum inclusive distance for the query. Set to 0 or negative
    // to disable maximum distances.
    // Default is 0.
    kdtree.setMaxDistance(2.5 * 2.5);
    // Set how many threads should be used during the query. Set to 0 or
    // negative to autodetect the optimal number of threads (OpenMP only).
    // Default is 1.
    kdtree.setThreads(2);

    // Build the tree. This consumes some time.
    kdtree.build();

    // Create a querypoint. We will search for this points nearest neighbors.
    Matrix queryPoints(3, 1);
    queryPoints << 0, 1, 0;

    Matrixi indices;
    Matrix distances;
    // Search for 3 nearest neighbors.
    // The matrices indices and distances hold the index and distance of the
    // respective nearest neighbors.
    // Their value is set to -1 if no further neighbor was found.
    kdtree.query(queryPoints, 3, indices, distances);

    // Do something with the results.
    std::cout
        << "Data points:" << std::endl
        << dataPoints << std::endl
        << "Query points:" << std::endl
        << queryPoints << std::endl
        << "Neighbor indices:" << std::endl
        << indices << std::endl
        << "Neighbor distances:" << std::endl
        << distances << std::endl;

    return 0;
}
```

## References

1. *Songrit Maneewongvatana and David M. Mount*, **Analysis of Approximate
Nearest Neighbor Searching with Clustered Point Sets**, DIMACS Series in
Discrete Mathematics and Theoretical Computer Science, 2002

2. *Mohammad Norouzi, Ali Punjani and David J. Fleet*, **Fast Search in Hamming
Space with Multi-Index Hashing**, In Proceedings of 2012 IEEE Conference on Computer
Vision and Pattern Recognition
