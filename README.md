# kdtree-eigen

![Cpp11](https://img.shields.io/badge/C%2B%2B-11-blue.svg)
![License](https://img.shields.io/packagist/l/doctrine/orm.svg)
![Travis Status](https://travis-ci.org/Rookfighter/kdtree-eigen.svg?branch=master)
![Appveyer Status](https://ci.appveyor.com/api/projects/status/r52757j9k4uybfu6?svg=true)

kdtree-eigen is a header-only C++ library for KNN nearest neighbour search. It
implements various interfaces for KNN search using the ```Eigen3``` library.

## Install

Simply copy the header files you need into your project or any directory such
that your build system can find them.

## Usage

There are different header files for different use cases:

```kdtree_eigen.h``` implements a pure ```Eigen3``` KDTree for KNN-search based
on the paper [1] of Maneewongvatana and Mount. This only requires that ```Eigen3```
can be found by your build system.

```kdtree_flann.h``` is a wrapper for the ```FLANN``` library such that it
can be easily and efficiently used with Eigen3. This header assumes that ```Eigen3```
and ```FLANN``` can be found by your build system.

All kdtrees share a similar interface. Here is a basic example on how to build a
KDTree and query it.

```cpp
#include <kdtree_eigen.h>

int main()
{
    // create Eigen::MatrixXd dataPoints
    // each column is a single data point
    // ...

    // Create a KDTreed object with double precision,
    // you could also use KDTreef or KDTree<MyType>
    // set the data points with the constructor
    // optionally you can also use the setData() method
    // setting data is fast
    kdt::KDTreed kdtree(datapoints);

    // build kdtree, this consumes time
    kdtree.build();

    // create Eigen::MatrixXd queryPoints
    // each column is a single query point
    // each query point must have the same dimension as in dataPoints
    // ...

    // initialize result matrices
    Eigen::MatrixXd dists;
    Eigen::MatrixXi idx;
    size_t knn = 10;

    // query kdtree
    // idx and dists have the shape knn x queryPoints.cols()
    // each column correspond to one query point
    // if less than knn neighbours were found for a query point,
    // the remaining idx fields will be set to -1
    kdtree.query(queryPoints, knn, idx, dists);
}
```

The ```FLANN``` kdtree works analogously with the class ```kdt::KDTreeFlann```.

## References

1. *Songrit Maneewongvatana and David M. Mount*, **Analysis of Approximate
Nearest Neighbor Searching with Clustered Point Sets**, DIMACS Series in
Discrete Mathematics and Theoretical Computer Science, 2002
