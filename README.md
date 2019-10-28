# knn-cpp

![Cpp11](https://img.shields.io/badge/C%2B%2B-11-blue.svg)
![License](https://img.shields.io/packagist/l/doctrine/orm.svg)
![Travis Status](https://travis-ci.org/Rookfighter/kdtree-eigen.svg?branch=master)
![Appveyer Status](https://ci.appveyor.com/api/projects/status/r52757j9k4uybfu6?svg=true)

```knn-cpp``` is a header-only C++ library for KNN nearest neighbour search. It
implements various interfaces for KNN search using the ```Eigen3``` library.

## Install

Simply copy the header files you need into your project or any directory such
that your build system can find them.

## Usage

There are different header files for different use cases:

```kdtree_eigen.h``` implements a pure ```Eigen3``` KDTree for KNN-search based
on [cKDTree](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.cKDTree.html)
by the scipy project and the paper of Maneewongvatana and Mount [1].
This only requires that ```Eigen3``` can be found by your build system.

```kdtree_flann.h``` is a wrapper for the ```FLANN``` library such that it
can be easily and efficiently used with Eigen3. This header assumes that ```Eigen3```
and ```FLANN``` can be found by your build system.

All kdtrees share a similar interface. Here is a basic example on how to build a
KDTree and query it.

```cpp
#include <kdtree_eigen.h>

int main()
{
    // create kdt::KDTreed::Matrix dataPoints
    // each column is a single data point
    // ...

    // Create a KDTreed object with double precision.
    // You could also use KDTreef or KDTree<MyType>.
    // Default distance is euclidean, but you can also
    // use Manhatten or general Minkowski, e.g.
    // KDTree<double, MinkowskiDistance<double, 4>>

    // Set the data points with the constructor.
    // Optionally you can also use the setData() method.
    // Setting data is fast!
    kdt::KDTreed kdtree(dataPoints);

    // Build the kdtree, this consumes time!
    kdtree.build();

    // Create kdt::KDTreed::Matrix queryPoints.
    // Each column is a single query point.
    // Each query point must have the same dimension
    // as in dataPoints.
    // ...

    // Initialize result matrices.
    kdt::KDTreed::Matrix dists; // basically Eigen::MatrixXd
    kdt::KDTreed::MatrixI idx; // basically Eigen::Matrix<Eigen::Index>
    size_t knn = 10;

    // Query the kdtree.
    // idx and dists have the shape knn x queryPoints.cols()
    // Each column corresponds to one query point.
    // If less than knn neighbours were found for a query point,
    // the remaining idx and dists fields will be set to -1.
    kdtree.query(queryPoints, knn, idx, dists);
}
```

The ```FLANN``` kdtree works analogously with the class ```kdt::KDTreeFlann```.

## References

1. *Songrit Maneewongvatana and David M. Mount*, **Analysis of Approximate
Nearest Neighbor Searching with Clustered Point Sets**, DIMACS Series in
Discrete Mathematics and Theoretical Computer Science, 2002
