# kdtree-eigen

kdtree-eigen implements various interfaces using eigen for approximate
k-nearest-neighbour search.

## Install

Simply copy the header files you need into your project or any directory such
that your build system can find them.

## Usage

There are different header files for different use cases:

```kdtree_eigen.h``` implements a pure Eigen3 KDTree for KNN-search based
on the paper [1] of Maneewongvatana and Mount. This only requires that Eigen3
can be found by your build system.

```kdtree_flann.h``` is a wrapper for the FLANN library such that it
can be easily and efficiently used with Eigen3. This header assumes that Eigen3
and FLANN can be found by your build system.

All kdtrees share a similar interface. In the most basic example you create
a KDTree object, set its data points, build it and query it with query points.

```cpp
#include <kdtree_eigen.h>

int main()
{
    // create Eigen::MatrixXd dataPoints
    // each column is a single data point
    // ...

    // set the data points with the constructor
    // optionally you can also use the setData() method
    // setting data is fast
    kdt::KDTreed kdtree(datapoints);

    // build kdtree, this consumes time
    kdtree.build();

    // create Eigen::MatrixXd queryPoints
    // each column is a single data point
    // each data point must have the same dimension as in dataPoints
    // ...

    Eigen::MatrixXd dists;
    Eigen::MatrixXi idx;
    size_t knn = 10;

    // query kdtree resulting idx and dists matrix have the shape
    // knn x queryPoints.cols()
    // if less than knn neighbours were found, the remaining idx will be set
    // to -1
    kdtree.query(queryPoints, knn, idx, dists);
}
```

The FLANN kdtree works analogously with the class ```kdt::KDTreeFlann```.

## References

1. *Songrit Maneewongvatana and David M. Mount*, **Analysis of Approximate
Nearest Neighbor Searching with Clustered Point Sets**, DIMACS Series in
Discrete Mathematics and Theoretical Computer Science, 2002
