/* kdtree_minkowski_search.cpp
 *
 *     Author: Fabian Meyer
 * Created On: 31 Oct 2019
 */

#include <iostream>
#include <knncpp.h>

typedef Eigen::MatrixXd Matrix;
typedef typename knncpp::KDTreeFlann<double>::Matrixi Matrixi;

int main()
{
    // Define some data points, which should be searched.
    // Each column defines one datapoint.
    Matrix dataPoints(3, 9);
    dataPoints << 1, 2, 3, 1, 2, 3, 1, 2, 3,
                  2, 1, 0, 3, 2, 1, 0, 3, 4,
                  3, 1, 3, 1, 3, 4, 4, 2, 1;

    // Create a KDTreeFlann object and set the data points.
    // Data is not copied by default. You can also pass an additional bool flag
    // to create a data copy. The tree is not built yet.
    // You can also use the setData() method to set the data at a later point.
    // The distance type is defined by the second template parameter from FLANN
    // functors.
    knncpp::KDTreeFlann<double, flann::L2_Simple<double>> kdtree(dataPoints);

    // See the FLANN documentation for expalantion of the parameters.
    kdtree.setIndexParams(flann::KDTreeSingleIndexParams(15));
    kdtree.setChecks(32);
    kdtree.setSorted(true);
    kdtree.setMaxDistance(2.5 * 2.5);
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
