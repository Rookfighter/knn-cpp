/* test_kdtree_eigen.cpp
 *
 *     Author: Fabian Meyer
 * Created On: 08 Nov 2018
 */

#include <catch.hpp>
#include <kdtree_flann.h>

using namespace kdt;

TEST_CASE("KDTreeFlann")
{
    KDTreeFlannd kdtree;

    SECTION("query one")
    {
        Eigen::MatrixXd data(3, 3);
        data << 1, 3, 0,
            0, 1, 2,
            3, 2, 0;

        kdtree.setData(data);
        kdtree.build();

        Eigen::MatrixXd points(3, 1);
        points << 0, 1, 0;
        Eigen::MatrixXi indices;
        Eigen::MatrixXd distances;

        kdtree.query(points, 1, indices, distances);

        REQUIRE(indices.size() == 1);
        REQUIRE(indices(0) == 2);
        REQUIRE(distances.size() == 1);
        REQUIRE(distances(0) == Approx(1.0));


    }
}
