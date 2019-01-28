/* test_kdtree_eigen.cpp
 *
 *     Author: Fabian Meyer
 * Created On: 08 Nov 2018
 */

#include <catch.hpp>
#include <kdtree_flann.h>

typedef kdt::KDTreeFlannd KDTree;

TEST_CASE("KDTreeFlann")
{
    KDTree kdtree;

    SECTION("query one")
    {
        KDTree::Matrix data(3, 4);
        data << 1, 3, 0, 4,
            0, 1, 2, 3,
            3, 2, 0, 5;

        kdtree.setData(data);
        kdtree.build();

        REQUIRE(kdtree.size() == 4);

        KDTree::Matrix points(3, 1);
        points << 0, 1, 0;
        KDTree::MatrixI indices;
        KDTree::Matrix distances;

        kdtree.query(points, 1, indices, distances);

        REQUIRE(indices.size() == 1);
        REQUIRE(indices(0) == 2);
        REQUIRE(distances.size() == 1);
        REQUIRE(distances(0) == Approx(1.0));
    }

    SECTION("build no data")
    {
        REQUIRE_THROWS(kdtree.build());
    }

    SECTION("build empty")
    {
        KDTree::Matrix data(3, 0);

        kdtree.setData(data);
        REQUIRE(kdtree.size() == 0);
        REQUIRE_THROWS(kdtree.build());
    }

    SECTION("query all")
    {
        KDTree::Matrix data(3, 9);
        data << 1, 2, 3, 1, 2, 3, 1, 2, 3,
                2, 1, 0, 3, 2, 1, 0, 3, 0,
                2, 1, 3, 1, 2, 2, 3, 2, 1;

        kdtree.setData(data);
        kdtree.build();

        REQUIRE(kdtree.size() == 9);

        KDTree::Matrix points(3, 1);
        points << 0, 1, 0;

        KDTree::MatrixI indices;
        KDTree::Matrix distances;
        kdtree.query(points, 9, indices, distances);

        REQUIRE(indices.size() == 9);
        REQUIRE(distances.size() == 9);
        for(Eigen::Index i = 0; i < indices.size(); ++i)
        {
            REQUIRE(indices(i) >= 0);
            REQUIRE(distances(i) > 0);
        }
    }

    SECTION("query maximum distance")
    {
        KDTree::Matrix data(3, 4);
        data << 1, 3, 0, 5,
            0, 1, 3, 4,
            3, 2, 0, 3;

        kdtree.setData(data);
        kdtree.setMaxDistance(4.1);
        kdtree.build();

        REQUIRE(kdtree.size() == 4);

        KDTree::Matrix points(3, 1);
        points << 0, 1, 0;
        KDTree::MatrixI indices;
        KDTree::Matrix distances;

        kdtree.query(points, 2, indices, distances);

        REQUIRE(indices.cols() == 1);
        REQUIRE(indices.rows() == 2);
        REQUIRE(indices(0, 0) == 2);
        REQUIRE(indices(1, 0) == -1);
        REQUIRE(distances.cols() == 1);
        REQUIRE(distances.rows() == 2);
        REQUIRE(distances(0, 0) == Approx(4.0));
        REQUIRE(distances(1, 0) == -1.0);
    }
}
