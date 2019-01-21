/* test_kdtree_eigen.cpp
 *
 *     Author: Fabian Meyer
 * Created On: 08 Nov 2018
 */

#include <catch.hpp>
#include <kdtree_eigen.h>

using namespace kdt;

TEST_CASE("KDTree")
{
    KDTreed kdtree;

    SECTION("build normal")
    {
        Eigen::MatrixXd data(3, 3);
        data << 1, 3, 0,
            0, 1, 2,
            3, 2, 0;
        kdtree.setBucketSize(2);
        kdtree.setData(data);
        kdtree.build();

        REQUIRE(kdtree.size() == 3);
        REQUIRE(kdtree.tree() != nullptr);
        REQUIRE(kdtree.tree()->left != nullptr);
        REQUIRE(kdtree.tree()->left->isLeaf());
        REQUIRE(kdtree.tree()->left->idx.size() == 2);
        REQUIRE(kdtree.tree()->left->idx(0) == 0);
        REQUIRE(kdtree.tree()->left->idx(1) == 2);

        REQUIRE(kdtree.tree()->right != nullptr);
        REQUIRE(kdtree.tree()->right->isLeaf());
        REQUIRE(kdtree.tree()->right->idx.size() == 1);
        REQUIRE(kdtree.tree()->right->idx(0) == 1);
    }

    SECTION("build no data")
    {
        REQUIRE_THROWS(kdtree.build());
    }

    SECTION("build empty")
    {
        Eigen::MatrixXd data(3, 0);

        kdtree.setData(data);
        REQUIRE(kdtree.size() == 0);
        REQUIRE_THROWS(kdtree.build());
    }

    SECTION("build parallel")
    {
        Eigen::MatrixXd data(3, 9);
        data << 1, 2, 3, 1, 2, 3, 1, 2, 3,
                2, 1, 0, 3, 2, 1, 0, 3, 0,
                2, 1, 3, 1, 2, 2, 3, 2, 1;

        kdtree.setData(data);
        kdtree.setThreads(4);
        kdtree.build();

        REQUIRE(kdtree.tree() != nullptr);
    }

    SECTION("query one")
    {
        Eigen::MatrixXd data(3, 3);
        data << 1, 3, 0,
            0, 1, 2,
            3, 2, 0;

        kdtree.setBucketSize(2);
        kdtree.setData(data);
        kdtree.build();
        REQUIRE(kdtree.size() == 3);

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

    SECTION("query all")
    {
        Eigen::MatrixXd data(3, 9);
        data << 1, 2, 3, 1, 2, 3, 1, 2, 3,
                2, 1, 0, 3, 2, 1, 0, 3, 0,
                2, 1, 3, 1, 2, 2, 3, 2, 1;

        kdtree.setBucketSize(2);
        kdtree.setData(data);
        kdtree.build();

        REQUIRE(kdtree.size() == 9);

        Eigen::MatrixXd points(3, 1);
        points << 0, 1, 0;

        Eigen::MatrixXi indices;
        Eigen::MatrixXd distances;
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
        Eigen::MatrixXd data(3, 3);
        data << 1, 3, 0,
            0, 1, 2,
            3, 2, 0;
        kdtree.setBucketSize(2);
        kdtree.setData(data);
        kdtree.setMaxDistance(1.5);
        kdtree.build();

        REQUIRE(kdtree.size() == 3);

        Eigen::MatrixXd points(3, 1);
        points << 0, 1, 0;
        Eigen::MatrixXi indices;
        Eigen::MatrixXd distances;

        kdtree.query(points, 2, indices, distances);

        REQUIRE(indices.cols() == 1);
        REQUIRE(indices.rows() == 2);
        REQUIRE(indices(0, 0) == 2);
        REQUIRE(indices(1, 0) == -1);
        REQUIRE(distances.cols() == 1);
        REQUIRE(distances.rows() == 2);
        REQUIRE(distances(0, 0) == Approx(1.0));
        REQUIRE(distances(1, 0) == 0.0);
    }
}
