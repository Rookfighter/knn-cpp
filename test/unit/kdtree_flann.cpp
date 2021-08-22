/* test_kdtree_eigen.cpp
 *
 *     Author: Fabian Meyer
 * Created On: 08 Nov 2018
 */

#ifdef KNNCPP_FLANN

#include <catch2/catch.hpp>
#include <knncpp.h>

typedef double Scalar;
typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
typedef typename knncpp::KDTreeFlann<Scalar>::Matrixi Matrixi;

TEST_CASE("kdtree_flann")
{
    SECTION("query one")
    {
        knncpp::KDTreeFlann<Scalar> kdtree;
        Matrix data(3, 4);
        data << 1, 3, 0, 4,
            0, 1, 2, 3,
            3, 2, 0, 5;

        kdtree.setData(data);
        kdtree.build();

        REQUIRE(kdtree.size() == 4);

        Matrix points(3, 1);
        points << 0, 1, 0;

        Matrixi indices;
        Matrix distances;
        kdtree.query(points, 1, indices, distances);

        REQUIRE(indices.size() == 1);
        REQUIRE(indices(0) == 2);
        REQUIRE(distances.size() == 1);
        REQUIRE(distances(0) == Approx(1.0));
    }

    SECTION("build no data")
    {
        knncpp::KDTreeFlann<Scalar> kdtree;
        REQUIRE_THROWS(kdtree.build());
    }

    SECTION("build empty")
    {
        knncpp::KDTreeFlann<Scalar> kdtree;
        Matrix data(3, 0);

        kdtree.setData(data);
        REQUIRE(kdtree.size() == 0);
        REQUIRE_THROWS(kdtree.build());
    }

    SECTION("query all")
    {
        knncpp::KDTreeFlann<Scalar> kdtree;
        Matrix data(3, 9);
        data << 1, 2, 3, 1, 2, 3, 1, 2, 3,
                2, 1, 0, 3, 2, 1, 0, 3, 0,
                2, 1, 3, 1, 2, 2, 3, 2, 1;

        kdtree.setData(data);
        kdtree.build();

        REQUIRE(kdtree.size() == 9);

        Matrix points(3, 1);
        points << 0, 1, 0;

        Matrixi indices;
        Matrix distances;
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
        knncpp::KDTreeFlann<Scalar> kdtree;
        Matrix data(3, 4);
        data << 1, 3, 0, 5,
            0, 1, 3, 4,
            3, 2, 0, 3;

        kdtree.setData(data);
        kdtree.setMaxDistance(4.1);
        kdtree.build();

        REQUIRE(kdtree.size() == 4);

        Matrix points(3, 1);
        points << 0, 1, 0;
        Matrixi indices;
        Matrix distances;

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

#endif