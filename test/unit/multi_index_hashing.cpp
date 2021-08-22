/* multi_index_hashing.cpp
 *
 *     Author: Fabian Meyer
 * Created On: 30 Oct 2019
 */

#include <knncpp.h>
#include "assert/eigen_require.h"

typedef uint32_t Scalar;
typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
typedef knncpp::Matrixi Matrixi;

TEST_CASE("multi_index_hashing")
{
    SECTION("query one")
    {
        knncpp::MultiIndexHashing<Scalar> mih;
        Matrix data(3, 4);
        data << 1, 3, 0, 5,
                0, 1, 3, 4,
                3, 2, 0, 3;

        mih.setData(data);
        mih.build();

        REQUIRE(mih.size() == 4);

        Matrix points(3, 1);
        points << 0, 1, 0;
        Matrixi indices;
        Matrix distances;

        mih.query(points, 1, indices, distances);

        REQUIRE(indices.size() == 1);
        REQUIRE(indices(0) == 2);
        REQUIRE(distances.size() == 1);
        REQUIRE(distances(0) == Approx(1));
    }

    SECTION("query multiple")
    {
        knncpp::MultiIndexHashing<Scalar> mih;
        Matrix data(3, 9);
        data << 3, 2, 3, 1, 2, 3, 3, 2, 0,
                2, 1, 0, 3, 2, 1, 0, 3, 1,
                3, 1, 3, 1, 3, 3, 4, 5, 1;

        mih.setData(data);
        mih.build();

        REQUIRE(mih.size() == 9);

        Matrix points(3, 1);
        points << 0, 1, 0;

        Matrixi indicesExp(3, 1);
        indicesExp << 8, 1, 3;
        Matrix distancesExp(3, 1);
        distancesExp << 1, 2, 3;

        Matrixi indices;
        Matrix distances;
        mih.query(points, 3, indices, distances);

        REQUIRE(indices.size() == 3);
        REQUIRE(distances.size() == 3);
        REQUIRE_MATRIX(indicesExp, indices);
        REQUIRE_MATRIX(distancesExp, distances);
    }

    SECTION("query max distance")
    {
        knncpp::MultiIndexHashing<Scalar> mih;
        Matrix data(3, 9);
        data << 3, 2, 3, 1, 2, 3, 3, 2, 0,
                2, 1, 0, 3, 2, 1, 0, 3, 1,
                3, 1, 3, 1, 3, 3, 4, 5, 1;

        mih.setMaxDistance(2);
        mih.setData(data);
        mih.build();

        REQUIRE(mih.size() == 9);

        Matrix points(3, 1);
        points << 0, 1, 0;

        Matrixi indicesExp(3, 1);
        indicesExp << 8, 1, -1;
        Matrix distancesExp(3, 1);
        distancesExp << 1, 2, -1;

        Matrixi indices;
        Matrix distances;
        mih.query(points, 3, indices, distances);

        REQUIRE(indices.size() == 3);
        REQUIRE(distances.size() == 3);
        REQUIRE_MATRIX(indicesExp, indices);
        REQUIRE_MATRIX(distancesExp, distances);
    }
}
