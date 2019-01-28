/* test_kdtree_eigen.cpp
 *
 *     Author: Fabian Meyer
 * Created On: 08 Nov 2018
 */

#include <kdtree_eigen.h>
#include <catch.hpp>
#include "eigen_assert.h"

typedef kdt::KDTreed KDTree;

TEST_CASE("KDTree")
{
    KDTree kdtree;

    SECTION("build unbalanced")
    {
        KDTree::Matrix data(3, 4);
        data << 1, 3, 0, 1,
            0, 1, 2, 0,
            3, 2, 0, 1;
        kdtree.setBucketSize(2);
        kdtree.setData(data);
        kdtree.build();

        REQUIRE(kdtree.size() == 4);
        REQUIRE(kdtree.depth() == 3);
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

    SECTION("query one")
    {
        KDTree::Matrix data(3, 4);
        data << 1, 3, 0, 5,
            0, 1, 2, 4,
            3, 2, 0, 3;

        kdtree.setBucketSize(2);
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

    SECTION("query multiple")
    {
        KDTree::Matrix data(3, 9);
        data << 1, 2, 3, 1, 2, 3, 1, 2, 3,
                2, 1, 0, 3, 2, 1, 0, 3, 0,
                2, 1, 3, 1, 2, 2, 3, 2, 1;

        kdtree.setBucketSize(2);
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

    SECTION("query many")
    {
        KDTree::Matrix dataPts(3, 50);
        dataPts << -22.58, -80.33, 42.48, -10.11, -87.03, -40.01, -9.88, 98.56,
            -43.11, -49.37, 1.31, -55.78, -89.13, 54.78, -84.68, 67.34, 17.49,
            -62.44, 24.18, -15.52, -9.47, -66.12, -2.22, 68.41, 90.74, -49.57,
            99.09, 55.18, 72.26, 2.27, -64.53, -63.97, -87.99, -6.89, -82.75,
            70.12, 10.94, -37.09, -7.88, 92.73, -71.91, 3.79, 35.29, 7.67,
            -36.83, 64.22, 30.96, 25.27, -64.83, 53.04,
            -14.61, 0.73, -31.26, -69.45, -38.55, 51.85, 39.4, 48.55, -31.89,
            -16.84, -97.97, -58.06, -8.26, -55.75, 26.04, -24.62, -6.15, 97.48,
            66.16, -3.69, 54.79, 33.01, 18.17, 43.73, -86.75, 29.36, -17.6,
            80.72, 52.04, -51.31, 0.02, 24.51, 43.85, 79.39, 9.67, 23.89,
            14.83, -69.49, -9.15, -85.2, 75.52, -63.4, -21.58, -40.22, -3.33,
            15.32, 8.19, -8.97, 84.4, 9.49,
            -41.04, -97.58, -72.12, 58.85, 90.76, 53.32, 88.52, -1.05, 91.65,
            42.65, 6.65, 46.77, -12.09, 72.38, 7.57, -41.85, -93.3, -63.13,
            -43.69, 65.03, 8.91, -26.22, 77.2, -58.54, -51.48, -76.84, 7.54,
            -30.87, -74.93, 68.33, -19.26, -24.38, 39.45, 64.71, 98.92, -21.59,
            -38.91, 48.53, 93.14, 91.72, 6.88, -31.08, 29.21, -45.23, -70.86,
            -7.52, -28.58, 75.98, -94.6, 8.68;
        KDTree::Matrix queryPts(3, 5);
        queryPts << 63.1, 30.16, 51.24, 78.61, -17.01,
            42.52, 81.1, 24.44, -5.45, -26.54,
            3.92, -80.29, -45.93, 53.99, 41.71;

        size_t knn = 10;
        KDTree::MatrixI indicesExp(knn, queryPts.cols());
        indicesExp << 45, 18, 23, 26, 19,
            35, 28, 35, 42, 9,
            49, 27, 46, 49, 29,
            7, 23, 45, 47, 3,
            27, 36, 36, 13, 37,
            46, 16, 28, 45, 11,
            23, 46, 18, 7, 42,
            18, 35, 15, 35, 38,
            26, 25, 49, 22, 8,
            36, 17, 27, 39, 47;

        KDTree::Matrix distsExp(knn, queryPts.cols());
        distsExp << 29.5291042871, 39.981545743, 28.7389126447, 52.1982317708,
            32.6827936382,
            32.3591934387, 51.4356218977, 30.8089678503, 52.4485242881,
            33.7956091823,
            34.8545850642, 55.3938733074, 31.2468142376, 54.1297016434,
            41.1569641738,
            36.3107890303, 57.7290386201, 41.5569837693, 57.8023018573,
            46.7189222907,
            52.2715075352, 80.4576018783, 42.0205009489, 58.6186915582,
            47.9001179539,
            57.164486353, 89.1198715215, 45.2172577674, 66.4977074793,
            50.221777149,
            62.6969839785, 89.3892174706, 49.7777018353, 79.6454901423,
            54.0013110952,
            65.8811665045, 91.1892301755, 51.7951735203, 81.5184157108,
            55.0528282652,
            70.1626602973, 95.1094264519, 56.6479884903, 87.3504287339,
            56.6024389934,
            72.9507409147, 95.5904283911, 58.393198234, 89.3475785906,
            57.1903505847;

        SECTION("default")
        {
            kdtree.setData(dataPts);
            kdtree.build();

            KDTree::MatrixI indicesAct;
            KDTree::Matrix distsAct;
            kdtree.query(queryPts, knn, indicesAct, distsAct);

            REQUIRE_MAT(indicesExp, indicesAct);
            REQUIRE_MAT_APPROX(distsExp, distsAct, 1e-3);
        }

        SECTION("low bucket size")
        {
            kdtree.setBucketSize(2);
            kdtree.setData(dataPts);
            kdtree.build();

            KDTree::MatrixI indicesAct;
            KDTree::Matrix distsAct;
            kdtree.query(queryPts, knn, indicesAct, distsAct);

            REQUIRE_MAT(indicesExp, indicesAct);
            REQUIRE_MAT_APPROX(distsExp, distsAct, 1e-3);
        }

        SECTION("balanced")
        {
            kdtree.setBucketSize(2);
            kdtree.setBalanced(true);
            kdtree.setData(dataPts);
            kdtree.build();

            KDTree::MatrixI indicesAct;
            KDTree::Matrix distsAct;
            kdtree.query(queryPts, knn, indicesAct, distsAct);

            REQUIRE_MAT(indicesExp, indicesAct);
            REQUIRE_MAT_APPROX(distsExp, distsAct, 1e-3);
        }

        SECTION("non-compact")
        {
            kdtree.setBucketSize(2);
            kdtree.setCompact(false);
            kdtree.setData(dataPts);
            kdtree.build();

            KDTree::MatrixI indicesAct;
            KDTree::Matrix distsAct;
            kdtree.query(queryPts, knn, indicesAct, distsAct);

            REQUIRE_MAT(indicesExp, indicesAct);
            REQUIRE_MAT_APPROX(distsExp, distsAct, 1e-3);
        }


    }

    SECTION("query maximum distance")
    {
        KDTree::Matrix data(3, 4);
        data << 1, 3, 0, 5,
            0, 1, 3, 4,
            3, 2, 0, 3;

        kdtree.setData(data);
        kdtree.setMaxDistance(2.1);
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
        REQUIRE(distances(0, 0) == Approx(2.0));
        REQUIRE(distances(1, 0) == -1.0);
    }
}
