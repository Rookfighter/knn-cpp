/* brute_force.cpp
 *
 *     Author: Fabian Meyer
 * Created On: 08 Nov 2018
 */

#include <knncpp.h>
#include "assert/eigen_require.h"

typedef double Scalar;
typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
typedef knncpp::Matrixi Matrixi;

TEST_CASE("brute_force")
{
    SECTION("query one")
    {
        knncpp::BruteForce<Scalar, knncpp::EuclideanDistance<Scalar>> bruteforce;
        Matrix data(3, 4);
        data << 1, 3, 0, 5,
            0, 1, 2, 4,
            3, 2, 0, 3;

        bruteforce.setData(data);
        bruteforce.build();

        REQUIRE(bruteforce.size() == 4);

        Matrix points(3, 1);
        points << 0, 1, 0;
        Matrixi indices;
        Matrix distances;

        bruteforce.query(points, 1, indices, distances);

        REQUIRE(indices.size() == 1);
        REQUIRE(indices(0) == 2);
        REQUIRE(distances.size() == 1);
        REQUIRE(distances(0) == Approx(1.0));
    }

    SECTION("euclidean query multiple")
    {
        knncpp::BruteForce<Scalar, knncpp::EuclideanDistance<Scalar>> bruteforce;

        Matrix data(3, 9);
        data << 1, 2, 3, 1, 2, 3, 1, 2, 1,
                2, 1, 0, 3, 2, 1, 0, 3, 1,
                3, 1, 3, 1, 3, 4, 4, 2, 1;

        bruteforce.setData(data);
        bruteforce.build();

        REQUIRE(bruteforce.size() == 9);

        Matrix points(3, 1);
        points << 0, 1, 0;

        Matrixi indicesExp(3, 1);
        indicesExp << 8, 1, 3;
        Matrix distancesExp(3, 1);
        distancesExp << std::sqrt(2), std::sqrt(5), std::sqrt(6);

        Matrixi indices;
        Matrix distances;
        bruteforce.query(points, 3, indices, distances);

        REQUIRE(indices.size() == 3);
        REQUIRE(distances.size() == 3);
        REQUIRE_MATRIX(indicesExp, indices);
        REQUIRE_MATRIX_APPROX(distancesExp, distances, 1e-3);
    }

    SECTION("manhatten query multiple")
    {
        knncpp::BruteForce<Scalar, knncpp::ManhattenDistance<Scalar>> bruteforce;

        Matrix data(3, 9);
        data << 1, 2, 3, 1, 2, 3, 1, 2, 3,
                2, 1, 0, 3, 2, 1, 0, 3, 4,
                3, 1, 3, 1, 3, 4, 4, 2, 1;

        bruteforce.setData(data);
        bruteforce.build();

        REQUIRE(bruteforce.size() == 9);

        Matrix points(3, 1);
        points << 0, 1, 0;

        Matrixi indicesExp(3, 1);
        indicesExp << 1, 3, 0;
        Matrix distancesExp(3, 1);
        distancesExp << 3, 4, 5;

        Matrixi indices;
        Matrix distances;
        bruteforce.query(points, 3, indices, distances);

        REQUIRE(indices.size() == 3);
        REQUIRE(distances.size() == 3);
        REQUIRE_MATRIX(indicesExp, indices);
        REQUIRE_MATRIX_APPROX(distancesExp, distances, 1e-3);
    }

    SECTION("minkowski query multiple")
    {
        knncpp::BruteForce<Scalar, knncpp::MinkowskiDistance<Scalar, 2>> bruteforce;

        Matrix data(3, 9);
        data << 1, 2, 3, 1, 2, 3, 1, 2, 1,
                2, 1, 0, 3, 2, 1, 0, 3, 1,
                3, 1, 3, 1, 3, 4, 4, 2, 1;

        bruteforce.setData(data);
        bruteforce.build();

        REQUIRE(bruteforce.size() == 9);

        Matrix points(3, 1);
        points << 0, 1, 0;

        Matrixi indicesExp(3, 1);
        indicesExp << 8, 1, 3;
        Matrix distancesExp(3, 1);
        distancesExp << std::sqrt(2), std::sqrt(5), std::sqrt(6);

        Matrixi indices;
        Matrix distances;
        bruteforce.query(points, 3, indices, distances);

        REQUIRE(indices.size() == 3);
        REQUIRE(distances.size() == 3);
        REQUIRE_MATRIX(indicesExp, indices);
        REQUIRE_MATRIX_APPROX(distancesExp, distances, 1e-3);
    }

    SECTION("chebyshev query multiple")
    {
        knncpp::BruteForce<Scalar, knncpp::ChebyshevDistance<Scalar>> bruteforce;

        Matrix data(3, 9);
        data << 1, 2, 4, 4, 4, 1, 1, 5, 3,
                2, 1, 0, 3, 2, 1, 0, 3, 4,
                3, 1, 3, 1, 3, 0, 4, 2, 6;

        bruteforce.setData(data);
        bruteforce.build();

        REQUIRE(bruteforce.size() == 9);

        Matrix points(3, 1);
        points << 0, 1, 0;

        Matrixi indicesExp(3, 1);
        indicesExp << 5, 1, 0;
        Matrix distancesExp(3, 1);
        distancesExp << 1, 2, 3;

        Matrixi indices;
        Matrix distances;
        bruteforce.query(points, 3, indices, distances);

        REQUIRE(indices.size() == 3);
        REQUIRE(distances.size() == 3);
        REQUIRE_MATRIX(indicesExp, indices);
        REQUIRE_MATRIX_APPROX(distancesExp, distances, 1e-3);
    }

    SECTION("query many")
    {
        knncpp::BruteForce<Scalar, knncpp::EuclideanDistance<Scalar>> bruteforce;
        Matrix dataPts(3, 50);
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
        Matrix queryPts(3, 5);
        queryPts << 63.1, 30.16, 51.24, 78.61, -17.01,
            42.52, 81.1, 24.44, -5.45, -26.54,
            3.92, -80.29, -45.93, 53.99, 41.71;

        size_t knncpp = 10;
        Matrixi indicesExp(knncpp, queryPts.cols());
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

        Matrix distsExp(knncpp, queryPts.cols());
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

        bruteforce.setData(dataPts);
        bruteforce.build();

        Matrixi indicesAct;
        Matrix distsAct;
        bruteforce.query(queryPts, knncpp, indicesAct, distsAct);

        REQUIRE_MATRIX(indicesExp, indicesAct);
        REQUIRE_MATRIX_APPROX(distsExp, distsAct, 1e-3);
    }

    SECTION("query maximum distance")
    {
        knncpp::BruteForce<Scalar, knncpp::EuclideanDistance<Scalar>> bruteforce;
        Matrix dataPts(3, 50);
        dataPts << 99.88, -19.59, -74.16, 86.5, 47.21, -72.68, -1.97, -54.12,
            -9.22, 79.25, 94.14, 44.77, -34.63, 52.89, -91.08, -34.02, 1.02,
            14.6, -41.38, 77.02, -33.63, 1.18, 33.28, 37.06, -68.19, -39.77,
            73.96, -29.34, 98.76, 69.12, -69.26, -97.06, 91.02, -54.27, -19.41,
            -63.78, 63.56, -56.25, 42.63, 58.83, 42.2, 48.07, -94.81, 74.61,
            -36.59, 16.65, 16.57, 98.9, 34.74, -83.87,
            -17.92, -85.04, -54.71, -53.14, -43.44, 57.76, -3.56, 12.08,
            -14.66, 91.27, 30.09, 72.55, 78.14, -98.44, -75.53, -97.77, 99.08,
            -13.76, -11.25, 0.39, 41.69, -95.19, 8.26, 93.32, -51.58, -54.19,
            85.85, -51.24, -63.33, 49.69, 89.85, 63.12, -89.89, -23.73, -4.1,
            11.13, 22.82, 45.44, 67.2, -93.46, -46.16, 47.57, -7.02, 54.61,
            -16.6, -69.1, 96.98, 38.38, -50.43, 68.72,
            -81.2, 7.82, 98.5, -99.23, 36.29, -56.29, 87.56, -6.11, -75.95,
            35.35, -91.22, 19.16, -4.54, -26.23, -78.36, -84.74, -28.37, 80.16,
            47.3, -53.94, 95.21, 47.9, -97.25, 80.98, 37.47, -68.59, -56.08,
            77.62, 39.13, -92.14, -96.47, -90.8, 2.15, -68.76, 61.21, 35.87,
            -86.98, 55.36, 48.69, -86.21, 19.11, -75.52, -40.56, 62.78, -56.22,
            -90.37, 21.51, 63.48, 70.21, -47.03;
        Matrix queryPts(3, 5);
        queryPts << 57.38, -75.03, -66.35, -51.12, -55.15,
            87.31, -50.46, -71.72, 48.48, -13.82,
            10.3, -96.74, 72.87, -19.65, 70.21;

        size_t knncpp = 10;
        Matrixi indicesExp(knncpp, queryPts.cols());
        indicesExp << 11, 14, 2, 12, 18,
            9, 33, 24, 7, 34,
            46, 25, 27, 5, 35,
            38, 15, 18, 49, 27,
            43, 44, -1, 35, 24,
            26, 42, -1, 16, 2,
            16, -1, -1, 42, 6,
            23, -1, -1, -1, 37,
            -1, -1, -1, -1, 20,
            -1, -1, -1, -1, 17;

        Matrix distsExp(knncpp, queryPts.cols());
        distsExp << 21.3393837774, 34.9847366719, 31.7369358949, 37.147648647,
        26.8530426581,
            33.4885204212, 43.9129923827, 40.769660288, 38.9524273955,
            38.1159546647,
            43.412315073, 45.2725413468, 42.5644570035, 43.5136943961,
            43.3152975287,
            45.7795445587, 63.7499505882, 70.242057914, 47.242824852,
            46.0577962999,
            64.1896666139, 65.3146507301, -1, 68.1011637199, 51.6504288462,
            68.434877073, 73.7188876747, -1, 73.1777151871, 53.2324741112,
            69.3566968648, -1, -1, 73.6633843914, 56.8718076027,
            73.7881081205, -1, -1, -1, 61.102210271,
            -1, -1, -1, -1, 64.5714371839,
            -1, -1, -1, -1, 70.4561466446;

        bruteforce.setMaxDistance(75.0 * 75.0);
        bruteforce.setData(dataPts);
        bruteforce.build();

        Matrixi indicesAct;
        Matrix distsAct;
        bruteforce.query(queryPts, knncpp, indicesAct, distsAct);

        REQUIRE_MATRIX(indicesExp, indicesAct);
        REQUIRE_MATRIX_APPROX(distsExp, distsAct, 1e-3);
    }
}
