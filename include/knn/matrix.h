/* matrix.h
 *
 *     Author: Fabian Meyer
 * Created On: 28 Oct 2019
 *    License: MIT
 */

#ifndef KNN_MATRIX_H_
#define KNN_MATRIX_H_

#include <Eigen/Geometry>

namespace knn
{
    typedef long int Index;

    typedef Eigen::Matrix<Index, Eigen::Dynamic, 1> Vectori;
    typedef Eigen::Matrix<Index, 2, 1> Vector2i;
    typedef Eigen::Matrix<Index, 3, 1> Vector3i;
    typedef Eigen::Matrix<Index, 4, 1> Vector4i;
    typedef Eigen::Matrix<Index, 5, 1> Vector5i;

    typedef Eigen::Matrix<Index, Eigen::Dynamic, Eigen::Dynamic> Matrixi;
    typedef Eigen::Matrix<Index, 2, 2> Matrix2i;
    typedef Eigen::Matrix<Index, 3, 3> Matrix3i;
    typedef Eigen::Matrix<Index, 4, 4> Matrix4i;
    typedef Eigen::Matrix<Index, 5, 5> Matrix5i;

    typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> Matrixf;
    typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> Matrixd;
}

#endif
