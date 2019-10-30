/* distance_functors.h
 *
 *     Author: Fabian Meyer
 * Created On: 28 Oct 2019
 *    License: MIT
 */

#ifndef KNN_DISTANCE_FUNCTORS_H_
#define KNN_DISTANCE_FUNCTORS_H_

#include <type_traits>
#include "knn/matrix.h"

namespace knn
{
    /** Manhatten distance functor.
      * This the same as the L1 minkowski distance but more efficient.
      * @see EuclideanDistance, ChebyshevDistance, MinkowskiDistance */
    template <typename Scalar>
    struct ManhattenDistance
    {
        /** Compute the unrooted distance between two vectors.
          * @param lhs vector on left hand side
          * @param rhs vector on right hand side */
        template<typename DerivedA, typename DerivedB>
        Scalar operator()(const Eigen::MatrixBase<DerivedA> &lhs,
            const Eigen::MatrixBase<DerivedB> &rhs) const
        {
            static_assert(
                std::is_same<typename Eigen::MatrixBase<DerivedA>::Scalar,Scalar>::value,
                "distance scalar and input matrix A must have same type");
            static_assert(
                std::is_same<typename Eigen::MatrixBase<DerivedB>::Scalar, Scalar>::value,
                "distance scalar and input matrix B must have same type");

            return (lhs - rhs).cwiseAbs().sum();
        }

        /** Compute the unrooted distance between two scalars.
          * @param lhs scalar on left hand side
          * @param rhs scalar on right hand side */
        Scalar operator()(const Scalar lhs,
            const Scalar rhs) const
        {
            return std::abs(lhs - rhs);
        }

        /** Compute the root of a unrooted distance value.
          * @param value unrooted distance value */
        Scalar operator()(const Scalar val) const
        {
            return val;
        }
    };

    /** Euclidean distance functor.
      * This the same as the L2 minkowski distance but more efficient.
      * @see ManhattenDistance, ChebyshevDistance, MinkowskiDistance */
    template <typename Scalar>
    struct EuclideanDistance
    {
        /** Compute the unrooted distance between two vectors.
          * @param lhs vector on left hand side
          * @param rhs vector on right hand side */
        template<typename DerivedA, typename DerivedB>
        Scalar operator()(const Eigen::MatrixBase<DerivedA> &lhs,
            const Eigen::MatrixBase<DerivedB> &rhs) const
        {
            static_assert(
                std::is_same<typename Eigen::MatrixBase<DerivedA>::Scalar,Scalar>::value,
                "distance scalar and input matrix A must have same type");
            static_assert(
                std::is_same<typename Eigen::MatrixBase<DerivedB>::Scalar, Scalar>::value,
                "distance scalar and input matrix B must have same type");

            return (lhs - rhs).cwiseAbs2().sum();
        }

        /** Compute the unrooted distance between two scalars.
          * @param lhs scalar on left hand side
          * @param rhs scalar on right hand side */
        Scalar operator()(const Scalar lhs,
            const Scalar rhs) const
        {
            Scalar diff = lhs - rhs;
            return diff * diff;
        }

        /** Compute the root of a unrooted distance value.
          * @param value unrooted distance value */
        Scalar operator()(const Scalar val) const
        {
            return std::sqrt(val);
        }
    };

    /** General minkowski distance functor.
      * The infinite version is only available through the chebyshev distance.
      * @see ManhattenDistance, EuclideanDistance, ChebyshevDistance  */
    template <typename Scalar, int P>
    struct MinkowskiDistance
    {
        struct Pow
        {
            Scalar operator()(const Scalar val) const
            {
                Scalar result = 1;
                for(int i = 0; i < P; ++i)
                    result *= val;
                return result;
            }
        };

        /** Compute the unrooted distance between two vectors.
          * @param lhs vector on left hand side
          * @param rhs vector on right hand side */
        template<typename DerivedA, typename DerivedB>
        Scalar operator()(const Eigen::MatrixBase<DerivedA> &lhs,
            const Eigen::MatrixBase<DerivedB> &rhs) const
        {
            static_assert(
                std::is_same<typename Eigen::MatrixBase<DerivedA>::Scalar,Scalar>::value,
                "distance scalar and input matrix A must have same type");
            static_assert(
                std::is_same<typename Eigen::MatrixBase<DerivedB>::Scalar, Scalar>::value,
                "distance scalar and input matrix B must have same type");

            return (lhs - rhs).cwiseAbs().unaryExpr(MinkowskiDistance::Pow()).sum();
        }

        /** Compute the unrooted distance between two scalars.
          * @param lhs scalar on left hand side
          * @param rhs scalar on right hand side */
        Scalar operator()(const Scalar lhs,
            const Scalar rhs) const
        {
            return std::pow(std::abs(lhs - rhs), P);;
        }

        /** Compute the root of a unrooted distance value.
          * @param value unrooted distance value */
        Scalar operator()(const Scalar val) const
        {
            return std::pow(val, 1 / static_cast<Scalar>(P));
        }
    };

    /** Chebyshev distance functor.
      * This distance is the same as infinity minkowski distance.
      * @see ManhattenDistance, EuclideanDistance, MinkowskiDistance */
    template<typename Scalar>
    struct ChebyshevDistance
    {
        /** Compute the unrooted distance between two vectors.
          * @param lhs vector on left hand side
          * @param rhs vector on right hand side */
        template<typename DerivedA, typename DerivedB>
        Scalar operator()(const Eigen::MatrixBase<DerivedA> &lhs,
            const Eigen::MatrixBase<DerivedB> &rhs) const
        {
            static_assert(
                std::is_same<typename Eigen::MatrixBase<DerivedA>::Scalar,Scalar>::value,
                "distance scalar and input matrix A must have same type");
            static_assert(
                std::is_same<typename Eigen::MatrixBase<DerivedB>::Scalar, Scalar>::value,
                "distance scalar and input matrix B must have same type");

            return (lhs - rhs).cwiseAbs().maxCoeff();
        }

        /** Compute the unrooted distance between two scalars.
          * @param lhs scalar on left hand side
          * @param rhs scalar on right hand side */
        Scalar operator()(const Scalar lhs,
            const Scalar rhs) const
        {
            return std::abs(lhs - rhs);
        }

        /** Compute the root of a unrooted distance value.
          * @param value unrooted distance value */
        Scalar operator()(const Scalar val) const
        {
            return val;
        }
    };

    /** Hamming distance functor.
      * The distance vectors have to be of integral type and should hold the
      * information vectors as bitmasks.
      * Performs a XOR operation on the vectors and counts the number of set
      * ones. */
    template<typename Scalar>
    struct HammingDistance
    {
        static_assert(std::is_integral<Scalar>::value,
            "HammingDistance requires integral Scalar type");

        struct XOR
        {
            Scalar operator()(const Scalar lhs, const Scalar rhs) const
            {
                return lhs ^ rhs;
            }
        };

        struct BitCount
        {
            Scalar operator()(const Scalar lhs) const
            {
                Scalar copy = lhs;
                Scalar result = 0;
                while(copy)
                {
                    ++result;
                    copy &= (copy - 1);
                }

                return result;
            }
        };

        /** Compute the unrooted distance between two vectors.
          * @param lhs vector on left hand side
          * @param rhs vector on right hand side */
        template<typename DerivedA, typename DerivedB>
        Scalar operator()(const Eigen::MatrixBase<DerivedA> &lhs,
            const Eigen::MatrixBase<DerivedB> &rhs) const
        {
            static_assert(
                std::is_same<typename Eigen::MatrixBase<DerivedA>::Scalar,Scalar>::value,
                "distance scalar and input matrix A must have same type");
            static_assert(
                std::is_same<typename Eigen::MatrixBase<DerivedB>::Scalar, Scalar>::value,
                "distance scalar and input matrix B must have same type");

            return lhs.
                binaryExp(rhs, XOR()).
                unaryExp(BitCount()).
                sum();
        }

        /** Compute the unrooted distance between two scalars.
          * @param lhs scalar on left hand side
          * @param rhs scalar on right hand side */
        Scalar operator()(const Scalar lhs,
            const Scalar rhs) const
        {
            BitCount cnt;
            XOR xOr;
            return cnt(xOr(lhs, rhs));
        }

        /** Compute the root of a unrooted distance value.
          * @param value unrooted distance value */
        Scalar operator()(const Scalar value) const
        {
            return value;
        }
    };
}

#endif
