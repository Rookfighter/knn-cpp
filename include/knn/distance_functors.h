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
        template<typename DerivedA, typename DerivedB>
        Scalar unrooted(const Eigen::MatrixBase<DerivedA> &lhs,
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

        template<typename DerivedA, typename DerivedB>
        Scalar rooted(const Eigen::MatrixBase<DerivedA> &lhs,
            const Eigen::MatrixBase<DerivedB> &rhs) const
        {
            return unrooted(lhs, rhs);
        }

        Scalar root(const Scalar val) const
        {
            return val;
        }

        Scalar power(const Scalar val) const
        {
            return std::abs(val);
        }
    };

    /** Euclidean distance functor.
      * This the same as the L2 minkowski distance but more efficient.
      * @see ManhattenDistance, ChebyshevDistance, MinkowskiDistance */
    template <typename Scalar>
    struct EuclideanDistance
    {
        template<typename DerivedA, typename DerivedB>
        Scalar unrooted(const Eigen::MatrixBase<DerivedA> &lhs,
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

        template<typename DerivedA, typename DerivedB>
        Scalar rooted(const Eigen::MatrixBase<DerivedA> &lhs,
            const Eigen::MatrixBase<DerivedB> &rhs) const
        {
            return std::sqrt(unrooted(lhs, rhs));
        }

        Scalar root(const Scalar val) const
        {
            return std::sqrt(val);
        }

        Scalar power(const Scalar val) const
        {
            return val * val;
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

        template<typename DerivedA, typename DerivedB>
        Scalar unrooted(const Eigen::MatrixBase<DerivedA> &lhs,
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

        template<typename DerivedA, typename DerivedB>
        Scalar rooted(const Eigen::MatrixBase<DerivedA> &lhs,
            const Eigen::MatrixBase<DerivedB> &rhs) const
        {
            return root(unrooted(lhs, rhs));
        }

        Scalar root(const Scalar val) const
        {
            return std::pow(val, 1 / static_cast<Scalar>(P));
        }

        Scalar power(const Scalar val) const
        {
            return std::pow(std::abs(val), P);
        }
    };

    /** Chebyshev distance functor.
      * This distance is the same as infinity minkowski distance.
      * @see ManhattenDistance, EuclideanDistance, MinkowskiDistance */
    template<typename Scalar>
    struct ChebyshevDistance
    {
        template<typename DerivedA, typename DerivedB>
        Scalar unrooted(const Eigen::MatrixBase<DerivedA> &lhs,
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

        template<typename DerivedA, typename DerivedB>
        Scalar rooted(const Eigen::MatrixBase<DerivedA> &lhs,
            const Eigen::MatrixBase<DerivedB> &rhs) const
        {
            return unrooted(lhs, rhs);
        }

        Scalar root(const Scalar val) const
        {
            return val;
        }

        Scalar power(const Scalar val) const
        {
            return std::abs(val);
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
        template<typename ScalarInt>
        struct XOR
        {
            ScalarInt operator()(const ScalarInt lhs, const ScalarInt rhs) const
            {
                return lhs ^ rhs;
            }
        };

        template<typename ScalarInt>
        struct BitCount
        {
            ScalarInt operator()(const ScalarInt lhs) const
            {
                ScalarInt copy = lhs;
                ScalarInt result = 0;
                while(copy)
                {
                    result++;
                    copy &= (copy - 1);
                }

                return result;
            }
        };

        template<typename DerivedA, typename DerivedB>
        Scalar unrooted(const Eigen::MatrixBase<DerivedA> &lhs,
            const Eigen::MatrixBase<DerivedB> &rhs) const
        {
            typedef typename Eigen::MatrixBase<DerivedA>::Scalar ScalarIntA;
            typedef typename Eigen::MatrixBase<DerivedB>::Scalar ScalarIntB;
            static_assert(
                std::is_same<ScalarIntA, ScalarIntB>::value,
                "HammingDistance requires matrices of same scalar type");
            static_assert(
                std::is_integral<ScalarIntA>::value,
                "HammingDistance requires integral matrix types");

            return static_cast<Scalar>(lhs.
                binaryExp(rhs, HammingDistance::XOR<ScalarIntA>()).
                unaryExp(HammingDistance::BitCount<ScalarIntA>()).
                sum());
        }

        template<typename DerivedA, typename DerivedB>
        Scalar rooted(const Eigen::MatrixBase<DerivedA> &lhs,
            const Eigen::MatrixBase<DerivedB> &rhs) const
        {
            return unrooted(lhs, rhs);
        }

        Scalar root(const Scalar val) const
        {
            return val;
        }

        Scalar power(const Scalar val) const
        {
            return val;
        }
    };
}

#endif
