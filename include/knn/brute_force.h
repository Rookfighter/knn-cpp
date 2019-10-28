/* brute_force.h
 *
 *     Author: Fabian Meyer
 * Created On: 28 Oct 2019
 *    License: MIT
 */

#ifndef KNN_BRUTE_FORCE_H_
#define KNN_BRUTE_FORCE_H_

#include "knn/distance_functors.h"
#include "knn/query_heap.h"

namespace knn
{
    template<typename Scalar,
        typename Distance=EuclideanDistance<Scalar>>
    class BruteForce
    {
    public:
        typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
        typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;
        typedef knn::Matrixi Matrixi;
    private:
        Distance distance_;
        Matrix dataCopy_;
        const Matrix *data_;

        bool sorted_;
        bool takeRoot_;
        Index threads_;
        Scalar maxDist_;
        Scalar maxDistP_;

    public:

        BruteForce()
            : distance_(), dataCopy_(), data_(nullptr), sorted_(true),
            takeRoot_(true), threads_(1), maxDist_(0), maxDistP_(0)
        { }

        /** Constructs a brute force instance with the given data.
          * @param data NxM matrix, M points of dimension N
          * @param copy if true copies the data, otherwise assumes static data */
        BruteForce(const Matrix &data, const bool copy = false)
            : BruteForce()
        {
            setData(data, copy);
        }

        /** Set if the points returned by the queries should be sorted
          * according to their distance to the query points.
          * @param sorted sort query results */
        void setSorted(const bool sorted)
        {
            sorted_ = sorted;
        }

        /** Set if the distances after the query should be rooted or not.
          * Taking the root of the distances increases query time, but the
          * function will return true distances instead of their powered
          * versions.
          * @param takeRoot set true if root should be taken else false */
        void setTakeRoot(const bool takeRoot)
        {
            takeRoot_ = takeRoot;
        }

        /** Set the amount of threads that should be used for querying.
          * OpenMP has to be enabled for this to work.
          * @param threads amount of threads, 0 for optimal choice */
        void setThreads(const unsigned int threads)
        {
            threads_ = threads;
        }

        /** Set the maximum distance for querying the tree.
          * The search will be pruned if the maximum distance is set to any
          * positive number.
          * @param maxDist maximum distance, <= 0 for no limit */
        void setMaxDistance(const Scalar maxDist)
        {
            maxDist_ = maxDist;
            maxDistP_ = distance_.power(maxDist);
        }

        /** Set the data points used for this tree.
          * This does not build the tree.
          * @param data NxM matrix, M points of dimension N
          * @param copy if true data is copied, assumes static data otherwise */
        void setData(const Matrix &data, const bool copy = false)
        {
            if(copy)
            {
                dataCopy_ = data;
                data_ = &dataCopy_;
            }
            else
            {
                data_ = &data;
            }
        }

        void setDistance(const Distance &distance)
        {
            distance_ = distance;
        }

        void build()
        { }

        template<typename Derived>
        void query(const Eigen::MatrixBase<Derived> &queryPoints,
            const size_t knn,
            Matrixi &indices,
            Matrix &distances) const
        {
            if(data_ == nullptr)
                throw std::runtime_error("cannot query BruteForce: data not set");
            if(data_->size() == 0)
                throw std::runtime_error("cannot query BruteForce: data is empty");
            if(queryPoints.rows() != dimension())
                throw std::runtime_error("cannot query BruteForce: data and query descriptors do not have same dimension");

            const Matrix &dataPoints = *data_;

            indices.setConstant(knn, queryPoints.cols(), -1);
            distances.setConstant(knn, queryPoints.cols(), -1);

            #pragma omp parallel for num_threads(threads_)
            for(Index i = 0; i < queryPoints.cols(); ++i)
            {
                Index *idxPoint = &indices.data()[i * knn];
                Scalar *distPoint = &distances.data()[i * knn];

                QueryHeap<Scalar> heap(idxPoint, distPoint, knn);

                for(Index j = 0; j < dataPoints.cols(); ++j)
                {
                    Scalar dist = distance_.unrooted(queryPoints.col(i), dataPoints.col(j));

                    if((maxDistP_ <= 0 || dist <= maxDistP_) &&
                        (!heap.full() || dist < heap.front()))
                    {
                        if(heap.full())
                            heap.pop();
                        heap.push(j, dist);
                    }
                }

                if(sorted_)
                    heap.sort();

                if(takeRoot_)
                {
                    for(size_t j = 0; j < knn; ++j)
                    {
                        if(distPoint[j] < 0)
                            break;
                        distPoint[j] = distance_.root(distPoint[j]);
                    }
                }
            }
        }

        /** Returns the amount of data points stored in the search index.
          * @return number of data points */
        Index size() const
        {
            return data_ == nullptr ? 0 : data_->cols();
        }

        /** Returns the dimension of the data points in the search index.
          * @return dimension of data points */
        Index dimension() const
        {
            return data_ == nullptr ? 0 : data_->rows();
        }
    };
}

#endif
