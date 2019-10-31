/* multi_index_hashing.h
 *
 *     Author: Fabian Meyer
 * Created On: 29 Oct 2019
 *    License: MIT
 *
 * Implementation is based on the work "Fast Search in Hamming Space with
 * Multi-Index Hashing" by Mohammad Norouzi, Ali Punjani and David J. Fleet.
 */

#ifndef KNN_MULTI_INDEX_HASHING_H_
#define KNN_MULTI_INDEX_HASHING_H_

#include <map>
#include <set>
#include "knn/distance_functors.h"
#include "knn/query_heap.h"

namespace knn
{
    /** Class for performing KNN search in hamming space by multi-index hashing. */
    template<typename Scalar>
    class MultiIndexHashing
    {
    public:
        static_assert(std::is_integral<Scalar>::value, "MultiIndexHashing Scalar has to be integral");

        typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
        typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;
        typedef knn::Matrixi Matrixi;

    private:
        HammingDistance<Scalar> distance_;

        Matrix dataCopy_;
        const Matrix *data_;

        bool sorted_;
        Scalar maxDist_;
        Index substrLen_;
        Index threads_;
        std::vector<std::map<Scalar, std::vector<Index>>> buckets_;

        template<typename Derived>
        Scalar extractCode(const Eigen::MatrixBase<Derived> &data,
            const Index idx,
            const Index offset) const
        {
            Index leftShift = std::max<Index>(0, static_cast<Index>(sizeof(Scalar)) - offset - substrLen_);
            Index rightShift = leftShift + offset;

            Scalar code = (data(idx, 0) << (leftShift * 8)) >> (rightShift * 8);

            if(static_cast<Index>(sizeof(Scalar)) - offset < substrLen_ && idx + 1 < data.rows())
            {
                Index shift = 2 * static_cast<Index>(sizeof(Scalar)) - substrLen_ - offset;
                code |= data(idx+1, 0) << (shift * 8);
            }

            return code;
        }
    public:
        MultiIndexHashing()
            : distance_(), dataCopy_(), data_(nullptr), sorted_(true),
            maxDist_(0), substrLen_(1), threads_(1)
        { }

        /** Constructs an index with the given data.
          * This does not build the the index.
          * @param data NxM matrix, M points of dimension N
          * @param copy if true copies the data, otherwise assumes static data */
        MultiIndexHashing(const Matrix &data, const bool copy=false)
            : MultiIndexHashing()
        {
            setData(data, copy);
        }

        /** Set the maximum distance for querying the index.
          * Note that if no maximum distance is used, this algorithm performs
          * basically a brute force search.
          * @param maxDist maximum distance, <= 0 for no limit */
        void setMaxDistance(const Scalar maxDist)
        {
            maxDist_ = maxDist;
        }

        /** Set if the points returned by the queries should be sorted
          * according to their distance to the query points.
          * @param sorted sort query results */
        void setSorted(const bool sorted)
        {
            sorted_ = sorted;
        }

        /** Set the amount of threads that should be used for building and
          * querying the tree.
          * OpenMP has to be enabled for this to work.
          * @param threads amount of threads, 0 for optimal choice */
        void setThreads(const unsigned int threads)
        {
            threads_ = threads;
        }

        /** Set the length of substrings (in bytes) used for multi index hashing.
          * @param len lentth of bucket substrings in bytes*/
        void setSubstringLength(const Index len)
        {
            substrLen_ = len;
        }

        /** Set the data points used for the KNN search.
          * @param data NxM matrix, M points of dimension N
          * @param copy if true data is copied, assumes static data otherwise */
        void setData(const Matrix &data, const bool copy = false)
        {
            clear();
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

        void build()
        {
            if(data_ == nullptr)
                throw std::runtime_error("cannot build MultiIndexHashing; data not set");
            if(data_->size() == 0)
                throw std::runtime_error("cannot build MultiIndexHashing; data is empty");

            const Matrix &data = *data_;
            const Index bytesPerVec = data.rows() * static_cast<Index>(sizeof(Scalar));
            if(bytesPerVec % substrLen_ != 0)
                throw std::runtime_error("cannot build MultiIndexHashing; cannot divide byte count per vector by substring length without remainings");

            buckets_.clear();
            buckets_.resize(bytesPerVec / substrLen_);

            for(size_t i = 0; i < buckets_.size(); ++i)
            {
                Index start = static_cast<Index>(i) * substrLen_;
                Index idx = start / static_cast<Index>(sizeof(Scalar));
                Index offset = start % static_cast<Index>(sizeof(Scalar));
                std::map<Scalar, std::vector<Index>> &map = buckets_[i];

                for(Index c = 0; c < data.cols(); ++c)
                {
                    Scalar code = extractCode(data.col(c), idx, offset);
                    if(map.find(code) == map.end())
                        map[code] = std::vector<Index>();
                    map[code].push_back(c);
                }
            }
        }

        template<typename Derived>
        void query(const Eigen::MatrixBase<Derived> &queryPoints,
            const size_t knn,
            Matrixi &indices,
            Matrix &distances) const
        {
            if(buckets_.size() == 0)
                throw std::runtime_error("cannot query MultiIndexHashing; not built yet");
            if(queryPoints.rows() != dimension())
                throw std::runtime_error("cannot query MultiIndexHashing; data and query points do not have same dimension");

            const Matrix &data = *data_;

            indices.setConstant(knn, queryPoints.cols(), -1);
            distances.setConstant(knn, queryPoints.cols(), -1);

            Index *indicesRaw = indices.data();
            Scalar *distsRaw = distances.data();

            Scalar maxDistPart = maxDist_ / buckets_.size();

            #pragma omp parallel for num_threads(threads_)
            for(Index c = 0; c < queryPoints.cols(); ++c)
            {
                std::set<Index> candidates;
                for(size_t i = 0; i < buckets_.size(); ++i)
                {
                    Index start = static_cast<Index>(i) * substrLen_;
                    Index idx = start / static_cast<Index>(sizeof(Scalar));
                    Index offset = start % static_cast<Index>(sizeof(Scalar));
                    const std::map<Scalar, std::vector<Index>> &map = buckets_[i];

                    Scalar code = extractCode(queryPoints.col(c), idx, offset);
                    for(const auto &x: map)
                    {
                        Scalar dist = distance_(x.first, code);
                        if(maxDistPart <= 0 || dist <= maxDistPart)
                        {
                            for(size_t j = 0; j < x.second.size(); ++j)
                                candidates.insert(x.second[j]);
                        }
                    }
                }

                Scalar *distPoint = &distsRaw[c * knn];
                Index *idxPoint = &indicesRaw[c * knn];
                // create heap to find nearest neighbours
                QueryHeap<Scalar> dataHeap(idxPoint, distPoint, knn);

                for(Index idx: candidates)
                {
                    Scalar dist = distance_(data.col(idx), queryPoints.col(c));

                    bool isInRange = maxDist_ <= 0 || dist <= maxDist_;
                    bool isImprovement = !dataHeap.full() ||
                        dist < dataHeap.front();
                    if(isInRange && isImprovement)
                    {
                        if(dataHeap.full())
                            dataHeap.pop();
                        dataHeap.push(idx, dist);
                    }
                }

                if(sorted_)
                    dataHeap.sort();
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

        void clear()
        {
            data_ = nullptr;
            dataCopy_.resize(0, 0);
            buckets_.clear();
        }

    };
}

#endif
