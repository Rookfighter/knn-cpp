/* kdtree_flann.h
 *
 *     Author: Fabian Meyer
 * Created On: 09 Nov 2018
 */

#ifndef KDT_KDTREE_FLANN_H_
#define KDT_KDTREE_FLANN_H_

#include <Eigen/Geometry>
#include <flann/flann.hpp>

namespace kdt
{
    template<typename Scalar,
        typename Distance=flann::L2_Simple<Scalar>>
    class KDTreeFlann
    {
    public:
        typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
        typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;
        typedef typename Matrix::Index Index;
        typedef Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic> MatrixI;

    private:
        typedef flann::Index<Distance> FlannIndex;

        Matrix dataCopy_;
        Matrix *dataPoints_;

        FlannIndex *index_;
        flann::SearchParams searchParams_;
        flann::IndexParams indexParams_;
        Scalar maxDist_;

    public:
        KDTreeFlann()
            : dataCopy_(), dataPoints_(nullptr), index_(nullptr),
            searchParams_(32, 0, false),
            indexParams_(flann::KDTreeSingleIndexParams(15)),
            maxDist_(0)
        {
        }

        KDTreeFlann(Matrix &data, const bool copy = false)
            : KDTreeFlann()
        {
            setData(data, copy);
        }

        ~KDTreeFlann()
        {
            clear();
        }

        void setIndexParams(const flann::IndexParams &params)
        {
            indexParams_ = params;
        }

        void setChecks(const int checks)
        {
            searchParams_.checks = checks;
        }

        void setSorted(const bool sorted)
        {
            searchParams_.sorted = sorted;
        }

        void setThreads(const int threads)
        {
            searchParams_.cores = threads;
        }

        void setEpsilon(const float eps)
        {
            searchParams_.eps = eps;
        }

        void setMaxDistance(const Scalar dist)
        {
            maxDist_ = dist;
        }

        void setData(Matrix &data, const bool copy = false)
        {
            if(copy)
            {
                dataCopy_ = data;
                dataPoints_ = &dataCopy_;
            }
            else
            {
                dataPoints_ = &data;
            }

            clear();
        }

        void build()
        {
            if(dataPoints_ == nullptr)
                throw std::runtime_error("cannot build KDTree; data not set");
            if(dataPoints_->size() == 0)
                throw std::runtime_error("cannot build KDTree; data is empty");

            if(index_ != nullptr)
                delete index_;

            flann::Matrix<Scalar> dataPts(
                dataPoints_->data(),
                dataPoints_->cols(),
                dataPoints_->rows());

            index_ = new FlannIndex(dataPts, indexParams_);
            index_->buildIndex();
        }

        void query(Matrix &queryPoints,
            const size_t knn,
            MatrixI &indices,
            Matrix &distances) const
        {
            if(index_ == nullptr)
                throw std::runtime_error("cannot query KDTree; not built yet");
            if(dataPoints_->rows() != queryPoints.rows())
                throw std::runtime_error("cannot query KDTree; KDTree has different dimension than query data");

            distances.setConstant(knn, queryPoints.cols(), 1.0 / 0.0);
            indices.setConstant(knn, queryPoints.cols(), -1);

            flann::Matrix<Scalar> queryPts(
                queryPoints.data(),
                queryPoints.cols(),
                queryPoints.rows());
            flann::Matrix<int> indicesF(
                indices.data(),
                indices.cols(),
                indices.rows());
            flann::Matrix<Scalar> distancesF(
                distances.data(),
                distances.cols(),
                distances.rows());

            if(maxDist_ > 0)
            {
                flann::SearchParams params = searchParams_;
                params.max_neighbors = knn;
                index_->radiusSearch(queryPts, indicesF, distancesF, maxDist_, params);
            }
            else
            {
                index_->knnSearch(queryPts, indicesF, distancesF, knn, searchParams_);
            }
        }

        Index size() const
        {
            return dataPoints_ == nullptr ? 0 : dataPoints_->cols();
        }

        Index dimension() const
        {
            return dataPoints_ == nullptr ? 0 : dataPoints_->rows();
        }

        void clear()
        {
            if(index_ != nullptr)
            {
                delete index_;
                index_ = nullptr;
            }
        }

        FlannIndex &flannIndex()
        {
            return index_;
        }
    };

    typedef KDTreeFlann<double> KDTreeFlannd;
    typedef KDTreeFlann<float> KDTreeFlannf;
}

#endif
