/* kdtree_ann.h
 *
 *     Author: Fabian Meyer
 * Created On: 23 Jan 2019
 */

#ifndef KDT_KDTREE_ANN_H_
#define KDT_KDTREE_ANN_H_

#include <Eigen/Geometry>
#include <ANN/ANN.h>

namespace kdt
{
    class KDTreeAnn
    {
    public:
        typedef ANNcoord Scalar;
        typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
        typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;
        typedef typename Matrix::Index Index;
        typedef Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic> MatrixI;

    private:
        Matrix dataCopy_;
        Matrix *dataPoints_;

        ANNkd_tree *kdtree_;

        Scalar epsilon_;
        Scalar maxDist_;
        int threads_;

    public:
        KDTreeAnn()
            : dataCopy_(), dataPoints_(nullptr), kdtree_(nullptr), epsilon_(0),
            maxDist_(0), threads_(1)
        {
        }

        KDTreeAnn(Matrix &data, const bool copy = false)
            : KDTreeAnn()
        {
            setData(data, copy);
        }

        ~KDTreeAnn()
        {
            clear();
        }

        void setThreads(const int threads)
        {
            threads_ = threads;
        }

        void setEpsilon(const Scalar eps)
        {
            epsilon_ = eps;
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

            if(kdtree_ != nullptr)
                delete kdtree_;

            ANNpointArray dataPts = dataPoints_->data();
            kdtree_ = new ANNkd_tree(dataPts, dataPoints_->cols(), dataPoints_->rows());
        }

        void query(Matrix &queryPoints,
            const size_t knn,
            MatrixI &indices,
            Matrix &distances)
        {
            if(kdtree_ == nullptr)
                throw std::runtime_error("cannot query KDTree; not built yet");
            if(dimension() != queryPoints.rows())
                throw std::runtime_error("cannot query KDTree; KDTree has different dimension than query data");

            distances.setZero(knn, queryPoints.cols());
            indices.setConstant(knn, queryPoints.cols(), -1);

            Scalar maxDistSq = maxDist_ * maxDist_;

            #pragma omp parallel num_threads(threads_)
            for(Index i = 0; i < queryPoints.cols(); ++i)
            {
                ANNpoint p = &queryPoints.data()[i * queryPoints.rows()];
                ANNidxArray idx = &indices.data()[i * knn];
                ANNdistArray dists = &distances.data()[i * knn];

                if(maxDist_ > 0)
                    kdtree_->annkFRSearch(p, maxDistSq, knn, idx, dists, epsilon_);
                else
                    kdtree_->annkSearch(p, knn, idx, dists, epsilon_);
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
            if(kdtree_ != nullptr)
            {
                delete kdtree_;
                kdtree_ = nullptr;
            }
        }
    };
}

#endif
