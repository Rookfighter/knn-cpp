
#ifndef KDT_KDTREE_EIGEN_H_
#define KDT_KDTREE_EIGEN_H_

#include <Eigen/Geometry>

namespace kdt
{
    template <typename Scalar, int P, int Dim=Eigen::Dynamic>
    struct MinkowskiDistance
    {
        typedef Eigen::Matrix<Scalar, Dim, 1> Vector;

        Scalar operator()(const Vector &vecA, const Vector &vecB) const
        {
            return (vecA - vecB).template lpNorm<P>();
        }
    };

    template<typename Scalar,
        int Dim=Eigen::Dynamic,
        typename Distance=MinkowskiDistance<Scalar, 2, Dim>>
    class KDTree
    {
    public:
        typedef Eigen::Matrix<Scalar, Dim, Eigen::Dynamic> Matrix;
        typedef Eigen::Matrix<Scalar, Dim, 1> Vector;
        typedef Eigen::Matrix<Eigen::Index, Eigen::Dynamic, 1> IndexVector;

        /** Struct representing a node in the KDTree.
          * It can be either a inner node or a leaf node. */
        struct Node
        {
            /** Indices of data points in this leaf node. */
            IndexVector idx;

            /** Left child of this inner node. */
            Node *left;
            /** Right child of this inner node. */
            Node *right;
            /** Axis of the axis aligned splitting hyper plane. */
            Eigen::Index axis;
            /** Translation of the axis aligned splitting hyper plane. */
            Scalar splitpoint;

            Node()
                : idx(), left(nullptr), right(nullptr), axis(-1)
            {

            }

            /** Constructor for leaf nodes */
            Node(const IndexVector &idx)
                :idx(idx), left(nullptr), right(nullptr), axis(-1)
            {

            }

            /** Constructor for inner nodes */
            Node(const Eigen::Index axis, const Scalar splitpoint, Node *left,
                Node *right)
                : idx(), left(left), right(right), axis(axis),
                splitpoint(splitpoint)
            {

            }

            bool isLeaf() const
            {
                return !hasLeft() && !hasRight();
            }

            bool hasLeft() const
            {
                return left != nullptr;
            }

            bool hasRight() const
            {
                return right != nullptr;
            }
        };
    private:
        Matrix dataCopy_;
        const Matrix *data_;

        Eigen::Index bucketSize_;
        bool balance_;
        int threads_;

        Distance distance_;

        Node *root_;

        void findBoundaries(const Matrix &data, const IndexVector &idx, Vector &mins,
            Vector &maxes) const
        {
            assert(idx.size() > 0);
            Eigen::Index dim = data.rows();
            mins = data.col(idx(0));
            maxes = mins;
            for(Eigen::Index i = 1; i < idx.size(); ++i)
            {
                Eigen::Index c = idx(i);
                for(Eigen::Index r = 0; r < dim; ++r)
                {
                    Scalar a = data(r, c);
                    mins(r) = a < mins(r) ? a : mins(r);
                    maxes(r) = a > maxes(r) ? a : maxes(r);
                }
            }
        }

        bool isRight(const Scalar point, const Scalar midpoint, bool inclusive)
            const
        {
            return (inclusive && point >= midpoint) || point > midpoint;
        }

        void splitMidpoint(const Matrix &data, const IndexVector &idx,
            const Scalar midpoint, const Eigen::Index axis, IndexVector &leftIdx,
            IndexVector &rightIdx, bool rightInclusive) const
        {
            Eigen::Index leftCnt = 0;
            Eigen::Index rightCnt = 0;
            for(Eigen::Index i = 0; i < idx.size(); ++i)
            {
                if(isRight(data(axis, idx(i)), midpoint, rightInclusive))
                    ++rightCnt;
                else
                    ++leftCnt;
            }

            leftIdx.resize(leftCnt);
            rightIdx.resize(rightCnt);
            leftCnt = 0;
            rightCnt = 0;

            for(Eigen::Index i = 0; i < idx.size(); ++i)
            {
                if(isRight(data(axis, idx(i)), midpoint, rightInclusive))
                {
                    rightIdx(rightCnt) = idx(i);
                    ++rightCnt;
                }
                else
                {
                    leftIdx(leftCnt) = idx(i);
                    ++leftCnt;
                }
            }
        }

        Node *buildR(const Matrix &data, const IndexVector &idx, const Vector &mins, const Vector &maxes) const
        {
            // check for base case
            if(idx.size() <= bucketSize_)
            {
                // return a leaf node
                return new Node(idx);
            }
            else
            {
                // get distance between min and max values
                Vector diff = maxes - mins;
                // search for axis with longest distance
                Eigen::Index axis;
                diff.maxCoeff(&axis);
                // retrieve the corresponding values
                Scalar minval = mins(axis);
                Scalar maxval = maxes(axis);
                // check if min and max are the same
                // this basically means that all data points are the same
                if(minval == maxval)
                    return new Node(idx);

                IndexVector leftIdx, rightIdx;
                // find midpoint as mid of longest axis
                Scalar midpoint = minval + diff(axis) / 2;
                // split points into left and right
                splitMidpoint(data, idx, midpoint, axis, leftIdx, rightIdx,
                    false);

                // check if left side is empty
                if(leftIdx.size() == 0)
                {
                    midpoint = minval;
                    splitMidpoint(data, idx, midpoint, axis, leftIdx, rightIdx,
                        false);
                }
                // check if right side is empty
                if(rightIdx.size() == 0)
                {
                    midpoint = maxval;
                    splitMidpoint(data, idx, midpoint, axis, leftIdx, rightIdx,
                        true);
                }
                // no side should be empty by now
                assert(leftIdx.size() != 0);
                assert(rightIdx.size() != 0);

                Node *leftNode = nullptr;
                Node *rightNode = nullptr;

                #pragma omp sections
                {
                #pragma omp section
                {
                    // find left boundaries
                    Vector leftMins, leftMaxes;
                    findBoundaries(data, leftIdx, leftMins, leftMaxes);
                    // start recursion
                    leftNode = buildR(data, leftIdx, leftMins, leftMaxes);
                }
                #pragma omp section
                {
                    // find right boundaries
                    Vector rightMins, rightMaxes;
                    findBoundaries(data, rightIdx, rightMins, rightMaxes);
                    // start recursion
                    rightNode = buildR(data, rightIdx, rightMins, rightMaxes);
                }
                }

                return new Node(axis, midpoint, leftNode, rightNode);
            }
        }

        void clearR(Node *node)
        {
            if(node->hasLeft())
                clearR(node->left);
            if(node->hasRight())
                clearR(node->right);
            delete node;
        }

    public:

        KDTree()
            : dataCopy_(), data_(nullptr), bucketSize_(16),
            balance_(true), threads_(1), distance_(), root_(nullptr)
        {

        }

        /**
          * @param data data points for the kdtree; one point per column
          * @param copy if true copies the data, otherwise assumes static data */
        KDTree(const Matrix &data, const bool copy=false)
            : dataCopy_(), data_(nullptr), bucketSize_(16),
            balance_(true), threads_(1), distance_(), root_(nullptr)
        {
            setData(data, copy);
        }

        void setBucketSize(const Eigen::Index bucketSize)
        {
            bucketSize_ = bucketSize;
        }

        void setBalanceTree(const bool balance)
        {
            balance_ = balance;
        }

        void setThreads(const int threads)
        {
            threads_ = threads;
        }

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
                throw std::runtime_error("cannot build KDTree; data not set");

            if(data_->size() == 0)
                throw std::runtime_error("cannot build KDTree; data is empty");

            if(root_ != nullptr)
                clear();

            Eigen::Index n = data_->cols();
            IndexVector idx(n);
            for(Eigen::Index i = 0; i < n; ++i)
                idx(i) = i;

            #pragma omp parallel num_threads(threads_ > 0 ? threads_ : omp_get_max_threads())
            {
            #pragma omp single
            {
                Vector mins, maxes;
                findBoundaries(*data_, idx, mins, maxes);
                root_ = buildR(*data_, idx, mins, maxes);
            }
            }
        }

        void query(const Matrix &points, const size_t knn, Eigen::MatrixXi &indices,
            Matrix &distances) const
        {
            if(root_ == nullptr)
                throw std::runtime_error("cannot query KDTree; not built yet");

            distances.setZero(knn, points.cols());
            indices.setOnes(knn, points.cols());
            indices *= -1;

            #pragma omp parallel for num_threads(threads_ > 0 ? threads_ : omp_get_max_threads())
            for(Eigen::Index i = 0; i < points.cols(); ++i)
            {
                Node *n = root_;
                // search for the leaf this point would belong to
                while(n != nullptr && !n->isLeaf())
                {
                    Scalar val = points(i, n->axis);
                    n = val > n->splitpoint ? n->right : n->left;
                }
                assert(n != nullptr && n->isLeaf());

                size_t cnt = 0;
                // go through all points in this leaf node
                for(Eigen::Index j = 0; j < n->idx.size(); ++j)
                {
                    Eigen::Index c = n->idx(j);
                    Scalar dist = distance_(points.col(i), data_->col(c));
                    // if the result vector is not full simply append
                    if(cnt < knn)
                    {
                        indices(cnt, i) = c;
                        distances(cnt, i) = dist;
                        ++cnt;
                    }
                    else
                    {
                        Eigen::Index maxIdx;
                        // check which is the maximum distance
                        distances.col(i).maxCoeff(&maxIdx);
                        if(dist < distances(maxIdx, i))
                        {
                            indices(maxIdx, i) = c;
                            distances(maxIdx, i) = dist;
                        }
                    }
                }

            }
        }

        void clear()
        {
            if(root_ != nullptr)
            {
                clearR(root_);
                root_ = nullptr;
            }
            data_ = nullptr;
        }

        const Node *tree() const
        {
            return root_;
        }


    };

    typedef KDTree<float> KDTreef;
    typedef KDTree<double> KDTreed;
}

#endif
