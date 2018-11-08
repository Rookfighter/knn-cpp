
#ifndef KDT_KDTREE_EIGEN_H_
#define KDT_KDTREE_EIGEN_H_

namespace kdt
{
    template <typename Scalar, int P, int Dim=Eigen::Dynamic>
    struct MinkowskiDistance
    {
        typedef Eigen::Matrix<Scalar, Dim, 1> Vector;

        Scalar operator()(const Vector &vecA, const Vector &vecB)
        {
            return (vecA - vecB).lpNorm<P>();
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
        typedef Eigen::Matrix<size_t, Eigen::Dynamic, 1> Index;
    private:
        struct Node
        {
            /** Indices of data points in this leaf node. */
            Index idx;

            /** Left child of this inner node. */
            Node *left;
            /** Right child of this inner node. */
            Node *right;
            /** Axis of the axis aligned splitting hyper plane. */
            size_t axis;
            /** Translation of the axis aligned splitting hyper plane. */
            Scalar splitpoint;

            Node()
                : left(nullptr), right(nullptr)
            {

            }

            /** Constructor for leaf nodes */
            Node(const Index &idx)
                :idx(idx), left(nullptr), right(nullptr)
            {

            }

            /** Constructor for inner nodes */
            Node(const size_t axis, const Scalar splitpoint, Node *left,
                Node *right)
                : idx(), left(left), right(right), idx(), axis(axis),
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

        Matrix dataCopy_;
        const Matrix *data_;

        size_t bucketSize_;
        bool balance_;
        size_t threads_;

        Distance distance_;

        Node *root_;

        size_t argmax(const Vector &vec) const
        {
            assert(vec.size() > 0);
            Scalar max = vec(0);
            size_t idx = 0;
            for(size_t i = 1; i < vec.size(); ++i)
                if(vec(i) > max)
                    idx = i;
            return idx;
        }

        void findBoundaries(const Matrix &data, const Index &idx, Vector &mins,
            Vector &maxes) const
        {
            assert(idx.size() > 0);
            mins = data.col(idx(0));
            maxes = mins;
            for(size_t i = 1; i < idx.size(); ++i)
            {
                size_t c = idx(i);
                for(size_t r = 0; r < outVec.size(); ++r)
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

        void splitMidpoint(const Matrix &data, const Index &idx,
            const Scalar midpoint, const size_t axis, Index &leftIdx,
            Index &rightIdx, bool rightInclusive)
        {
            size_t leftCnt = 0;
            size_t rightCnt = 0;
            for(size_t i = 0; i < idx.size(); ++i)
            {
                if(isRight(data(axis, idx(i)), midpoint))
                    ++rightCnt;
                else
                    ++leftCnt;
            }

            leftIdx.resize(leftCnt);
            rightIdx.resize(rightCnt);
            leftCnt = 0;
            rightCnt = 0;

            for(size_t i = 0; i < idx.size(); ++i)
            {
                if(isRight(data(axis, idx(i)), midpoint))
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

        Node *buildR(const Index &idx, const Vector &mins, const Vector &maxes)
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
                size_t axis = argmax(diff);
                // retrieve the corresponding values
                Scalar minval = mins(axis);
                Scalar maxval = maxes(axis);
                // check if min and max are the same
                // this basically means that all data points are the same
                if(minval == maxval)
                    return new Node(idx);

                Index leftIdx, rightIdx;
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
                // check if left side is still empty
                // this again means that all points are the same
                assert(leftIdx.size() != 0);
                assert(rightIdx.size() != 0);

                // find left boundaries
                Vector leftMins, leftMaxes;
                findBoundaries(data, leftIdx, leftMins, leftMaxes);
                // find right boundaries
                Vector rightMins, rightMaxes;
                findBoundaries(data, rightIdx, rightMins, rightMaxes);

                // start recursion
                Node* leftNode = buildR(data, leftIdx, leftMins, leftMaxes);
                Node* rightNode = buildR(data, rightIdx, rightMins, rightMaxes);

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

        /**
          * @param data data points for the kdtree; one point per column
          * @param copy if true copies the data, otherwise assumes static data */
        KDTree(const Matrix &data, const bool copy=false)
            : dataCopy_(), data_(nullptr), leafSize_(16),
            balance_(true), threads_(1), distance_()
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

        void setBucketSize(const size_t bucketSize)
        {
            bucketSize_ = bucketSize;
        }

        void setBalanceTree(const bool balance)
        {
            balance_ = balance;
        }

        void setThreads(const size_t threads)
        {
            threads_ = threads;
        }

        void build()
        {
            if(data_->cols() == 0)

            if(root_ != nullptr)
                clear();

            size_t n = data_->cols();
            Index idx(n);
            for(size_t i = 0; i < n; ++i)
                idx(i) = i;

            Vector mins, maxes;
            findBoundaries(*data_, idx, mins, maxes);

            root_ = buildR(*data_, idx, mins, maxes);
        }

        void query(const Matrix &points, const size_t knn, ) const
        {

        }

        void clear()
        {
            if(root_ != nullptr)
            {
                clearR(root_);
                root_ = nullptr;
            }
        }
    };

    typedef KDTree<float> KDTreef;
    typedef KDTree<double> KDTreed;
}

#endif
