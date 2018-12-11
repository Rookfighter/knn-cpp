/* kdtree_eigen.h
 *
 *     Author: Fabian Meyer
 * Created On: 08 Nov 2018
 */

#ifndef KDT_KDTREE_EIGEN_H_
#define KDT_KDTREE_EIGEN_H_

#include <Eigen/Geometry>

namespace kdt
{
    /** Functor for minkowski distance. */
    template <typename Scalar, int P>
    struct MinkowskiDistance
    {
        typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;

        Scalar operator()(const Vector &vecA, const Vector &vecB) const
        {
            return (vecA - vecB).template lpNorm<P>();
        }
    };

    /** Class for performing k nearest neighbour searches. */
    template<typename Scalar,
        typename Distance=MinkowskiDistance<Scalar, 2>>
    class KDTree
    {
    public:
        typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
        typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;
        typedef Eigen::Matrix<Eigen::Index, Eigen::Dynamic, Eigen::Dynamic> IndexMatrix;
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
                : idx(idx), left(nullptr), right(nullptr),
                axis(-1)
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

            bool isInner() const
            {
                return hasLeft() && hasRight();
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
        bool sorted_;
        int threads_;

        Distance distance_;

        Node *root_;

        void findBoundaries(const Matrix &data, const IndexVector &idx,
            Vector &mins, Vector &maxes) const
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

        Node *buildInnerNode(const Matrix &data, const IndexVector &idx,
            const Vector &mins, const Vector &maxes) const
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

            #ifndef _MSC_VER
            #pragma omp task shared(data, leftNode)
            #endif
            {
                // find left boundaries
                Vector leftMins, leftMaxes;
                findBoundaries(data, leftIdx, leftMins, leftMaxes);
                // start recursion
                leftNode = buildR(data, leftIdx, leftMins, leftMaxes);
            }

            #ifndef _MSC_VER
            #pragma omp task shared(data, rightNode)
            #endif
            {
                // find right boundaries
                Vector rightMins, rightMaxes;
                findBoundaries(data, rightIdx, rightMins, rightMaxes);
                // start recursion
                rightNode = buildR(data, rightIdx, rightMins, rightMaxes);
            }

            #ifndef _MSC_VER
            #pragma omp taskwait
            #endif

            assert(leftNode != nullptr && rightNode != nullptr);

            return new Node(axis, midpoint, leftNode, rightNode);
        }

        Node *buildR(const Matrix &data, const IndexVector &idx,
            const Vector &mins, const Vector &maxes) const
        {
            // check for base case
            if(idx.size() <= bucketSize_)
                return new Node(idx);
            else
                return buildInnerNode(data, idx, mins, maxes);
        }

        void clearRoot()
        {
            clearR(root_);
            root_ = nullptr;
        }

        void clearR(Node *node)
        {
            if(node->hasLeft())
                clearR(node->left);
            if(node->hasRight())
                clearR(node->right);
            delete node;
        }

        void queryLeafNode(const Node *n,
            const Eigen::Index col,
            const Matrix &queryPoints,
            Eigen::MatrixXi &indices,
            Matrix &distances,
            Eigen::Index &cnt) const
        {
            assert(n->isLeaf());

            // go through all points in this leaf node
            for(Eigen::Index j = 0; j < n->idx.size(); ++j)
            {
                // retrieve index of this child
                Eigen::Index c = n->idx(j);
                Scalar dist = distance_(queryPoints.col(col), data_->col(c));
                // check if all places in the result vector are already in use
                if(cnt < distances.rows())
                {
                    // if the result vector is not full simply append
                    indices(cnt, col) = c;
                    distances(cnt, col) = dist;
                    ++cnt;
                }
                else
                {
                    // result vector is full, retrieve current maximum distance
                    Eigen::Index maxIdx;
                    distances.col(col).maxCoeff(&maxIdx);
                    // check if this distance is an improvement
                    if(dist < distances(maxIdx, col))
                    {
                        indices(maxIdx, col) = c;
                        distances(maxIdx, col) = dist;
                    }
                }
            }
        }

        void queryInnerNode(const Node *n,
            const Eigen::Index col,
            const Matrix &queryPoints,
            Eigen::MatrixXi &indices,
            Matrix &distances,
            Eigen::Index &cnt) const
        {
            assert(n->isInner());

            Scalar val = queryPoints(n->axis, col);
            // check if right or left child should be visited
            bool visitRight = isRight(val, n->splitpoint, true);
            if(visitRight)
                queryR(n->right, col, queryPoints, indices, distances, cnt);
            else
                queryR(n->left, col, queryPoints, indices, distances, cnt);

            if(cnt < distances.rows())
            {
                // if cnt is not full yet, just visit the other child, too
                if(visitRight)
                    queryR(n->left, col, queryPoints, indices, distances, cnt);
                else
                    queryR(n->right, col, queryPoints, indices, distances, cnt);
            }
            else
            {
                // if cnt is full check if we could improve by visiting
                // the other child
                // check which is the minimum distance so far
                Scalar dist = distances.col(col).minCoeff();
                // get distance to splitting point
                Scalar diff = val - n->splitpoint;
                diff = diff < 0 ? -diff : diff;
                // check if distance to splitting point is lower than
                // current minimum distance
                if(diff < dist)
                {
                    // if right was already visited check left child now
                    if(visitRight)
                        queryR(n->left, col, queryPoints, indices, distances, cnt);
                    else
                        queryR(n->right, col, queryPoints, indices, distances, cnt);
                }
            }
        }

        void queryR(const Node *n,
            const Eigen::Index col,
            const Matrix &queryPoints,
            Eigen::MatrixXi &indices,
            Matrix &distances,
            Eigen::Index &cnt) const
        {
            // if node is a leaf just check it for neighbours
            if(n->isLeaf())
                queryLeafNode(n, col, queryPoints, indices, distances, cnt);
            else
                queryInnerNode(n, col, queryPoints, indices, distances, cnt);
        }

    public:

        /** Constructs an empty KDTree. */
        KDTree()
            : dataCopy_(), data_(nullptr), bucketSize_(16),
            sorted_(true), threads_(1), distance_(), root_(nullptr)
        {

        }

        /** Constructs KDTree with the given data. This does not build the
          * the index of the tree.
          * @param data NxM matrix, M points of dimension N
          * @param copy if true copies the data, otherwise assumes static data */
        KDTree(const Matrix &data, const bool copy=false)
            : dataCopy_(), data_(nullptr), bucketSize_(16),
            sorted_(true), threads_(1), distance_(), root_(nullptr)
        {
            setData(data, copy);
        }

        ~KDTree()
        {
            clear();
        }

        /** Set the maximum amount of data points per leaf in the tree (aka
          * bucket size).
          * @param bucketSize amount of points per leaf. */
        void setBucketSize(const Eigen::Index bucketSize)
        {
            bucketSize_ = bucketSize;
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
          * OMP has to be enabled for this to work.
          * @param threads amount of threads, -1 for optimal choice */
        void setThreads(const int threads)
        {
            threads_ = threads;
        }

        /** Set the data points used for this tree.
          * This does not build the tree.
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

        /** Builds the search index of the tree.
          * Data has to be set and must be non-empty. */
        void build()
        {
            if(data_ == nullptr)
                throw std::runtime_error("cannot build KDTree; data not set");

            if(data_->size() == 0)
                throw std::runtime_error("cannot build KDTree; data is empty");

            if(root_ != nullptr)
                clearRoot();

            Eigen::Index n = data_->cols();
            IndexVector idx(n);
            for(Eigen::Index i = 0; i < n; ++i)
                idx(i) = i;

            #ifndef _MSC_VER
            #pragma omp parallel num_threads(threads_ > 0 ? threads_ : omp_get_max_threads())
            #endif
            {
                #ifndef _MSC_VER
                #pragma omp single
                #endif
                {
                    Vector mins, maxes;
                    findBoundaries(*data_, idx, mins, maxes);
                    root_ = buildR(*data_, idx, mins, maxes);
                }
            }
        }

        /** Queries the tree for the nearest neighbours of the given query
          * points.
          * The tree has to be built before it can be queried. The queryPoints
          * have to have the same dimension as the data points of the tree.
          * @param queryPoints NxM matrix, M points of dimension N
          * @param knn amount of neighbours to be found
          * @param indices KNNxM matrix, indices of neighbours in the data set
          * @param distances KNNxM matrix, distance between querypoint and
          *        neighbours */
        void query(const Matrix &queryPoints, const size_t knn,
            Eigen::MatrixXi &indices, Matrix &distances) const
        {
            if(root_ == nullptr)
                throw std::runtime_error("cannot query KDTree; not built yet");

            if(queryPoints.rows() != dimension())
                throw std::runtime_error("cannot query KDTree; data and query points do not have same dimension");

            distances.setZero(knn, queryPoints.cols());
            indices.setOnes(knn, queryPoints.cols());
            indices *= -1;

            #pragma omp parallel for num_threads(threads_ > 0 ? threads_ : omp_get_max_threads())
            for(Eigen::Index i = 0; i < queryPoints.cols(); ++i)
            {
                Eigen::Index cnt = 0;
                queryR(root_, i, queryPoints, indices, distances, cnt);
            }
        }

        void clear()
        {
            if(root_ != nullptr)
                clearRoot();
            data_ = nullptr;
        }

        const Node *tree() const
        {
            return root_;
        }

        Eigen::Index size() const
        {
            return data_ == nullptr ? 0 : data_->cols();
        }

        Eigen::Index dimension() const
        {
            return data_ == nullptr ? 0 : data_->rows();
        }


    };

    typedef KDTree<float> KDTreef;
    typedef KDTree<double> KDTreed;
}

#endif
