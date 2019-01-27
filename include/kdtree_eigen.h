/* kdtree_eigen.h
 *
 *     Author: Fabian Meyer
 * Created On: 08 Nov 2018
 *    License: MIT
 *
 * Implementation is based on cKDtree of the scipy project and the splitting
 * midpoint rule described in "Analysis of Approximate Nearest Neighbor
 * Searching with Clustered Point Sets" by Songrit Maneewongvatana and David M.
 * Mount.
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

    template <typename Scalar>
    struct ManhattenDistance
    {
        typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;

        Scalar operator()(const Vector &vecA, const Vector &vecB) const
        {
            return (vecA - vecB).abs().sum();
        }
    };

    template <typename Scalar>
    struct EuclideanDistance
    {
        typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;

        Scalar operator()(const Vector &vecA, const Vector &vecB) const
        {
            return (vecA - vecB).norm();
        }
    };

    template <typename Scalar>
    struct EuclideanDistanceSq
    {
        typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;

        Scalar operator()(const Vector &vecA, const Vector &vecB) const
        {
            return (vecA - vecB).squaredNorm();
        }
    };

    /** Class for performing k nearest neighbour searches. */
    template<typename Scalar,
        typename Distance=EuclideanDistanceSq<Scalar>>
    class KDTree
    {
    public:
        typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
        typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;
        typedef Eigen::Matrix<Scalar, 1, 1> Vector1;
        typedef typename Matrix::Index Index;
        typedef Eigen::Matrix<Index, Eigen::Dynamic, 1> VectorI;
        typedef Eigen::Matrix<Index, Eigen::Dynamic, Eigen::Dynamic> MatrixI;

        /** Struct representing a node in the KDTree.
          * It can be either a inner node or a leaf node. */
        struct Node
        {
            /** Indices of data points in this leaf node. */
            Index startIdx;
            Index length;

            /** Left child of this inner node. */
            Index left;
            /** Right child of this inner node. */
            Index right;
            /** Axis of the axis aligned splitting hyper plane. */
            Index splitaxis;
            /** Translation of the axis aligned splitting hyper plane. */
            Scalar splitpoint;

            Node()
                : startIdx(0), length(0), left(-1), right(-1),
                splitaxis(-1), splitpoint(0)
            {

            }

            /** Constructor for leaf nodes */
            Node(const Index startIdx, const Index length)
                : startIdx(startIdx), length(length), left(-1), right(-1),
                splitaxis(-1), splitpoint(0)
            {

            }

            /** Constructor for inner nodes */
            Node(const Index splitaxis, const Scalar splitpoint,
                const Index left, const Index right)
                : startIdx(0), length(0), left(left), right(right),
                splitaxis(splitaxis), splitpoint(splitpoint)
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
                return left >= 0;
            }

            bool hasRight() const
            {
                return right >= 0;
            }
        };

    private:
        Matrix dataCopy_;
        const Matrix *data_;
        std::vector<Index> indices_;
        std::vector<Node> nodes_;
        Node *root_;

        Index bucketSize_;
        bool sorted_;
        bool compact_;
        bool balanced_;
        int threads_;
        Scalar maxDist_;

        Distance distance_;

        struct QueryResult
        {
            MatrixI &indices;
            Matrix &distances;
            Index count;
            Index min;
            Index max;

            QueryResult(MatrixI &indices, Matrix &distances)
            :indices(indices), distances(distances), count(0), min(0), max(0)
            { }

            bool isFull() const
            {
                return count >= indices.rows();
            }

            Scalar getMaxDistance(const Index col) const
            {
                return distances(max, col);
            }

            Scalar getMinDistance(const Index col) const
            {
                return distances(min, col);
            }
        };

        /** Finds the minimum and maximum values of each dimension (row) in the
         *  data matrix. Only respects the columns specified by the index
         *  vector. */
        void findDataMinMax(const Index startIdx,
            const Index length,,
            Vector &mins,
            Vector &maxes) const
        {
            assert(length > 0);
            assert(startIdx >= 0);
            assert(startIdx + length <= indices_.size());

            const Matrix &data = *data_;

            // initialize mins and maxes with first element of currIndices
            mins = data.col(indices_[startIdx]);
            maxes = mins;
            // search for min / max values in data
            for(Index i = 0; i < length; ++i)
            {
                // retrieve data index
                Index col = indices_[startIdx + i];
                assert(col >= 0 && col < data.cols());
                // check min and max for each dimension individually
                for(Index j = 0; j < data.rows(); ++j)
                {
                    Scalar val = data(j, col);
                    mins(j) = val < mins(j) ? val : mins(j);
                    maxes(j) = val > maxes(j) ? val : maxes(j);
                }
            }
        }

        /** Predicate to determine if the right or left node should be visited.
         *  @return returns true if the right node should be visited */
        bool visitRightNode(const Scalar point,
            const Scalar midpoint,
            bool inclusive)
            const
        {
            return (inclusive && point >= midpoint) || point > midpoint;
        }

        Index buildInnerNode(const Index startIdx,
            const Index length,
            const Vector &mins,
            const Vector &maxes) const
        {
            assert(startIdx >= 0);
            assert(length > 0);
            assert(startIdx  + length <= indices_.size());
            assert(maxes.size() == mins.size());

            const Matrix &data = *data_;

            // search for axis with longest distance
            Index splitaxis = 0;
            Scalar splitsize = 0;
            for(Index i = 0; i < maxes.size(); ++i)
            {
                Scalar diff = maxes(i) - mins(i);
                if(diff > splitsize)
                {
                    splitaxis = i;
                    splitsize = diff;
                }
            }
            // retrieve the corresponding values
            Scalar minval = mins(splitaxis);
            Scalar maxval = maxes(splitaxis);
            // check if min and max are the same
            // this basically means that all data points are the same
            if(minval == maxval)
            {
                nodes_.push_back(Node(startIdx, length));
                return nodes_.size() - 1;
            }

            // determine split point
            Scalar splitpoint;
            // check if tree should be balanced
            if(balanced_)
            {
                // use median for splitpoint
                // TODO currIndices is supposed to be sorted along split axis
                // auto compPred =
                //     [const &data, splitaxis](const Index lhs, const Index rhs)
                //     { return data(splitaxis, lhs) < data(splitaxis, rhs); }
                // std::sort(indices_.begin() + startIdx,
                //     indices_.begin() + startIdx + length,
                //     compPred);

                Index idx = indices_[startIdx + length / 2];
                splitpoint = data(splitaxis, idx);
            }
            else
            {
                // use sliding midpoint rule
                splitpoint = (minval + maxval) / 2;
            }

            Index leftIdx = startIdx;
            Index rightIdx = startIdx + length - 1;
            while(leftIdx <= rightIdx)
            {
                Scalar leftVal = data(splitaxis, indices_[leftIdx]);
                Scalar rightVal = data(splitaxis, indices_[rightIdx]);

                if(leftVal < splitpoint)
                {
                    // left value is less than split point
                    // keep it on left side
                    ++leftIdx;
                }
                else if(rightVal >= splitpoint)
                {
                    // right value is greater than split point
                    // keep it on right side
                    --rightIdx;
                }
                else
                {
                    // right value is less than splitpoint and left value is
                    // greater than split point
                    // simply swap sides
                    Index tmpIdx = indices_[leftIdx];
                    indices_[leftIdx] = indices_[rightIdx];
                    indices_[rightIdx] = tmpIdx;
                    ++leftIdx;
                    --rightIdx;
                }
            }

            if(leftIdx == startIdx)
            {
                // no values on left side, resolve trivial split
                // find value with minimum distance to splitpoint
                Index minIdx = startIdx;
                splitpoint = data(splitaxis, indices_[minIdx]);
                for(Index i = 0; i < length; ++i)
                {
                    Index idx = startIdx + i;
                    Scalar val = data(splitaxis, indices_[idx]);
                    if(val < splitpoint)
                    {
                        minIdx = idx;
                        splitpoint = val;
                    }
                }
                // put value with minimum distance on the left
                // this way there is exactly one value on the left
                Index tmpIdx = indices_[startIdx];
                indices_[startIdx] = indices_[minIdx];
                indices_[minIdx] = tmpIdx;
                leftIdx = startIdx + 1;
                rightIdx = startIdx;
            }
            else if(leftIdx == startIdx + length)
            {
                // no values on right side, resolve trivial split
                // find value with maximum distance to splitpoint
                Index maxIdx = startIdx + length - 1;
                splitpoint = data(splitaxis, indices_[maxIdx]);
                for(Index i = 0; i < length; ++i)
                {
                    Index idx = startIdx + i;
                    Scalar val = data(splitaxis, indices_[idx])
                    if(val > splitpoint)
                    {
                        maxIdx = idx;
                        splitpoint = val;
                    }
                }
                // put value with maximum distance on the right
                // this way there is exactly one value on the right
                Index tmpIdx = indices_[startIdx + length - 1];
                indices_[startIdx + length - 1] = indices_[maxIdx];
                indices_[maxIdx] = tmpIdx;
                leftIdx = startIdx + length - 1;
                rightIdx = startIdx + length - 2;
            }

            Index leftNode = -1;
            Index rightNode = -1;

            Index leftStart = startIdx;
            Index rightStart = leftIdx;
            Index leftLength = leftIdx - startIdx;
            Index rightLength = length - leftLength;

            if(compact_)
            {
                // recompute mins and maxes to make the tree more compact
                Vector minsN, maxesN;
                findDataMinMax(leftStart, leftLength, minsN, maxesN);
                leftNode = buildR(leftStart, leftLength, minsN, maxesN);

                findDataMinMax(rightStart, rightLength, minsN, maxesN);
                rightNode = buildR(rightStart, rightLength, minsN, maxesN);
            }
            else
            {
                // just re-use mins and maxes, but set splitaxies to value of
                // splitpoint
                Vector mids(maxes.size());
                for(Index i = 0; i < maxes.size(); ++i)
                    mids(i) = maxes(i);
                mids(splitaxis) = splitpoint;

                leftNode = buildR(leftStart, leftLength, mins, mids);

                for(Index i = 0; i < mins.size(); ++i)
                    mids(i) = mins(i);
                mids(splitaxis) = splitpoint;
                rightNode = buildR(leftStart, leftLength, mids, maxes);
            }

            nodes_.push_back(Node(splitaxis, splitpoint, leftNode, rightNode));
            return nodes_.size() - 1;
        }

        Index buildR(const Index startIdx,
            const Index length,
            const Vector &mins,
            const Vector &maxes) const
        {
            // check for base case
            if(length <= bucketSize_)
            {
                nodes_.push_back(Node(startIdx, length));
                return nodes_.size() - 1;
            }
            else
                return buildInnerNode(startIdx, length, mins, maxes);
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
            const Index col,
            const Matrix &queryPoints,
            QueryResult &queryResult) const
        {
            assert(n->isLeaf());

            // go through all points in this leaf node
            for(Index i = 0; i < n->indices.size(); ++i)
            {
                // retrieve index of this child
                Index c = n->indices(i);
                Scalar dist = distance_(queryPoints.col(col), data_->col(c));

                // if distance is greater than maximum distance then skip
                if(maxDist_ > 0 && dist >= maxDist_)
                    continue;

                // check if all places in the result vector are already in use
                if(!queryResult.isFull())
                {
                    // if the result vector is not full simply append
                    queryResult.indices(queryResult.count, col) = c;
                    queryResult.distances(queryResult.count, col) = dist;
                    ++queryResult.count;

                    Scalar currMax = queryResult.getMaxDistance(col);
                    Scalar currMin = queryResult.getMinDistance(col);
                    if(dist > currMax)
                        queryResult.max = c;
                    if(dist < currMin)
                        queryResult.min = c;
                }
                else
                {
                    // result vector is full, retrieve current maximum distance
                    Scalar currMax = queryResult.getMaxDistance(col);
                    // check if this distance is an improvement
                    if(dist < currMax)
                    {
                        // replace old max
                        queryResult.indices(queryResult.max, col) = c;
                        queryResult.distances(queryResult.max, col) = dist;
                        // find new maximum value
                        queryResult.distances.col(col).maxCoeff(&queryResult.max);
                        // update minimum
                        Scalar currMin = queryResult.getMinDistance(col);
                        if(dist < currMin)
                            queryResult.min = c;
                    }
                }
            }
        }

        void queryInnerNode(const Node *n,
            const Index col,
            const Matrix &queryPoints,
            QueryResult &queryResult) const
        {
            assert(n->isInner());

            Scalar val = queryPoints(n->splitaxis, col);

            // check if right or left child should be visited
            bool visitRight = visitRightNode(val, n->splitpoint, true);
            if(visitRight)
                queryR(n->right, col, queryPoints, queryResult);
            else
                queryR(n->left, col, queryPoints, queryResult);

            // get distance to midpoint
            Scalar distMid = distance_(Vector1(val), Vector1(n->splitpoint));

            // if distance is greater than maximum distance then return
            // the points on the other side cannot be closer then
            if(maxDist_ > 0 && distMid >= maxDist_)
                return;

            if(!queryResult.isFull())
            {
                // if result is not full yet, just visit the other child, too
                if(visitRight)
                    queryR(n->left, col, queryPoints, queryResult);
                else
                    queryR(n->right, col, queryPoints, queryResult);
            }
            else
            {
                // if cnt is full check if we could improve by visiting
                // the other child
                Scalar currMax = queryResult.getMaxDistance(col);
                // check if distance to splitting point is lower than
                // current maximum distance
                if(distMid < currMax)
                {
                    // if right was already visited check left child now
                    if(visitRight)
                        queryR(n->left, col, queryPoints, queryResult);
                    else
                        queryR(n->right, col, queryPoints, queryResult);
                }
            }
        }

        void queryR(const Node *n,
            const Index col,
            const Matrix &queryPoints,
            QueryResult &queryResult) const
        {
            if(n->isLeaf())
                queryLeafNode(n, col, queryPoints, queryResult);
            else
                queryInnerNode(n, col, queryPoints, queryResult);
        }

        Index depthR(Node *n)
        {
            if(n == nullptr)
                return 0;
            else
            {
                Index left = depthR(n->left);
                Index right = depthR(n->right);
                return std::max(left, right) + 1;
            }
        }

    public:

        /** Constructs an empty KDTree. */
        KDTree()
            : dataCopy_(), data_(nullptr), indices_(), nodes_(), root_(nullptr),
            bucketSize_(16), sorted_(true), threads_(1), maxDist_(0),
            distance_()
        {

        }

        /** Constructs KDTree with the given data. This does not build the
          * the index of the tree.
          * @param data NxM matrix, M points of dimension N
          * @param copy if true copies the data, otherwise assumes static data */
        KDTree(const Matrix &data, const bool copy=false)
            : KDTree()
        {
            setData(data, copy);
        }

        ~KDTree()
        {

        }

        /** Set the maximum amount of data points per leaf in the tree (aka
          * bucket size).
          * @param bucketSize amount of points per leaf. */
        void setBucketSize(const Index bucketSize)
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

        void setCompactTree(const bool compact)
        {
            compact_ = compact;
        }

        /** Set the amount of threads that should be used for building and
          * querying the tree.
          * OMP has to be enabled for this to work.
          * @param threads amount of threads, 0 for optimal choice */
        void setThreads(const unsigned int threads)
        {
            threads_ = threads;
        }

        /** Set the maximum distance for querying the tree.
          * The search will be pruned if the maximum distance is set to any
          * positive number.
          * @param maxDist maximum distance, 0 for no limit */
        void setMaxDistance(const Scalar maxDist)
        {
            maxDist_ = maxDist;
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

            clear();

            indices_.resize(data_->cols());
            for(Index i = 0; i < indices_.size(); ++i)
                indices_[i] = i;

            Vector mins, maxes;
            Index startIdx = 0;
            Index length = indices_.size();

            findDataMinMax(startIdx, length, mins, maxes);

            root_ = buildR(startIdx, length, mins, maxes);
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
        void query(const Matrix &queryPoints,
            const size_t knn,
            MatrixI &indices,
            Matrix &distances) const
        {
            if(root_ == nullptr)
                throw std::runtime_error("cannot query KDTree; not built yet");

            if(queryPoints.rows() != dimension())
                throw std::runtime_error("cannot query KDTree; data and query points do not have same dimension");

            distances.setConstant(knn, queryPoints.cols(), 1.0 / 0.0);
            indices.setConstant(knn, queryPoints.cols(), -1);

            #pragma omp parallel for num_threads(threads_)
            for(Index i = 0; i < queryPoints.cols(); ++i)
            {
                QueryResult queryResult(indices, distances);
                queryR(root_, i, queryPoints, queryResult);
            }
        }

        void clear()
        {
            nodes_.clear();
            root_ = nullptr;
        }

        const Node &tree() const
        {
            if(root_ == nullptr)
                throw std::runtime_error("KDTree not built yet");

            return *root_;
        }

        Index size() const
        {
            return data_ == nullptr ? 0 : data_->cols();
        }

        Index dimension() const
        {
            return data_ == nullptr ? 0 : data_->rows();
        }

        Index depth() const
        {
            return depthR(root_);
        }
    };

    typedef KDTree<float> KDTreef;
    typedef KDTree<double> KDTreed;
}

#endif
