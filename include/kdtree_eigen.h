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
        typename Distance=EuclideanDistanceSq<Scalar>,
        typename Index=typename Eigen::Matrix<Scalar, 1, 1>::Index>
    class KDTree
    {
    public:
        typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
        typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;
        typedef Eigen::Matrix<Scalar, 1, 1> Vector1;
        typedef Eigen::Matrix<Index, Eigen::Dynamic, 1> VectorI;
        typedef Eigen::Matrix<Index, Eigen::Dynamic, Eigen::Dynamic> MatrixI;

        /** Struct representing a node in the KDTree.
          * It can be either a inner node or a leaf node. */
        struct Node
        {
            /** Indices of data points in this leaf node. */
            VectorI indices;

            /** Left child of this inner node. */
            Node *left;
            /** Right child of this inner node. */
            Node *right;
            /** Axis of the axis aligned splitting hyper plane. */
            Index axis;
            /** Translation of the axis aligned splitting hyper plane. */
            Scalar splitpoint;

            Node()
                : indices(), left(nullptr), right(nullptr), axis(-1)
            {

            }

            /** Constructor for leaf nodes */
            Node(const VectorI &indices)
                : indices(indices), left(nullptr), right(nullptr),
                axis(-1)
            {

            }

            /** Constructor for inner nodes */
            Node(const Index axis, const Scalar splitpoint, Node *left,
                Node *right)
                : indices(), left(left), right(right), axis(axis),
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

        Index bucketSize_;
        bool sorted_;
        int threads_;
        Scalar maxDist_;

        Distance distance_;

        Node *root_;

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
        void findDataMinMax(const Matrix &data,
            const VectorI &currIndices,
            Vector &mins,
            Vector &maxes) const
        {
            assert(currIndices.size() > 0);
            assert(currIndices(0) >= 0 && currIndices(0) < data.cols());

            // initialize mins and maxes with first element of currIndices
            mins = data.col(currIndices(0));
            maxes = mins;
            // search for min / max values in data
            for(Index i = 1; i < currIndices.size(); ++i)
            {
                // retrieve data index
                Index c = currIndices(i);
                assert(c >= 0 && c < data.cols());
                // check min and max for each dimension individually
                for(Index r = 0; r < data.rows(); ++r)
                {
                    Scalar a = data(r, c);
                    mins(r) = a < mins(r) ? a : mins(r);
                    maxes(r) = a > maxes(r) ? a : maxes(r);
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

        void splitAtMidpoint(const Matrix &data,
            const VectorI &currIndices,
            const Scalar midpoint,
            const Index axis,
            VectorI &leftIndices,
            VectorI &rightIndices,
            bool rightInclusive) const
        {
            Index leftCnt = 0;
            Index rightCnt = 0;
            for(Index i = 0; i < currIndices.size(); ++i)
            {
                Scalar val = data(axis, currIndices(i));
                if(visitRightNode(val, midpoint, rightInclusive))
                    ++rightCnt;
                else
                    ++leftCnt;
            }

            leftIndices.resize(leftCnt);
            rightIndices.resize(rightCnt);
            leftCnt = 0;
            rightCnt = 0;

            for(Index i = 0; i < currIndices.size(); ++i)
            {
                Scalar val = data(axis, currIndices(i));
                if(visitRightNode(val, midpoint, rightInclusive))
                {
                    rightIndices(rightCnt) = currIndices(i);
                    ++rightCnt;
                }
                else
                {
                    leftIndices(leftCnt) = currIndices(i);
                    ++leftCnt;
                }
            }
        }

        Node *buildInnerNode(const Matrix &data, const VectorI &currIndices,
            const Vector &mins, const Vector &maxes) const
        {
            // get distance between min and max values
            Vector diff = maxes - mins;
            // search for axis with longest distance
            Index axis;
            diff.maxCoeff(&axis);
            // retrieve the corresponding values
            Scalar minval = mins(axis);
            Scalar maxval = maxes(axis);
            // check if min and max are the same
            // this basically means that all data points are the same
            if(minval == maxval)
                return new Node(currIndices);

            VectorI leftIndices, rightIndices;
            // find midpoint as mid of longest axis
            Scalar midpoint = minval + diff(axis) / 2;
            // split points into left and right
            splitAtMidpoint(data, currIndices, midpoint, axis, leftIndices,
                rightIndices, false);

            // check if left side is empty
            // this means all values are greater than midpoint
            if(leftIndices.size() == 0)
            {
                midpoint = minval;
                splitAtMidpoint(data, currIndices, midpoint, axis, leftIndices,
                    rightIndices, false);
            }
            // check if right side is empty
            // this means all values are leq than midpoint
            if(rightIndices.size() == 0)
            {
                midpoint = maxval;
                splitAtMidpoint(data, currIndices, midpoint, axis, leftIndices,
                    rightIndices, true);
            }

            // no side should be empty by now
            assert(leftIndices.size() != 0);
            assert(rightIndices.size() != 0);

            Node *leftNode = nullptr;
            Node *rightNode = nullptr;

            #ifndef _MSC_VER
            #pragma omp task shared(data, leftNode)
            #endif
            {
                // find left boundaries
                Vector leftMins, leftMaxes;
                findDataMinMax(data, leftIndices, leftMins, leftMaxes);
                // start recursion
                leftNode = buildR(data, leftIndices, leftMins, leftMaxes);
            }

            #ifndef _MSC_VER
            #pragma omp task shared(data, rightNode)
            #endif
            {
                // find right boundaries
                Vector rightMins, rightMaxes;
                findDataMinMax(data, rightIndices, rightMins, rightMaxes);
                // start recursion
                rightNode = buildR(data, rightIndices, rightMins, rightMaxes);
            }

            #ifndef _MSC_VER
            #pragma omp taskwait
            #endif

            assert(leftNode != nullptr && rightNode != nullptr);

            return new Node(axis, midpoint, leftNode, rightNode);
        }

        Node *buildR(const Matrix &data,
            const VectorI &currIndices,
            const Vector &mins,
            const Vector &maxes) const
        {
            // check for base case
            if(currIndices.size() <= bucketSize_)
                return new Node(currIndices);
            else
                return buildInnerNode(data, currIndices, mins, maxes);
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

            Scalar val = queryPoints(n->axis, col);

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
            : dataCopy_(), data_(nullptr), bucketSize_(16),
            sorted_(true), threads_(1), maxDist_(0), distance_(), root_(nullptr)
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

            if(root_ != nullptr)
                clearRoot();

            Index n = data_->cols();
            VectorI currIndices(n);
            for(Index i = 0; i < n; ++i)
                currIndices(i) = i;

            #ifndef _MSC_VER
            #pragma omp parallel num_threads(threads_)
            #pragma omp single
            #endif
            {
                Vector mins, maxes;
                findDataMinMax(*data_, currIndices, mins, maxes);
                root_ = buildR(*data_, currIndices, mins, maxes);
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
            if(root_ != nullptr)
                clearRoot();
            data_ = nullptr;
        }

        const Node *tree() const
        {
            return root_;
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
