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
#include <vector>
#include <algorithm>

namespace kdt
{
    /** Functor for manhatten distance. This the same as the L1 minkowski
      * distance but more efficient.*/
    template <typename Scalar>
    struct ManhattenDistance
    {
        typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;
        typedef typename Vector::Index Index;

        Scalar unrooted(const Scalar *vecA,
            const Scalar *vecB,
            const Index len) const
        {
            Scalar result = 0.0;
            for(Index i = 0; i < len; ++i)
            {
                Scalar diff = vecA[i] - vecB[i];
                result += power(diff);
            }
            return result;
        }

        Scalar rooted(const Scalar *vecA,
            const Scalar *vecB,
            const Index len) const
        {
            return unrooted(vecA, vecB, len);
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

    /** Functor for euclidean distance. This the same as the L2 minkowski
      * distance but more efficient. */
    template <typename Scalar>
    struct EuclideanDistance
    {
        typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;
        typedef typename Vector::Index Index;

        Scalar unrooted(const Scalar *vecA,
            const Scalar *vecB,
            const Index len) const
        {
            Scalar result = 0.0;
            for(Index i = 0; i < len; ++i)
            {
                Scalar diff = vecA[i] - vecB[i];
                result += power(diff);
            }
            return result;
        }

        Scalar rooted(const Scalar *vecA,
            const Scalar *vecB,
            const Index len) const
        {
            return root(unrooted(vecA, vecB, len));
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

    /** Functor for general minkowski distance. The infinite version is only
      * available through the ChebyshevDistance functor. */
    template <typename Scalar, int P>
    struct MinkowskiDistance
    {
        typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;
        typedef typename Vector::Index Index;

        Scalar unrooted(const Scalar *vecA,
            const Scalar *vecB,
            const Index len) const
        {
            Scalar result = 0.0;
            for(Index i = 0; i < len; ++i)
            {
                Scalar diff = vecA[i] - vecB[i];
                result += power(diff);
            }
            return result;
        }

        Scalar rooted(const Scalar *vecA,
            const Scalar *vecB,
            const Index len) const
        {
            return root(unrooted(vecA, vecB, len));
        }

        Scalar root(const Scalar val) const
        {
            return std::pow(val, 1.0 / static_cast<Scalar>(P));
        }

        Scalar power(const Scalar val) const
        {
            Scalar result = std::abs(val);
            for(int i = 1; i < P; ++i)
                result *= result;
            return result;
        }
    };

    /** Chebyshev distance functor. This distance is the same as infinity
      * minkowski distance. */
    template<typename Scalar>
    struct ChebyshevDistance
    {
        typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;
        typedef typename Vector::Index Index;

        Scalar unrooted(const Scalar *vecA,
            const Scalar *vecB,
            const Index len) const
        {
            Scalar result = 0.0;
            for(Index i = 0; i < len; ++i)
            {
                Scalar diff = vecA[i] - vecB[i];
                result = std::max(result, power(diff));
            }
            return result;
        }

        Scalar rooted(const Scalar *vecA,
            const Scalar *vecB,
            const Index len) const
        {
            return root(unrooted(vecA, vecB, len));
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

    /** Class for performing k nearest neighbour searches. */
    template<typename Scalar,
        typename Distance=EuclideanDistance<Scalar>>
    class KDTree
    {
    public:
        typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
        typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;
        typedef typename Matrix::Index Index;
        typedef Eigen::Matrix<Index, Eigen::Dynamic, Eigen::Dynamic> MatrixI;
        typedef Eigen::Matrix<Index, Eigen::Dynamic, 1> VectorI;

    private:
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

        class QueryHeap
        {
        private:
            Index *indices_;
            Scalar *distances_;
            size_t maxSize_;
            size_t size_;
        public:
            QueryHeap(Index *indices, Scalar *distances, const size_t maxSize)
                : indices_(indices), distances_(distances), maxSize_(maxSize),
                size_(0)
            {
            }

            void push(const Index idx, const Scalar dist)
            {
                if(full())
                    throw std::runtime_error("heap is full");
                // add new value at the end
                indices_[size_] = idx;
                distances_[size_] = dist;
                ++size_;

                // upheap
                size_t k = size_ - 1;
                while(k > 0 && distances_[(k - 1) / 2] < dist)
                {
                    size_t tmp = (k - 1) / 2;
                    distances_[k] = distances_[tmp];
                    indices_[k] = indices_[tmp];
                    k = tmp;
                }
                distances_[k] = dist;
                indices_[k] = idx;
            }

            void pop()
            {
                if(empty())
                    throw std::runtime_error("heap is empty");
                // replace first element with last
                distances_[0] = distances_[size_-1];
                indices_[0] = indices_[size_-1];
                --size_;

                // downheap
                size_t k = 0;
                Scalar dist = distances_[0];
                Index idx = indices_[0];
                while(2 * k + 1 < size_)
                {
                    size_t j = 2 * k + 1;
                    if(j + 1 < size_ && distances_[j+1] > distances_[j])
                        ++j;
                    // j references now greates child
                    if(dist >= distances_[j])
                        break;
                    distances_[k] = distances_[j];
                    indices_[k] = indices_[j];
                    k = j;
                }
                distances_[k] = dist;
                indices_[k] = idx;
            }

            Scalar front() const
            {
                if(empty())
                    throw std::runtime_error("heap is empty");

                return distances_[0];
            }

            bool full() const
            {
                return size_ >= maxSize_;
            }

            bool empty() const
            {
                return size_ == 0;
            }

            size_t size() const
            {
                return size_;
            }

            void clear()
            {
                size_ = 0;
            }

            void sort()
            {
                size_t cnt = size_;
                for(size_t i = 0; i < cnt; ++i)
                {
                    Index idx = indices_[0];
                    Scalar dist = distances_[0];
                    pop();
                    indices_[cnt - i - 1] = idx;
                    distances_[cnt - i - 1] = dist;
                }
            }
        };

        Matrix dataCopy_;
        const Matrix *data_;
        std::vector<Index> indices_;
        std::vector<Node> nodes_;

        Index bucketSize_;
        bool sorted_;
        bool compact_;
        bool balanced_;
        bool takeRoot_;
        int threads_;
        Scalar maxDist_;
        Scalar maxDistP_;

        Distance distance_;

        /** Finds the minimum and maximum values of each dimension (row) in the
         *  data matrix. Only respects the columns specified by the index
         *  vector. */
        void findDataMinMax(const Index startIdx,
            const Index length,
            Vector &mins,
            Vector &maxes) const
        {
            assert(length > 0);
            assert(startIdx >= 0);
            assert(static_cast<size_t>(startIdx + length) <= indices_.size());

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

        Index buildInnerNode(const Index startIdx,
            const Index length,
            const Vector &mins,
            const Vector &maxes)
        {
            assert(startIdx >= 0);
            assert(length > 0);
            assert(static_cast<size_t>(startIdx  + length) <= indices_.size());
            assert(maxes.size() == mins.size());

            const Matrix &data = *data_;

            // create node
            Index nodeIdx = nodes_.size();
            nodes_.push_back(Node());

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
                nodes_[nodeIdx] = Node(startIdx, length);
                return nodeIdx;
            }

            // determine split point
            Scalar splitpoint;
            // check if tree should be balanced
            if(balanced_)
            {
                // use median for splitpoint
                auto compPred =
                    [&data, splitaxis](const Index lhs, const Index rhs)
                    { return data(splitaxis, lhs) < data(splitaxis, rhs); };
                std::sort(indices_.begin() + startIdx,
                    indices_.begin() + startIdx + length,
                    compPred);

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
                    Scalar val = data(splitaxis, indices_[idx]);
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
                rightNode = buildR(rightStart, rightLength, mids, maxes);
            }

            nodes_[nodeIdx] = Node(splitaxis, splitpoint, leftNode, rightNode);
            return nodeIdx;
        }

        Index buildR(const Index startIdx,
            const Index length,
            const Vector &mins,
            const Vector &maxes)
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

        void queryLeafNode(const Node &node,
            const Scalar *queryPoint,
            const Index len,
            QueryHeap &dataHeap) const
        {
            assert(node.isLeaf());

            const Matrix &data = *data_;
            const Scalar *dataRaw = data.data();

            // go through all points in this leaf node and do brute force search
            for(Index i = 0; i < node.length; ++i)
            {
                size_t idx = node.startIdx + i;
                assert(idx < indices_.size());

                // retrieve index of the current data point
                Index dataIdx = indices_[idx];
                const Scalar *dataPoint = &dataRaw[dataIdx * len];
                Scalar dist = distance_.unrooted(queryPoint, dataPoint, len);

                // check if point is in range if max distance was set
                bool isInRange = maxDist_ <= 0 || dist < maxDistP_;
                // check if this node was an improvement if heap is already full
                bool isImprovement = !dataHeap.full() ||
                    dist < dataHeap.front();

                if(isInRange && isImprovement)
                {
                    if(dataHeap.full())
                        dataHeap.pop();
                    dataHeap.push(dataIdx, dist);
                }
            }
        }

        void queryInnerNode(const Node &node,
            const Scalar *queryPoint,
            const Index len,
            QueryHeap &dataHeap) const
        {
            assert(node.isInner());

            Scalar splitval = queryPoint[node.splitaxis];

            // check if right or left child should be visited
            bool visitRight = splitval >= node.splitpoint;
            if(visitRight)
                queryR(nodes_[node.right], queryPoint, len, dataHeap);
            else
                queryR(nodes_[node.left], queryPoint, len, dataHeap);

            // get distance to split point
            Scalar splitdist = distance_.power(splitval - node.splitpoint);

            // check if node is in range if max distance was set
            bool isInRange = maxDist_ <= 0 || splitdist < maxDistP_;
            // check if this node was an improvement if heap is already full
            bool isImprovement = !dataHeap.full() ||
                splitdist < dataHeap.front();

            if(isInRange && isImprovement)
            {
                if(visitRight)
                    queryR(nodes_[node.left], queryPoint, len, dataHeap);
                else
                    queryR(nodes_[node.right], queryPoint, len, dataHeap);
            }
        }

        void queryR(const Node &node,
            const Scalar *queryPoint,
            const Index len,
            QueryHeap &dataHeap) const
        {
            if(node.isLeaf())
                queryLeafNode(node, queryPoint, len, dataHeap);
            else
                queryInnerNode(node, queryPoint, len, dataHeap);
        }

        Index depthR(const Node &node) const
        {
            if(node.isLeaf())
                return 1;
            else
            {
                Index left = depthR(nodes_[node.left]);
                Index right = depthR(nodes_[node.right]);
                return std::max(left, right) + 1;
            }
        }

    public:

        /** Constructs an empty KDTree. */
        KDTree()
            : dataCopy_(), data_(nullptr), indices_(), nodes_(),
            bucketSize_(16), sorted_(true), compact_(true), balanced_(false),
            takeRoot_(true), threads_(0), maxDist_(0), maxDistP_(0),
            distance_()
        {

        }

        /** Constructs KDTree with the given data. This does not build the
          * the index of the tree.
          *
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
          *
          * @param bucketSize amount of points per leaf. */
        void setBucketSize(const Index bucketSize)
        {
            bucketSize_ = bucketSize;
        }

        /** Set if the points returned by the queries should be sorted
          * according to their distance to the query points.
          *
          * @param sorted sort query results */
        void setSorted(const bool sorted)
        {
            sorted_ = sorted;
        }

        /** Set if the tree should be built as balanced as possible.
          *
          * This increases build time, but decreases search time.
          * @param balanced set true to build a balanced tree */
        void setBalanced(const bool balanced)
        {
            balanced_ = balanced;
        }

        /** Set if the distances after the query should be rooted or not.
          *
          * Taking the root of the distances increases query time, but the
          * function will return true distances instead of their powered
          * versions.
          * @param takeRoot set true if root should be taken else false */
        void setTakeRoot(const bool takeRoot)
        {
            takeRoot_ = takeRoot;
        }

        /** Set if the tree should be built with compact leaf nodes.
          *
          * This increases build time, but makes leaf nodes denser (more)
          * points. Thus less visits are necessary.
          * @param compact set true ti build a tree with compact leafs */
        void setCompact(const bool compact)
        {
            compact_ = compact;
        }

        /** Set the amount of threads that should be used for building and
          * querying the tree.
          *
          * OpenMP has to be enabled for this to work.
          * @param threads amount of threads, 0 for optimal choice */
        void setThreads(const unsigned int threads)
        {
            threads_ = threads;
        }

        /** Set the maximum distance for querying the tree.
          *
          * The search will be pruned if the maximum distance is set to any
          * positive number.
          * @param maxDist maximum distance, <= 0 for no limit */
        void setMaxDistance(const Scalar maxDist)
        {
            maxDist_ = maxDist;
            maxDistP_ = distance_.power(maxDist);
        }

        /** Set the data points used for this tree.
          *
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
          *
          * Data has to be set and must be non-empty. */
        void build()
        {
            if(data_ == nullptr)
                throw std::runtime_error("cannot build KDTree; data not set");

            if(data_->size() == 0)
                throw std::runtime_error("cannot build KDTree; data is empty");

            clear();

            indices_.resize(data_->cols());
            for(size_t i = 0; i < indices_.size(); ++i)
                indices_[i] = i;

            Vector mins, maxes;
            Index startIdx = 0;
            Index length = indices_.size();

            findDataMinMax(startIdx, length, mins, maxes);

            Index root = buildR(startIdx, length, mins, maxes);
        }

        /** Queries the tree for the nearest neighbours of the given query
          * points.
          *
          * The tree has to be built before it can be queried.
          *
          * The query points have to have the same dimension as the data points
          * of the tree.
          *
          * The result matrices will be resized appropriatley.
          * Indices and distances will be set to -1 if less than knn neighbours
          * were found.
          *
          * @param queryPoints NxM matrix, M points of dimension N
          * @param knn amount of neighbours to be found
          * @param indices KNNxM matrix, indices of neighbours in the data set
          * @param distances KNNxM matrix, distance between query points and
          *        neighbours */
        void query(const Matrix &queryPoints,
            const size_t knn,
            MatrixI &indices,
            Matrix &distances) const
        {
            if(nodes_.size() == 0)
                throw std::runtime_error("cannot query KDTree; not built yet");

            if(queryPoints.rows() != dimension())
                throw std::runtime_error("cannot query KDTree; data and query points do not have same dimension");

            Index queryNum = queryPoints.cols();
            Index dim = queryPoints.rows();

            distances.setConstant(knn, queryNum, -1);
            indices.setConstant(knn, queryNum, -1);

            const Scalar *queryPointsRaw = queryPoints.data();
            Index *indicesRaw = indices.data();
            Scalar *distsRaw = distances.data();

            #pragma omp parallel for num_threads(threads_)
            for(Index i = 0; i < queryNum; ++i)
            {
                const Scalar *queryPoint = &queryPointsRaw[i * dim];
                Scalar *distPoint = &distsRaw[i * knn];
                Index *idxPoint = &indicesRaw[i * knn];

                // create heap to find nearest neighbours
                QueryHeap dataHeap(idxPoint, distPoint, knn);

                queryR(nodes_[0], queryPoint, dim, dataHeap);

                if(sorted_)
                    dataHeap.sort();

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

        /** Clears the tree. */
        void clear()
        {
            nodes_.clear();
        }

        /** Returns the amount of data points stored in the search index.
          *
          * @return number of data points */
        Index size() const
        {
            return data_ == nullptr ? 0 : data_->cols();
        }

        /** Returns the dimension of the data points in the search index.
          *
          * @return dimension of data points */
        Index dimension() const
        {
            return data_ == nullptr ? 0 : data_->rows();
        }

        /** Returns the maxximum depth of the tree.
          *
          * @return maximum depth of the tree */
        Index depth() const
        {
            return nodes_.size() == 0 ? 0 : depthR(nodes_.front());
        }
    };

    typedef KDTree<float> KDTreef;
    typedef KDTree<double> KDTreed;
}

#endif
