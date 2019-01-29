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
#include <deque>
#include <algorithm>

namespace kdt
{
    /** Functor for manhatten distance. */
    template <typename Scalar>
    struct ManhattenDistance
    {
        typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;
        typedef typename Vector::Index Index;

        Scalar unrooted(const Vector &vecA, const Vector &vecB) const
        {
            assert(vecA.size() == vecB.size());

            Scalar result = 0.0;
            for(Index i = 0; i < vecA.size(); ++i)
            {
                Scalar diff = vecA(i) - vecB(i);
                result += power(diff);
            }
            return result;
        }

        Scalar rooted(const Vector &vecA, const Vector &vecB) const
        {
            return unrooted(vecA, vecB);
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

    /** Functor for euclidean distance. */
    template <typename Scalar>
    struct EuclideanDistance
    {
        typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;
        typedef typename Vector::Index Index;

        Scalar unrooted(const Vector &vecA, const Vector &vecB) const
        {
            assert(vecA.size() == vecB.size());

            Scalar result = 0.0;
            for(Index i = 0; i < vecA.size(); ++i)
            {
                Scalar diff = vecA(i) - vecB(i);
                result += power(diff);
            }
            return result;
        }

        Scalar rooted(const Vector &vecA, const Vector &vecB) const
        {
            return root(unrooted(vecA, vecB));
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

    /** Functor for minkowski distance. */
    template <typename Scalar, int P>
    struct MinkowskiDistance
    {
        typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;
        typedef typename Vector::Index Index;

        Scalar unrooted(const Vector &vecA, const Vector &vecB) const
        {
            assert(vecA.size() == vecB.size());

            Scalar result = 0.0;
            for(Index i = 0; i < vecA.size(); ++i)
            {
                Scalar diff = vecA(i) - vecB(i);
                result += power(diff);
            }
            return result;
        }

        Scalar rooted(const Vector &vecA, const Vector &vecB) const
        {
            return root(unrooted(vecA, vecB));
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

    template<typename T,
        typename Compare=std::less<T>,
        typename Container=std::vector<T>>
    class Heap
    {
    public:
        typedef typename Container::reference reference;
        typedef typename Container::const_reference const_reference;
        typedef typename Container::iterator iterator;
        typedef typename Container::const_iterator const_iterator;
        typedef typename Container::size_type size_type;
    private:
        Container data_;
        size_type maxSize_;
        Compare comp_;
    public:
        Heap()
        : data_(), maxSize_(0), comp_()
        {

        }

        Heap(const size_type size)
        : data_(), maxSize_(size), comp_()
        {
            data_.reserve(maxSize_);
        }

        void push(const T &val)
        {
            if(full())
                throw std::runtime_error("heap is full");
            data_.push_back(val);
            std::push_heap(data_.begin(), data_.end(), comp_);
        }

        void pop()
        {
            std::pop_heap(data_.begin(), data_.end(), comp_);
            data_.pop_back();
        }

        reference front()
        {
            return data_.front();
        }

        const_reference front() const
        {
            return data_.front();
        }

        bool full() const
        {
            return maxSize_ > 0 && data_.size() >= maxSize_;
        }

        bool empty() const
        {
            return data_.empty();
        }

        size_type size() const
        {
            return data_.size();
        }

        void clear()
        {
            data_.clear();
        }

        void sort()
        {
            std::sort_heap(data_.begin(), data_.end(), comp_);
        }

        reference operator[](const size_type idx)
        {
            return data_[idx];
        }

        const_reference operator[](const size_type idx) const
        {
            return data_[idx];
        }

        iterator begin()
        {
            return data_.begin();
        }

        const_iterator begin() const
        {
            return data_.begin();
        }

        iterator end()
        {
            return data_.end();
        }

        const_iterator end() const
        {
            return data_.end();
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
        typedef Eigen::Matrix<Scalar, 1, 1> Vector1;
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

        struct QueryHeapElement
        {
            Index idx;
            Scalar distance;

            QueryHeapElement(const Index idx, const Scalar dist)
            :idx(idx), distance(dist) { }

            bool operator<(const QueryHeapElement &rhs) const
            {
                return distance < rhs.distance;
            }
        };

        typedef Heap<QueryHeapElement> QueryHeap;

        Matrix dataCopy_;
        const Matrix *data_;
        std::vector<Index> indices_;
        std::vector<Node> nodes_;

        Index bucketSize_;
        bool sorted_;
        bool compact_;
        bool balanced_;
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
            const Vector &queryPoint,
            QueryHeap &dataHeap) const
        {
            assert(node.isLeaf());

            const Matrix &data = *data_;

            // go through all points in this leaf node and do brute force search
            for(Index i = 0; i < node.length; ++i)
            {
                size_t idx = node.startIdx + i;
                assert(idx < indices_.size());

                // retrieve index of the current data point
                Index dataIdx = indices_[idx];
                Scalar dist = distance_.unrooted(queryPoint, data.col(dataIdx));

                // check if point is in range if max distance was set
                bool isInRange = maxDist_ <= 0 || dist < maxDistP_;
                // check if this node was an improvement if heap is already full
                bool isImprovement = !dataHeap.full() ||
                    dist < dataHeap.front().distance;

                if(isInRange && isImprovement)
                {
                    if(dataHeap.full())
                        dataHeap.pop();
                    dataHeap.push({dataIdx, dist});
                }
            }
        }

        void queryInnerNode(const Node &node,
            const Vector &queryPoint,
            QueryHeap &dataHeap) const
        {
            assert(node.isInner());

            Scalar splitval = queryPoint(node.splitaxis);

            // check if right or left child should be visited
            bool visitRight = splitval >= node.splitpoint;
            if(visitRight)
                queryR(nodes_[node.right], queryPoint, dataHeap);
            else
                queryR(nodes_[node.left], queryPoint, dataHeap);

            // get distance to split point
            Scalar splitdist = distance_.unrooted(Vector1(splitval),
                Vector1(node.splitpoint));

            // check if node is in range if max distance was set
            bool isInRange = maxDist_ <= 0 || splitdist < maxDistP_;
            // check if this node was an improvement if heap is already full
            bool isImprovement = !dataHeap.full() ||
                splitdist < dataHeap.front().distance;

            if(isInRange && isImprovement)
            {
                if(visitRight)
                    queryR(nodes_[node.left], queryPoint, dataHeap);
                else
                    queryR(nodes_[node.right], queryPoint, dataHeap);
            }
        }

        void queryR(const Node &node,
            const Vector &queryPoint,
            QueryHeap &dataHeap) const
        {
            if(node.isLeaf())
                queryLeafNode(node, queryPoint, dataHeap);
            else
                queryInnerNode(node, queryPoint, dataHeap);
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
            threads_(0), maxDist_(0), maxDistP_(0), distance_()
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

        /** Set if the tree should be built as balanced as possible.
          * This increases build time, but decreases search time.
          * @param balanced set true to build a balanced tree */
        void setBalanced(const bool balanced)
        {
            balanced_ = balanced;
        }

        /** Set if the tree should be built with compact leaf nodes.
          * This increases build time, but makes leaf nodes denser (more)
          * points. Thus less visits are necessary.
          * @param compact set true ti build a tree with compact leafs */
        void setCompact(const bool compact)
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
            maxDistP_ = distance_.power(maxDist);
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
            if(nodes_.size() == 0)
                throw std::runtime_error("cannot query KDTree; not built yet");

            if(queryPoints.rows() != dimension())
                throw std::runtime_error("cannot query KDTree; data and query points do not have same dimension");

            distances.setConstant(knn, queryPoints.cols(), -1);
            indices.setConstant(knn, queryPoints.cols(), -1);

            #pragma omp parallel for num_threads(threads_)
            for(Index i = 0; i < queryPoints.cols(); ++i)
            {
                // create heap to find nearest neighbours
                QueryHeap dataHeap(knn);

                queryR(nodes_[0], queryPoints.col(i), dataHeap);

                if(sorted_)
                    dataHeap.sort();

                for(size_t j = 0; j < dataHeap.size(); ++j)
                {
                    indices(j, i) = dataHeap[j].idx;
                    distances(j, i) = distance_.root(dataHeap[j].distance);
                }
            }
        }

        void clear()
        {
            nodes_.clear();
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
            return nodes_.size() == 0 ? 0 : depthR(nodes_.front());
        }
    };

    typedef KDTree<float> KDTreef;
    typedef KDTree<double> KDTreed;
}

#endif
