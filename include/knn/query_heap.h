/* query_heap.h
 *
 *     Author: Fabian Meyer
 * Created On: 28 Oct 2019
 *    License: MIT
 */

#ifndef KNN_QUERY_HEAP_H_
#define KNN_QUERY_HEAP_H_

#include <stdexcept>
#include <vector>
#include "knn/matrix.h"

namespace knn
{
    /** Efficient heap structure to query nearest neighbours. */
    template<typename Scalar>
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
        { }

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
}

#endif
