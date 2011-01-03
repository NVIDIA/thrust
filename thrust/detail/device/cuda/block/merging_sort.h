/*
 *  Copyright 2008-2011 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */


/*! \file merging_sort.h
 *  \brief Block version of merge sort
 */

#pragma once

#include <thrust/iterator/iterator_traits.h>
#include <thrust/detail/type_traits.h>

namespace thrust
{
namespace detail
{
namespace device
{
namespace cuda
{
namespace block
{

    template<class RandomAccessIterator, class T, class Cmp, class Result>
    __device__
    // XXX nvcc 2.2 has trouble with the correct way to define this function:
    //RandomAccessIterator lower_bound(RandomAccessIterator begin, RandomAccessIterator end, const T& x, Cmp cmp)
    void lower_bound_workaround(RandomAccessIterator begin,
                                RandomAccessIterator end,
                                const T& x,
                                Cmp cmp,
                                Result &result)
    {
        while( begin<end )
        {
            RandomAccessIterator mid = begin + (end-begin)/2;

            if( cmp(*mid, x) )
              begin = mid+1;
            else
              end = mid;
        }

        result = begin;
        //return begin;
    }

    template<class RandomAccessIterator, class T, class Cmp, class Result>
    __device__
    // XXX nvcc 2.2 has trouble with the correct way to define this function:
    //RandomAccessIterator upper_bound(RandomAccessIterator begin, RandomAccessIterator end, const T& x, Cmp cmp)
    void upper_bound_workaround(RandomAccessIterator begin,
                                RandomAccessIterator end,
                                const T& x,
                                Cmp cmp,
                                Result &result)
    {
        while( begin<end )
        {
            RandomAccessIterator mid = begin + (end-begin)/2;

            if( cmp(x, *mid) )
              end = mid;
            else
              begin = mid+1;
        }

        result = begin;
        //return begin;
    }

    template<typename RandomAccessIterator1, typename RandomAccessIterator2,
             typename Cmp>
    __device__ void conditional_swap(RandomAccessIterator1 keys,
                                     RandomAccessIterator2 values,
                                     const unsigned int i,
                                     const unsigned int end,
                                     bool pred,
                                     Cmp comp)
    {
        typedef typename thrust::iterator_traits<RandomAccessIterator1>::value_type KeyType;
        typedef typename thrust::iterator_traits<RandomAccessIterator2>::value_type ValueType;

        if(pred && i+1<end)
        {
            KeyType xi = keys[i];
            KeyType xj = keys[i+1];

            // swap if xj sorts before xi
            if(comp( xj, xi))
            {
                ValueType yi;
                yi = values[i];
                ValueType yj;
                yj = values[i+1];

                keys[i]     = xj;
                keys[i+1]   = xi;
                values[i]   = yj;
                values[i+1] = yi;
            }
        }
    }

    template<typename RandomAccessIterator1, typename RandomAccessIterator2,
             typename Cmp>
    __device__ void transposition_sort(RandomAccessIterator1 keys,
                                       RandomAccessIterator2 values,
                                       const unsigned int i,
                                       const unsigned int end,
                                       const unsigned int size,
                                       Cmp comp)
    {
        const bool is_odd = i&0x1;

        for(unsigned int round=size/2; round>0; --round)
        {
            // ODDS
            conditional_swap(keys, values, i, end, is_odd, comp);
            __syncthreads();

            // EVENS
            conditional_swap(keys, values, i, end, !is_odd, comp);
            __syncthreads();
        }
    }

template<typename RandomAccessIterator1,
         typename RandomAccessIterator2,
         typename StrictWeakOrdering>
__device__ void merge(RandomAccessIterator1 keys, 
                      RandomAccessIterator2 data,
                      const unsigned int i,
                      const unsigned int n,
                      unsigned int begin,
                      unsigned int end,
                      unsigned int h,
                      StrictWeakOrdering cmp)
{

    // INVARIANT: Every element i resides within a sequence [begin,end)
    //            of length h which is already sorted
    while( h<n )
    {
        h *= 2;

        unsigned int new_begin = i&(~(h-1)),
                     new_end   = min(n,new_begin+h);

        typedef typename thrust::iterator_traits<RandomAccessIterator1>::value_type KeyType;
        typedef typename thrust::iterator_traits<RandomAccessIterator2>::value_type ValueType;

        KeyType xi;
        ValueType yi;

        unsigned int rank = i - begin;

        // prevent out-of-bounds access
        if( i < new_end )
        {
          xi = keys[i];

          if( begin==new_begin )  // in the left side of merging pair
          {
              //rank += lower_bound(keys+end, keys+new_end, xi, cmp) - (keys+end);
              
              RandomAccessIterator1 result;
              lower_bound_workaround(keys+end, keys+new_end, xi, cmp, result);
              rank += (result - (keys+end));
          }
          else                    // in the right side of merging pair
          {
              //rank += upper_bound(keys+new_begin, keys+begin, xi, cmp) - (keys+new_begin);

              RandomAccessIterator1 result;
              upper_bound_workaround(keys+new_begin, keys+begin, xi, cmp, result);
              rank += (result - (keys+new_begin));
          }

          yi = data[i];
        }

        __syncthreads();
        if( i < new_end )
        {
          keys[new_begin+rank] = xi;
          data[new_begin+rank] = yi;
        }
        __syncthreads();

        begin = new_begin;
        end   = new_end;
    }
}


/*! Block-wise implementation of merge sort.
 *  It provides the same external interface as odd_even_sort.
 */
template<typename RandomAccessIterator1,
         typename RandomAccessIterator2,
         typename StrictWeakOrdering>
__device__ void merging_sort(RandomAccessIterator1 keys,
                             RandomAccessIterator2 data,
                             const unsigned int n,
                             StrictWeakOrdering comp)
{
    // Phase 1: Sort subsequences of length 32 using odd-even
    //          transposition sort.  The code below assumes that h is a
    //          power of 2.  Empirically, 32 delivers best results,
    //          which is not surprising since that's the warp width.
    unsigned int i = threadIdx.x;
    unsigned int h = 32;
    unsigned int begin=i&(~(h-1)),  end=min(n,begin+h);

    transposition_sort(keys, data, i, end, h, comp);

    // Phase 2: Apply merge tree to produce final sorted results
    merge(keys, data, i, n, begin, end, h, comp);
} // end merging_sort()

} // end namespace block
} // end namespace cuda
} // end namespace device
} // end namespace detail
} // end namespace thrust

