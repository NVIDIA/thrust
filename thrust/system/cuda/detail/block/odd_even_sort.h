/*
 *  Copyright 2008-2012 NVIDIA Corporation
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


/*! \file odd_even_sort.h
 *  \brief Block versions of Batcher's Odd-Even Merge Sort
 */

#pragma once

namespace thrust
{
namespace detail
{
namespace backend
{
namespace cuda
{
namespace block
{


/*! Block-wise implementation of Batcher's Odd-Even Merge Sort
 *  This implementation is based on Nadathur Satish's.
 */
template<typename KeyType,
         typename ValueType,
         typename StrictWeakOrdering>
  __device__ void odd_even_sort(KeyType *keys,
                                ValueType *data,
                                const unsigned int n,
                                StrictWeakOrdering comp)
{
  for(unsigned int p = blockDim.x>>1; p > 0; p >>= 1)
  {
    unsigned int q = blockDim.x>>1, r = 0, d = p;

    while(q >= p)
    {
      unsigned int j = threadIdx.x + d;

      // if j lies beyond the end of the array, we consider it "sorted" wrt i
      // regardless of whether i lies beyond the end of the array 
      if(threadIdx.x < (blockDim.x-d) && (threadIdx.x & p) == r && j < n)
      {
        KeyType xikey = keys[threadIdx.x];
        KeyType xjkey = keys[j];

        ValueType xivalue = data[threadIdx.x];
        ValueType xjvalue = data[j];

        // does xj sort before xi?
        if(comp(xjkey, xikey))
        {
          keys[threadIdx.x] = xjkey;
          keys[j] = xikey;

          data[threadIdx.x] = xjvalue;
          data[j] = xivalue;
        } // end if
      } // end if

      d = q - p;
      q >>= 1;
      r = p;

      __syncthreads();
    } // end while
  } // end for p
} // end odd_even_sort()

template<typename KeyType,
         typename ValueType,
         typename StrictWeakOrdering>
  __device__ void stable_odd_even_sort(KeyType *keys,
                                       ValueType *data,
                                       const unsigned int n,
                                       StrictWeakOrdering comp)
{
  for(unsigned int i = 0;
      i < blockDim.x>>1;
      ++i)
  {
    bool thread_is_odd = threadIdx.x & 0x1;

    // do odds first
    if(thread_is_odd && threadIdx.x + 1 < n)
    {
      KeyType xikey = keys[threadIdx.x];
      KeyType xjkey = keys[threadIdx.x + 1];

      ValueType xivalue = data[threadIdx.x];
      ValueType xjvalue = data[threadIdx.x + 1];

      // does xj sort before xi?
      if(comp(xjkey, xikey))
      {
        keys[threadIdx.x] = xjkey;
        keys[threadIdx.x + 1] = xikey;

        data[threadIdx.x] = xjvalue;
        data[threadIdx.x + 1] = xivalue;
      } // end if
    } // end if

    __syncthreads();

    // do evens second
    if(!thread_is_odd && threadIdx.x + 1 < n)
    {
      KeyType xikey = keys[threadIdx.x];
      KeyType xjkey = keys[threadIdx.x + 1];

      ValueType xivalue = data[threadIdx.x];
      ValueType xjvalue = data[threadIdx.x + 1];

      // does xj sort before xi?
      if(comp(xjkey, xikey))
      {
        keys[threadIdx.x] = xjkey;
        keys[threadIdx.x + 1] = xikey;

        data[threadIdx.x] = xjvalue;
        data[threadIdx.x + 1] = xivalue;
      } // end if
    } // end if

    __syncthreads();
  } // end for i
} // end stable_odd_even_sort()


} // end namespace block
} // end namespace cuda
} // end namespace backend
} // end namespace detail
} // end namespace thrust

