/*
 *  Copyright 2008-2013 NVIDIA Corporation
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

#pragma once

#include <thrust/detail/config.h>

namespace thrust
{
namespace system
{
namespace cuda
{
namespace detail
{
namespace block
{

template<typename Context,
         typename RandomAccessIterator1,
         typename RandomAccessIterator2,
         typename RandomAccessIterator3,
         typename StrictWeakOrdering>
__device__ __thrust_forceinline__
  RandomAccessIterator3 merge(Context context,
                              RandomAccessIterator1 first1,
                              RandomAccessIterator1 last1,
                              RandomAccessIterator2 first2,
                              RandomAccessIterator2 last2,
                              RandomAccessIterator3 result,
                              StrictWeakOrdering comp);

// XXX assumes that context.block_dimension() <= n1 and
//                  context.block_dimension() <= n2
// This algorithm is analogous to inplace_merge
// but instead of working on the ranges
// [first, middle) and [middle, last)
// it works on the ranges
// [first, first + n1) and [first + n1, first + n1 + n2)
template<typename Context,
         typename RandomAccessIterator1,
         typename RandomAccessIterator2,
         typename Size1,
         typename Size2,
         typename StrictWeakOrdering>
__device__ __thrust_forceinline__
  void inplace_merge_by_key_n(Context context,
                              RandomAccessIterator1 keys_first,
                              RandomAccessIterator2 values_first,
                              Size1 n1,
                              Size2 n2,
                              StrictWeakOrdering comp);

} // end namespace block
} // end namespace detail
} // end namespace cuda
} // end namespace system
} // end namespace thrust

#include <thrust/system/cuda/detail/block/merge.inl>

