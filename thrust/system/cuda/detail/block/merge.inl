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

#include <thrust/iterator/iterator_traits.h>
#include <thrust/detail/raw_reference_cast.h>
#include <thrust/system/detail/generic/scalar/binary_search.h>

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
                              StrictWeakOrdering comp)
{
  typedef typename thrust::iterator_difference<RandomAccessIterator1>::type difference1;
  typedef typename thrust::iterator_difference<RandomAccessIterator2>::type difference2;

  difference1 n1 = last1 - first1;
  difference2 n2 = last2 - first2;

  // find the rank of each element in the other array
  difference2 rank2 = 0;
  if(context.thread_index() < n1)
  {
    RandomAccessIterator1 x = first1;
    x += context.thread_index();

    // lower_bound ensures that x sorts before any equivalent element of input2
    // this ensures stability
    rank2 = thrust::system::detail::generic::scalar::lower_bound(first2, last2, raw_reference_cast(*x), comp) - first2;
  } // end if

  difference1 rank1 = 0;
  if(context.thread_index() < n2)
  {
    RandomAccessIterator2 x = first2 + context.thread_index();

    // upper_bound ensures that x sorts before any equivalent element of input1
    // this ensures stability
    rank1 = thrust::system::detail::generic::scalar::upper_bound(first1, last1, raw_reference_cast(*x), comp) - first1;
  } // end if

  if(context.thread_index() < n1)
  {
    // scatter each element from input1
    RandomAccessIterator1 src = first1 + context.thread_index();
    RandomAccessIterator3 dst = result + context.thread_index() + rank2;

    *dst = *src;
  }

  if(context.thread_index() < n2)
  {
    // scatter each element from input2
    RandomAccessIterator2 src = first2 + context.thread_index();
    RandomAccessIterator3 dst = result + context.thread_index() + rank1;

    *dst = *src;
  }

  return result + n1 + n2;
} // end merge

} // end namespace block
} // end namespace detail
} // end namespace cuda
} // end namespace system
} // end namespace thrust

