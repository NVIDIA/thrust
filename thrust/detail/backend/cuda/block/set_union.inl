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
#include <thrust/detail/backend/cuda/block/merge.h>
#include <thrust/detail/backend/generic/scalar/binary_search.h>
#include <thrust/detail/backend/dereference.h>
#include <thrust/tuple.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/counting_iterator.h>

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


template<typename RandomAccessIterator1,
         typename RandomAccessIterator2,
         typename RandomAccessIterator3,
         typename RandomAccessIterator4,
         typename StrictWeakOrdering>
__device__ __forceinline__
  RandomAccessIterator4 set_union(RandomAccessIterator1 first1,
                                  RandomAccessIterator1 last1,
                                  RandomAccessIterator2 first2,
                                  RandomAccessIterator2 last2,
                                  RandomAccessIterator3 temporary,
                                  RandomAccessIterator4 result,
                                  StrictWeakOrdering comp)
{
  typedef typename thrust::iterator_difference<RandomAccessIterator1>::type difference1;
  typedef typename thrust::iterator_difference<RandomAccessIterator2>::type difference2;

  difference1 n1 = last1 - first1;
  difference2 n2 = last2 - first2;

  if(n1 == 0 && n2 == 0) return result;

  // for each element in the second range
  // count the number of matches in the first range
  // initialize rank1 to an impossible result
  difference1 rank1 = difference1(-1);

  if(threadIdx.x < n2)
  {
    RandomAccessIterator2 x = first2;
    x += threadIdx.x;

    // count the number of previous occurrances of x in the second range
    difference2 sub_rank2 = x - thrust::detail::backend::generic::scalar::lower_bound(first2,x,dereference(x),comp);

    // count the number of equivalent elements of x in the first range
    thrust::pair<RandomAccessIterator1,RandomAccessIterator1> matches = 
      thrust::detail::backend::generic::scalar::equal_range(first1,last1,dereference(x),comp);

    difference2 num_matches = matches.second - matches.first;

    // the element should be output if its rank is gequal to the number of matches
    if(sub_rank2 >= num_matches)
    {
      rank1 = (matches.second - first1);
    } // end if
  } // end if

  // for the second range of elements,
  // mark in the scratch array if we need
  // to be output
  RandomAccessIterator3 temp = temporary;
  temp += threadIdx.x;
  dereference(temp) = (rank1 >= difference1(0)) ? 1 : 0;

  __syncthreads();

  // inclusive scan the scratch array
  block::inplace_inclusive_scan_n(temporary, n2, thrust::plus<int>());

  // find the rank of each element in the first range in the second range
  // modulo the fact that some elements of the second range will not appear in the output
  // these irrelevant elements should be skipped when computing ranks
  // note that every element of the first range gets output
  difference2 rank2 = 0;
  if(threadIdx.x < n1)
  {
    RandomAccessIterator1 x = first1;
    x += threadIdx.x;

    // lower_bound ensures that x sorts before any equivalent element of input2
    // this ensures stability
    rank2 = thrust::detail::backend::generic::scalar::lower_bound(first2, last2, dereference(x), comp) - first2;

    // since the temporary array contains, for each element inclusive,
    // the number of previous active elements from the second range,
    // we can compute the final rank2 simply by using the current value
    // of rank2 as an index into the temporary array
    if(rank2 > difference2(0))
    {
      // subtract one during the index because the scan was inclusive
      rank2 = temporary[rank2-1];
    } // end if
  } // end if

  // scatter elements from the first range to their place in the output
  if(threadIdx.x < n1)
  {
    RandomAccessIterator1 src = first1;
    src += threadIdx.x;

    RandomAccessIterator4 dst = result;
    dst += threadIdx.x + rank2;

    dereference(dst) = dereference(src);
  } // end if

  // scatter elements from the second range
  if(threadIdx.x < n2 && (rank1 >= difference1(0)))
  {
    // find the index to write our element
    unsigned int num_elements_from_second_range_before_me = 0;
    if(threadIdx.x > 0)
    {
      RandomAccessIterator3 src = temporary;
      src += threadIdx.x - 1;
      num_elements_from_second_range_before_me = dereference(src);
    } // end if

    RandomAccessIterator2 src = first2;
    src += threadIdx.x;

    RandomAccessIterator4 dst = result;
    dst += num_elements_from_second_range_before_me + rank1;

    dereference(dst) = dereference(src);
  } // end if

  // finding the size of the result:
  // range 1: all of range 1 gets output, so add n1
  // range 2: the temporary array contains, for each element inclusive,
  //          the cumulative number of elements from the second range to output 
  //          add the cumulative sum at the final element of the second range
  //          but carefully handle the case where the range is empty
  // XXX we could handle empty input as a special case at the beginning of the function

  return result + n1 + (n2 ? temporary[n2-1] : 0);
} // end set_union()


} // end block
} // end cuda
} // end backend
} // end detail
} // end thrust

