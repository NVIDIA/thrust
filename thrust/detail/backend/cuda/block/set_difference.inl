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
#include <thrust/detail/backend/generic/scalar/binary_search.h>
#include <thrust/detail/backend/cuda/block/inclusive_scan.h>
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
  RandomAccessIterator4 set_difference(RandomAccessIterator1 first1,
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

  if(n1 == 0) return result;

  // search for all non-matches in the second range for each element in the first
  bool needs_output = false;
  if(threadIdx.x < n1)
  {
    RandomAccessIterator1 x = first1;
    x += threadIdx.x;

    // count the number of previous occurrances of x the first range
    difference1 rank = x - thrust::detail::backend::generic::scalar::lower_bound(first1,x,dereference(x),comp);

    // count the number of equivalent elements of x in the second range
    thrust::pair<RandomAccessIterator2,RandomAccessIterator2> matches = 
      thrust::detail::backend::generic::scalar::equal_range(first2,last2,dereference(x),comp);

    difference2 num_matches = matches.second - matches.first;

    // the element needs output if its rank is gequal than the number of matches
    needs_output = rank >= num_matches;
  } // end if

  // mark whether my element needs output in the scratch array
  RandomAccessIterator3 temp = temporary;
  temp += threadIdx.x;
  dereference(temp) = needs_output;

  block::inplace_inclusive_scan_n(temporary, n1, thrust::plus<int>());

  // copy_if
  if(needs_output)
  {
    // find the index to write our element
    unsigned int output_index = 0;
    if(threadIdx.x > 0)
    {
      RandomAccessIterator3 src = temporary;
      src += threadIdx.x - 1;
      output_index = dereference(src);
    } // end if

    RandomAccessIterator1 x = first1;
    x += threadIdx.x;

    RandomAccessIterator4 dst = result;
    dst += output_index;
    dereference(dst) = dereference(x);
  } // end if

  return result + temporary[n1-1];
} // end set_difference

} // end block
} // end cuda
} // end backend
} // end detail
} // end thrust


