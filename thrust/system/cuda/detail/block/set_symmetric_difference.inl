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
#include <thrust/system/detail/generic/scalar/binary_search.h>
#include <thrust/detail/raw_reference_cast.h>

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
         typename RandomAccessIterator4,
         typename StrictWeakOrdering>
__device__ __thrust_forceinline__
  RandomAccessIterator4 set_symmetric_difference(Context context,
                                                 RandomAccessIterator1 first1,
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

  RandomAccessIterator3 temporary1 = temporary;

  // XXX this formulation seems to compile incorrectly
  //RandomAccessIterator3 temporary2 = temporary1 + n1;
  RandomAccessIterator3 temporary2 = temporary1 + context.block_dimension();

  // for each element in range A
  // - count the number of equivalent elements in range A
  // - find the subrank of the element within the range of equivalent elements
  // - count the number of equivalent elements in range B
  // - the element should appear in the result if its subrank is gequal to the
  //   number of equivalent elements in range B

  difference2 rank_of_element_from_range1_in_range2(0);
  bool needs_output1 = false;
  if(context.thread_index() < n1)
  {
    RandomAccessIterator1 x = first1;
    x += context.thread_index();

    // count the number of previous occurrances of x in the first range
    difference1 subrank = x - thrust::system::detail::generic::scalar::lower_bound(first1,x,raw_reference_cast(*x),comp);
    
    // count the number of equivalent elements of x in the second range
    thrust::pair<RandomAccessIterator2,RandomAccessIterator2> matches = 
      thrust::system::detail::generic::scalar::equal_range(first2,last2,raw_reference_cast(*x),comp);

    difference2 num_matches = matches.second - matches.first;

    // the element should be output if its rank is gequal to the number of matches
    needs_output1 = subrank >= num_matches;
    rank_of_element_from_range1_in_range2 = matches.first - first2;
  } // end if

  difference1 rank_of_element_from_range2_in_range1(0);
  bool needs_output2 = false;
  if(context.thread_index() < n2)
  {
    RandomAccessIterator2 x = first2;
    x += context.thread_index();

    // count the number of previous occurrances of x in the first range
    difference2 subrank = x - thrust::system::detail::generic::scalar::lower_bound(first2,x,raw_reference_cast(*x),comp);
    
    // count the number of equivalent elements of x in the second range
    thrust::pair<RandomAccessIterator1,RandomAccessIterator1> matches = 
      thrust::system::detail::generic::scalar::equal_range(first1,last1,raw_reference_cast(*x),comp);

    difference1 num_matches = matches.second - matches.first;

    // the element should be output if its rank is gequal to the number of matches
    needs_output2 = subrank >= num_matches;
    rank_of_element_from_range2_in_range1 = matches.first - first1;
  } // end if

  // mark in the scratch arrays if each element needs to be output
  RandomAccessIterator3 temp1 = temporary1 + context.thread_index();
  *temp1 = needs_output1;

  RandomAccessIterator3 temp2 = temporary2 + context.thread_index();
  *temp2 = needs_output2;
  
  context.barrier();

  // scan both arrays
  block::inclusive_scan_n(context, temporary1, n1, thrust::plus<int>());
  block::inclusive_scan_n(context, temporary2, n2, thrust::plus<int>());

  // scatter elements from the first range to their place in the output
  if(needs_output1)
  {
    RandomAccessIterator1 src = first1 + context.thread_index();
    RandomAccessIterator4 dst = result;

    if(context.thread_index() > 0u)
    {
      // subtract one during indexing because the scan was inclusive
      dst += temporary1[context.thread_index()-1];
    } // end if

    if(rank_of_element_from_range1_in_range2 > difference2(0))
    {
      // subtract one during indexing because the scan was inclusive
      dst += temporary2[rank_of_element_from_range1_in_range2-1];
    } // end if

    *dst = *src;
  } // end if

  if(needs_output2)
  {
    RandomAccessIterator2 src = first2;
    src += context.thread_index();

    RandomAccessIterator4 dst = result;

    if(context.thread_index() > 0u)
    {
      // subtract one during indexing because the scan was inclusive
      dst += temporary2[context.thread_index()-1];
    } // end if

    if(rank_of_element_from_range2_in_range1 > difference1(0))
    {
      // subtract one during indexing because the scan was inclusive
      dst += temporary1[rank_of_element_from_range2_in_range1-1];
    } // end if

    *dst = *src;
  } // end if

  if(n1 > 0)
  {
    result += temporary1[n1-1];
  } // end if

  if(n2 > 0)
  {
    result += temporary2[n2-1];
  } // end if

  return result;
} // end set_symmetric_difference()

} // end namespace block
} // end namespace detail
} // end namespace cuda
} // end namespace system
} // end namespace thrust

