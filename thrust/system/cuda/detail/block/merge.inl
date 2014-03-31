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
                              StrictWeakOrdering comp)
{
  RandomAccessIterator1 input1 = keys_first;
  RandomAccessIterator1 input2 = keys_first + n1;

  RandomAccessIterator2 input1val = values_first;
  RandomAccessIterator2 input2val = values_first + n1;
  
  typedef typename thrust::iterator_value<RandomAccessIterator1>::type KeyType;
  typedef typename thrust::iterator_value<RandomAccessIterator2>::type ValueType;

  // XXX use uninitialized here
  KeyType inp1 = input1[context.thread_index()]; ValueType inp1val = input1val[context.thread_index()];
  KeyType inp2 = input2[context.thread_index()]; ValueType inp2val = input2val[context.thread_index()];
  
  // to merge input1 and input2, use binary search to find the rank of inp1 & inp2 in arrays input2 & input1, respectively
  // as before, the "end" variables point to one element after the last element of the arrays
  
  // start by looking through input2 for inp1's rank
  unsigned int start_1 = 0;
  
  // don't do the search if our value is beyond the end of input1
  if(context.thread_index() < n1)
  {
    start_1 = thrust::system::detail::generic::scalar::lower_bound_n(input2, n2, inp1, comp) - input2;
  } // end if
  
  // now look through input1 for inp2's rank
  unsigned int start_2 = 0;
  
  // don't do the search if our value is beyond the end of input2
  if(context.thread_index() < n2)
  {
    // upper_bound ensures that equivalent elements in the first range sort before the second
    start_2 = thrust::system::detail::generic::scalar::upper_bound_n(input1, n1, inp2, comp) - input1;
  } // end if

  context.barrier();
  
  // Write back into the right position to the input arrays; can be done in place since we read in
  // the input arrays into registers before.
  if(context.thread_index() < n1)
  {
    input1[start_1 + context.thread_index()] = inp1;
    input1val[start_1 + context.thread_index()] = inp1val;
  } // end if
  
  if(context.thread_index() < n2)
  {
    input1[start_2 + context.thread_index()] = inp2;
    input1val[start_2 + context.thread_index()] = inp2val;
  } // end if
} // end inplace_merge_by_key_n()


} // end namespace block
} // end namespace detail
} // end namespace cuda
} // end namespace system
} // end namespace thrust

