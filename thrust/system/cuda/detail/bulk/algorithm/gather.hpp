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

#include <thrust/system/cuda/detail/bulk/detail/config.hpp>
#include <thrust/system/cuda/detail/bulk/algorithm/copy.hpp>
#include <thrust/system/cuda/detail/bulk/execution_policy.hpp>
#include <thrust/iterator/permutation_iterator.h>


BULK_NAMESPACE_PREFIX
namespace bulk
{


// XXX eliminate me!
template<std::size_t bound,
         std::size_t grainsize,
         typename RandomAccessIterator1,
         typename RandomAccessIterator2,
         typename RandomAccessIterator3>
__forceinline__ __device__
RandomAccessIterator3 gather(const bounded<bound,agent<grainsize> > &,
                             RandomAccessIterator1 map_first,
                             RandomAccessIterator1 map_last,
                             RandomAccessIterator2 input_first,
                             RandomAccessIterator3 result)
{
  typedef typename bulk::bounded<bound,bulk::agent<grainsize> >::size_type size_type;

  size_type n = map_last - map_first;

  if(bound <= n)
  {
    for(size_type i = 0; i < bound; ++i)
    {
      result[i] = input_first[map_first[i]];
    }
  }
  else
  {
    for(size_type i = 0; i < bound; ++i)
    {
      if(i < n)
      {
        result[i] = input_first[map_first[i]];
      }
    }
  }

  return result + n;
} // end scatter_if()


template<typename ExecutionGroup, typename RandomAccessIterator1, typename RandomAccessIterator2, typename RandomAccessIterator3>
__forceinline__ __device__
RandomAccessIterator3 gather(ExecutionGroup &g,
                             RandomAccessIterator1 map_first,
                             RandomAccessIterator1 map_last,
                             RandomAccessIterator2 input_first,
                             RandomAccessIterator3 result)
{
  return bulk::copy_n(g,
                      thrust::make_permutation_iterator(input_first, map_first),
                      map_last - map_first,
                      result);
} // end gather()


} // end bulk
BULK_NAMESPACE_SUFFIX

