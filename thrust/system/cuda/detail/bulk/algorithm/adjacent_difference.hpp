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
#include <thrust/system/cuda/detail/bulk/execution_policy.hpp>

BULK_NAMESPACE_PREFIX
namespace bulk
{


template<std::size_t grainsize,
         typename RandomAccessIterator1,
         typename RandomAccessIterator2,
         typename T,
         typename BinaryOperation>
__device__
RandomAccessIterator2 adjacent_difference(bulk::agent<grainsize> &exec,
                                          RandomAccessIterator1 first, RandomAccessIterator1 last,
                                          RandomAccessIterator2 result,
                                          T init,
                                          BinaryOperation binary_op)
{
  for(; first != last; ++first, ++result)
  {
    T temp = *first;
    *result = binary_op(temp, init);
    init = temp;
  } // end result

  return result;
} // end adjacent_difference()


template<std::size_t groupsize,
         std::size_t grainsize_,
         typename RandomAccessIterator1,
         typename RandomAccessIterator2,
         typename T,
         typename BinaryOperation>
__device__
RandomAccessIterator2 adjacent_difference(bulk::concurrent_group<bulk::agent<grainsize_>,groupsize> &g,
                                          RandomAccessIterator1 first, RandomAccessIterator1 last,
                                          RandomAccessIterator2 result,
                                          T init,
                                          BinaryOperation binary_op)
{
  // XXX this implementation allows first to be equal to result
  //     when the input and output do not overlap, we can avoid the need for next_init
  //     and the barriers
  
  typedef typename bulk::concurrent_group<bulk::agent<grainsize_>,groupsize>::size_type size_type;

  RandomAccessIterator2 return_me = result + (last - first);

  const size_type grainsize = g.this_exec.grainsize();
  const size_type tile_size = g.size() * grainsize;

  // set the first iteration's init
  RandomAccessIterator1 first_init = first + grainsize * g.this_exec.index() - 1;
  if(first <= first_init && first_init < last)
  {
    init = *first_init;
  }
  
  g.wait();

  for(; first < last; first += tile_size, result += tile_size)
  {
    size_type local_offset = grainsize * g.this_exec.index();
    size_type local_size = thrust::max(0, thrust::min<size_type>(grainsize, last - (first + local_offset)));

    // get the init for the next iteration
    T next_init = (first + local_offset + tile_size - 1 < last) ? first[tile_size-1] : init;

    g.wait();

    // consume grainsize elements
    bulk::adjacent_difference(g.this_exec,
                              first + local_offset,
                              first + local_offset + local_size,
                              result + local_offset,
                              init,
                              binary_op);

    init = next_init;
  }

  g.wait();

  return return_me;
} // end adjacent_difference()


template<std::size_t groupsize,
         std::size_t grainsize,
         typename RandomAccessIterator1,
         typename RandomAccessIterator2,
         typename BinaryOperation>
__device__
RandomAccessIterator2 adjacent_difference(bulk::concurrent_group<bulk::agent<grainsize>,groupsize> &g,
                                          RandomAccessIterator1 first, RandomAccessIterator1 last,
                                          RandomAccessIterator2 result,
                                          BinaryOperation binary_op)
{
  if(first < last)
  {
    typename thrust::iterator_value<RandomAccessIterator1>::type init = *first;

    // we need to wait because first may be the same as result
    g.wait();

    if(g.this_exec.index() == 0)
    {
      *result = init;
    }

    result = bulk::adjacent_difference(g, first + 1, last, result + 1, init, binary_op); 
  } // end if

  return result;
} // end adjacent_difference()


} // end bulk
BULK_NAMESPACE_SUFFIX

