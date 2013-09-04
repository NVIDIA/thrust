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
#include <thrust/system/cuda/detail/bulk/algorithm/gather.hpp>
#include <thrust/system/cuda/detail/bulk/algorithm/copy.hpp>
#include <thrust/system/cuda/detail/bulk/algorithm/merge.hpp>
#include <thrust/detail/type_traits.h>
#include <thrust/detail/minmax.h>
#include <thrust/iterator/counting_iterator.h>

BULK_NAMESPACE_PREFIX
namespace bulk
{


// XXX forward declaration for inplace_merge_adjacent_partitions below
template<std::size_t bound,
         std::size_t grainsize,
         typename RandomAccessIterator1,
         typename RandomAccessIterator2,
         typename Compare>
__forceinline__ __device__
void stable_sort_by_key(const bounded<bound,agent<grainsize> > &exec,
                        RandomAccessIterator1 keys_first, RandomAccessIterator1 keys_last,
                        RandomAccessIterator2 values_first,
                        Compare comp);


namespace detail
{
namespace stable_merge_sort_detail
{


template<std::size_t bound, std::size_t groupsize, std::size_t grainsize, typename KeyType, typename ValType, typename Compare>
__device__
typename thrust::detail::enable_if<
  bound <= groupsize * grainsize
>::type
inplace_merge_adjacent_partitions(bulk::bounded<bound,bulk::concurrent_group<bulk::agent<grainsize>, groupsize> > &g,
                                  KeyType local_keys[grainsize], ValType local_values[grainsize], void* stage_ptr, int count, int local_size, Compare comp)
{
  union stage_t
  {
    KeyType *keys;
    ValType *vals;
  };
  
  stage_t stage;
  stage.keys = reinterpret_cast<KeyType*>(stage_ptr);

  typedef typename bulk::agent<grainsize>::size_type size_type;

  size_type local_offset = grainsize * g.this_exec.index();

  // XXX this loop seems to assume that groupsize is a power of two
  //     NPOT groupsize crashes merge sort
  for(size_type num_agents_per_merge = 2; num_agents_per_merge <= groupsize; num_agents_per_merge *= 2)
  {
    // copy keys into the stage so we can dynamically index them
    bulk::copy_n(bulk::bound<grainsize>(g.this_exec), local_keys, local_size, stage.keys + local_offset);

    g.wait();

    // find the index of the first array this agent will merge
    size_type list = ~(num_agents_per_merge - 1) & g.this_exec.index();
    size_type diag = thrust::min<size_type>(count, grainsize * ((num_agents_per_merge - 1) & g.this_exec.index()));
    size_type start = grainsize * list;

    // the size of each of the two input arrays we're merging
    size_type input_size = grainsize * (num_agents_per_merge / 2);

    size_type partition_first1 = thrust::min<size_type>(count, start);
    size_type partition_first2 = thrust::min<size_type>(count, partition_first1 + input_size);
    size_type partition_last2  = thrust::min<size_type>(count, partition_first2 + input_size);

    size_type n1 = partition_first2 - partition_first1;
    size_type n2 = partition_last2  - partition_first2;

    size_type mp = bulk::merge_path(stage.keys + partition_first1, n1, stage.keys + partition_first2, n2, diag, comp);

    // each agent merges sequentially locally
    // note the source index of each merged value so that we can gather values into merged order later
    size_type gather_indices[grainsize];
    bulk::merge_by_key(bulk::bound<grainsize>(g.this_exec),
                       stage.keys + partition_first1 + mp,        stage.keys + partition_first2,
                       stage.keys + partition_first2 + diag - mp, stage.keys + partition_last2,
                       thrust::make_counting_iterator<size_type>(partition_first1 + mp),
                       thrust::make_counting_iterator<size_type>(partition_first2 + diag - mp),
                       local_keys,
                       gather_indices,
                       comp);
    
    // move values into the stage so we can index them
    bulk::copy_n(bulk::bound<grainsize>(g.this_exec), local_values, local_size, stage.vals + local_offset);

    // gather values into registers
    bulk::gather(bulk::bound<grainsize>(g.this_exec), gather_indices, gather_indices + local_size, stage.vals, local_values);

    g.wait();
  } // end for
} // end inplace_merge_adjacent_partitions()


} // end stable_merge_sort_detail


template<std::size_t bound, std::size_t groupsize, std::size_t grainsize,
         typename RandomAccessIterator1,
         typename RandomAccessIterator2,
         typename Compare>
__device__
typename thrust::detail::enable_if<
  bound <= groupsize * grainsize
>::type
stable_merge_sort_by_key(bulk::bounded<bound,bulk::concurrent_group<bulk::agent<grainsize>,groupsize> > &g,
                         RandomAccessIterator1 keys_first, RandomAccessIterator1 keys_last,
                         RandomAccessIterator2 values_first,
                         Compare comp)
{
  typedef typename thrust::iterator_value<RandomAccessIterator1>::type key_type;
  typedef typename thrust::iterator_value<RandomAccessIterator2>::type value_type;

  typedef typename bulk::agent<grainsize>::size_type size_type;

  size_type n = keys_last - keys_first;
  const size_type tile_size = groupsize * grainsize;

  size_type local_offset = grainsize * g.this_exec.index();
  size_type local_size = thrust::max<size_type>(0, thrust::min<size_type>(grainsize, n - local_offset));

#if __CUDA_ARCH__ >= 200
  union
  {
    key_type   *keys;
    value_type *values;
  } stage;

  stage.keys = static_cast<key_type*>(bulk::malloc(g, tile_size * thrust::max(sizeof(key_type), sizeof(value_type))));
#else
  __shared__ union
  {
    key_type   keys[tile_size];
    value_type values[tile_size];
  } stage;
#endif
  
  // load each agent's keys into registers
  bulk::copy_n(bulk::bound<tile_size>(g), keys_first, n, stage.keys);

  key_type local_keys[grainsize];
  bulk::copy_n(bulk::bound<grainsize>(g.this_exec), stage.keys + local_offset, local_size, local_keys);

  // load each agent's values into registers
  bulk::copy_n(bulk::bound<tile_size>(g), values_first, n, stage.values);

  value_type local_values[grainsize];
  bulk::copy_n(bulk::bound<grainsize>(g.this_exec), stage.values + local_offset, local_size, local_values);

  // each agent sorts its local partition of the array
  bulk::stable_sort_by_key(bulk::bound<grainsize>(g.this_exec), local_keys, local_keys + local_size, local_values, comp);
  
  // merge adjacent partitions together
  // avoid dynamic sizes when possible
  if(n == tile_size)
  {
    stable_merge_sort_detail::inplace_merge_adjacent_partitions(g, local_keys, local_values, stage.keys, tile_size, grainsize, comp);
  } // end if
  else
  {
    stable_merge_sort_detail::inplace_merge_adjacent_partitions(g, local_keys, local_values, stage.keys, n, local_size, comp);
  } // end else

  // store the sorted keys back to the input
  bulk::copy_n(bulk::bound<grainsize>(g.this_exec), local_keys, local_size, stage.keys + local_offset);
  g.wait();

  bulk::copy_n(bulk::bound<tile_size>(g), stage.keys, n, keys_first);
  
  // store the sorted values back to the input
  bulk::copy_n(bulk::bound<grainsize>(g.this_exec), local_values, local_size, stage.values + local_offset);
  g.wait();

  bulk::copy_n(bulk::bound<tile_size>(g), stage.values, n, values_first);

#if __CUDA_ARCH__ >= 200
  bulk::free(g, stage.keys);
#endif
} // end stable_merge_sort_by_key()


} // end detail
} // end bulk
BULK_NAMESPACE_SUFFIX

