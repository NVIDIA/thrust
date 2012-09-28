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


#include <thrust/detail/config.h>
#include <thrust/system/cuda/detail/detail/stable_sort_by_count.h>
#include <thrust/system/cuda/detail/detail/launch_closure.h>
#include <thrust/system/cuda/detail/detail/uninitialized.h>
#include <thrust/system/cuda/detail/block/merging_sort.h>
#include <thrust/detail/minmax.h>
#include <thrust/system/cuda/detail/runtime_introspection.h>


namespace thrust
{
namespace system
{
namespace cuda
{
namespace detail
{
namespace detail
{
namespace stable_sort_by_count_detail
{


template<unsigned int block_size,
         typename RandomAccessIterator1,
         typename RandomAccessIterator2,
         typename StrictWeakOrdering,
         typename Context>
struct stable_sort_by_count_closure
{
  typedef Context context_type;

  RandomAccessIterator1 keys_first;
  RandomAccessIterator2 values_first;
  StrictWeakOrdering comp; // XXX this should probably be thrust::detail::device_function
  const unsigned int n;
  context_type context;

  stable_sort_by_count_closure(RandomAccessIterator1 keys_first,
                               RandomAccessIterator2 values_first,
                               StrictWeakOrdering comp,
                               const unsigned int n,
                               context_type context = context_type())
    : keys_first(keys_first),
      values_first(values_first),
      comp(comp),
      n(n),
      context(context)
  {}

  __device__ __thrust_forceinline__
  void operator()(void)
  {
    typedef typename iterator_value<RandomAccessIterator1>::type KeyType;
    typedef typename iterator_value<RandomAccessIterator2>::type ValueType;
  
    __shared__ uninitialized_array<KeyType,block_size>   s_keys;
    __shared__ uninitialized_array<ValueType,block_size> s_data;
  
    const unsigned int grid_size = context.grid_dimension() * context.block_dimension();
  
    // block_offset records the global index of this block's 0th thread
    unsigned int block_offset = context.block_index() * block_size;
    unsigned int i = context.thread_index() + block_offset;
  
    // advance iterators
    keys_first   += i;
    values_first += i;
  
    for(;
        block_offset < n;
        block_offset += grid_size, i += grid_size, keys_first += grid_size, values_first += grid_size)
    {
      context.barrier();
      // copy input to shared
      if(i < n)
      {
        s_keys[context.thread_index()] = *keys_first;
        s_data[context.thread_index()] = *values_first;
      } // end if
      context.barrier();
  
      // this block could be partially full
      unsigned int length = block_size;
      if(block_offset + block_size > n)
      {
        length = n - block_offset;
      } // end if
  
      // run merge_sort over the block
      block::merging_sort(context, s_keys.begin(), s_data.begin(), length, comp);
  
      // write result
      if(i < n)
      {
        *keys_first   = s_keys[context.thread_index()];
        *values_first = s_data[context.thread_index()];
      } // end if
    } // end for i
  }


  static size_t max_grid_size()
  {
    const device_properties_t& properties = device_properties();

    const unsigned int max_threads = properties.maxThreadsPerMultiProcessor * properties.multiProcessorCount;
    const unsigned int max_blocks  = properties.maxGridSize[0];
    
    return thrust::min<size_t>(max_blocks, 3 * max_threads / block_size);
  } // end max_grid_size()


  size_t grid_size() const
  {
    // compute the maximum number of blocks we can launch on this arch
    const unsigned int max_num_blocks = max_grid_size();

    // first, sort within each block
    size_t num_blocks = n / block_size;
    if(n % block_size) ++num_blocks;

    return thrust::min<size_t>(num_blocks, max_num_blocks);
  } // end grid_size()
}; // stable_sort_by_count_closure


} // end stable_sort_by_count_detail


template<unsigned int count,
         typename System,
         typename RandomAccessIterator1,
         typename RandomAccessIterator2,
         typename Compare>
void stable_sort_by_count(dispatchable<System> &,
                          RandomAccessIterator1 keys_first,
                          RandomAccessIterator1 keys_last,
                          RandomAccessIterator2 values_first,
                          Compare comp)
{
  typedef stable_sort_by_count_detail::stable_sort_by_count_closure<
    count,
    RandomAccessIterator1,
    RandomAccessIterator2,
    Compare,
    detail::statically_blocked_thread_array<count>
  > Closure;

  Closure closure(keys_first, values_first, comp, keys_last - keys_first);
 
  // do an odd-even sort per block of data
  detail::launch_closure(closure, closure.grid_size(), count);
} // end stable_sort_by_count()


} // end detail
} // end detail
} // end cuda
} // end system
} // end thrust

