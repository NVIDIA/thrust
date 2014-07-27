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
#include <thrust/system/cuda/detail/bulk/detail/alignment.hpp>
#include <thrust/system/cuda/detail/bulk/detail/cuda_task.hpp>
#include <thrust/system/cuda/detail/bulk/detail/cuda_launcher/runtime_introspection.hpp>
#include <thrust/system/cuda/detail/bulk/detail/cuda_launcher/triple_chevron_launcher.hpp>
#include <thrust/system/cuda/detail/bulk/detail/cuda_launcher/cuda_launch_config.hpp>
#include <thrust/system/cuda/detail/bulk/detail/synchronize.hpp>
#include <thrust/detail/minmax.h>
#include <thrust/pair.h>


// It's not possible to launch a CUDA kernel unless __BULK_HAS_CUDART__
// is 1, so we'd like to just hide all this code when that macro is 0.
// Unfortunately, we can't actually modulate kernel launches based on that macro
// because that will hide __global__ function template instantiations from critical
// nvcc compilation phases. This means that nvcc won't actually place the kernel in the
// binary and we'll get an undefined __global__ function error at runtime.
// So we allow the user to unconditionally create instances of classes like cuda_launcher
// even though the member function .launch(...) isn't always available.


BULK_NAMESPACE_PREFIX
namespace bulk
{
namespace detail
{


// XXX instead of passing block_size_ as a template parameter to cuda_launcher_base,
//     find a way to fish it out of ExecutionGroup
template<unsigned int block_size_, typename ExecutionGroup, typename Closure>
struct cuda_launcher_base
  : public triple_chevron_launcher<
      block_size_,
      cuda_task<ExecutionGroup,Closure>
    >
{
  typedef triple_chevron_launcher<block_size_, cuda_task<ExecutionGroup,Closure> > super_t;
  typedef typename super_t::task_type                                              task_type;
  typedef typename ExecutionGroup::size_type                                       size_type;


  __host__ __device__
  cuda_launcher_base()
    : m_device_properties(bulk::detail::device_properties())
  {}


  __host__ __device__
  void launch(size_type num_blocks, size_type block_size, size_type num_dynamic_smem_bytes, cudaStream_t stream, task_type task)
  {
    if(num_blocks > 0)
    {
      super_t::launch(num_blocks, block_size, num_dynamic_smem_bytes, stream, task);

      bulk::detail::synchronize_if_enabled("bulk_kernel_by_value");
    } // end if
  } // end launch()


  __host__ __device__
  static size_type max_active_blocks_per_multiprocessor(const device_properties_t &props,
                                                        const function_attributes_t &attr,
                                                        size_type num_threads_per_block,
                                                        size_type num_smem_bytes_per_block)
  {
    return bulk::detail::cuda_launch_config_detail::max_active_blocks_per_multiprocessor(props, attr, num_threads_per_block, num_smem_bytes_per_block);
  } // end max_active_blocks_per_multiprocessor()


  // returns
  // 1. maximum number of additional dynamic smem bytes that would not lower the kernel's occupancy
  // 2. kernel occupancy
  __host__ __device__
  static thrust::pair<size_type,size_type> dynamic_smem_occupancy_limit(const device_properties_t &props, const function_attributes_t &attr, size_type num_threads_per_block, size_type num_smem_bytes_per_block)
  {
    // figure out the kernel's occupancy with 0 bytes of dynamic smem
    size_type occupancy = max_active_blocks_per_multiprocessor(props, attr, num_threads_per_block, num_smem_bytes_per_block);

    // if the kernel footprint is already too large, return (0,0)
    if(occupancy < 1) return thrust::make_pair(0,0);

    return thrust::make_pair(bulk::detail::proportional_smem_allocation(props, attr, occupancy), occupancy);
  } // end smem_occupancy_limit()


  __host__ __device__
  size_type choose_heap_size(const device_properties_t &props, size_type group_size, size_type requested_size)
  {
    function_attributes_t attr = bulk::detail::function_attributes(super_t::global_function_pointer());

    // if the kernel's ptx version is < 200, we return 0 because there is no heap
    // if the user requested no heap, give him no heap
    if(attr.ptxVersion < 20 || requested_size == 0)
    {
      return 0;
    } // end if

    // how much smem could we allocate without reducing occupancy?
    size_type result = 0, occupancy = 0;
    thrust::tie(result,occupancy) = dynamic_smem_occupancy_limit(props, attr, group_size, 0);

    // let's try to increase the heap size, but only if the following are true:
    // 1. the user asked for more heap than the default
    // 2. there's occupancy to spare
    if(requested_size != use_default && requested_size > result && occupancy > 1)
    {
      // first add in a few bytes to the request for the heap data structure
      requested_size += 48;

      // are we asking for more heap than is available at this occupancy level?
      if(requested_size > result)
      {
        // the request overflows occupancy, so we might as well bump it to the next level
        size_type next_level_result = 0, next_level_occupancy = 0;
        thrust::tie(next_level_result, next_level_occupancy) = dynamic_smem_occupancy_limit(props, attr, group_size, requested_size);

        // if we didn't completely overflow things, use this new heap size
        // otherwise, the heap remains the default size
        if(next_level_occupancy > 0) result = next_level_result;
      } // end else
    } // end i

    return result;
  } // end choose_smem_size()


  __host__ __device__
  size_type choose_group_size(size_type requested_size)
  {
    size_type result = requested_size;

    if(result == use_default)
    {
      bulk::detail::function_attributes_t attr = bulk::detail::function_attributes(super_t::global_function_pointer());

      return bulk::detail::block_size_with_maximum_potential_occupancy(attr, device_properties());
    } // end if

    return result;
  } // end choose_group_size()


  __host__ __device__
  size_type choose_subscription(size_type block_size)
  {
    // given no other info, this is a reasonable guess
    return block_size > 0 ? device_properties().maxThreadsPerMultiProcessor / block_size : 0;
  }


  __host__ __device__
  size_type choose_num_groups(size_type requested_num_groups, size_type group_size)
  {
    size_type result = requested_num_groups;

    if(result == use_default)
    {
      // given no other info, a reasonable number of groups
      // would simply occupy the machine as well as possible
      size_type subscription = choose_subscription(group_size);

      result = thrust::min<size_type>(subscription * device_properties().multiProcessorCount, max_physical_grid_size());
    } // end if

    return result;
  } // end choose_num_groups()


  __host__ __device__
  size_type max_physical_grid_size()
  {
    // get the limit of the actual device
    int actual_limit = device_properties().maxGridSize[0];

    // get the limit of the PTX version of the kernel
    int ptx_version = bulk::detail::function_attributes(super_t::global_function_pointer()).ptxVersion;

    int ptx_limit = 0;

    // from table 9 of the CUDA C Programming Guide
    if(ptx_version < 30)
    {
      ptx_limit = 65535;
    } // end if
    else
    {
      ptx_limit = (1u << 31) - 1;
    } // end else

    return thrust::min<size_type>(actual_limit, ptx_limit);
  } // end max_physical_grid_size()


  __host__ __device__
  const device_properties_t &device_properties() const
  {
    return m_device_properties;
  }


  device_properties_t m_device_properties;
}; // end cuda_launcher_base


template<typename ExecutionGroup, typename Closure> struct cuda_launcher;


template<std::size_t gridsize, std::size_t blocksize, std::size_t grainsize, typename Closure>
struct cuda_launcher<
  parallel_group<
    concurrent_group<
      agent<grainsize>,
      blocksize
    >,
    gridsize
  >,
  Closure
>
  : public cuda_launcher_base<blocksize, typename cuda_grid<gridsize,blocksize,grainsize>::type,Closure>
{
  typedef cuda_launcher_base<blocksize, typename cuda_grid<gridsize,blocksize,grainsize>::type,Closure> super_t;
  typedef typename super_t::size_type size_type;

  typedef typename cuda_grid<gridsize,blocksize,grainsize>::type grid_type;
  typedef typename grid_type::agent_type                         block_type;
  typedef typename block_type::agent_type                        thread_type;

  typedef typename super_t::task_type task_type;

  // launch(...) requires CUDA launch capability
  __host__ __device__
  void launch(grid_type request, Closure c, cudaStream_t stream)
  {
    grid_type g = configure(request);

    size_type num_blocks = g.size();
    size_type block_size = g.this_exec.size();

    if(num_blocks > 0 && block_size > 0)
    {
      size_type heap_size  = g.this_exec.heap_size();

      size_type max_physical_grid_size = super_t::max_physical_grid_size();

      // launch multiple grids in order to accomodate potentially too large grid size requests
      // XXX these will all go in sequential order in the same stream, even though they are logically
      //     parallel
      if(block_size > 0)
      {
        size_type num_remaining_physical_blocks = num_blocks;
        for(size_type block_offset = 0;
            block_offset < num_blocks;
            block_offset += max_physical_grid_size)
        {
          task_type task(g, c, block_offset);

          size_type num_physical_blocks = thrust::min<size_type>(num_remaining_physical_blocks, max_physical_grid_size);

          super_t::launch(num_physical_blocks, block_size, heap_size, stream, task);

          num_remaining_physical_blocks -= num_physical_blocks;
        } // end for block_offset
      } // end if
    } // end if
  } // end go()

  __host__ __device__
  grid_type configure(grid_type g)
  {
    size_type block_size = super_t::choose_group_size(g.this_exec.size());
    size_type heap_size  = super_t::choose_heap_size(device_properties(), block_size, g.this_exec.heap_size());
    size_type num_blocks = g.size();

    return make_grid<grid_type>(num_blocks, make_block<block_type>(block_size, heap_size));
  } // end configure()

  // chooses a number of groups and a group size
  __host__ __device__
  thrust::pair<size_type, size_type> choose_sizes(size_type requested_num_groups, size_type requested_group_size)
  {
    // if a static blocksize is set, we ignore the requested group size
    // and just use the static value
    size_type group_size = blocksize;
    if(group_size == 0)
    {
      group_size = super_t::choose_group_size(requested_group_size);
    } // end if

    // if a static gridsize is set, we ignore the requested group size
    // and just use the static value
    size_type num_groups = gridsize;
    if(num_groups == 0)
    {
      num_groups = super_t::choose_num_groups(requested_num_groups, group_size);
    } // end if

    return thrust::make_pair(num_groups, group_size);
  } // end choose_sizes()
}; // end cuda_launcher


template<std::size_t blocksize, std::size_t grainsize, typename Closure>
struct cuda_launcher<
  concurrent_group<
    agent<grainsize>,
    blocksize
  >,
  Closure
>
  : public cuda_launcher_base<blocksize,concurrent_group<agent<grainsize>,blocksize>,Closure>
{
  typedef cuda_launcher_base<blocksize,concurrent_group<agent<grainsize>,blocksize>,Closure> super_t;
  typedef typename super_t::size_type size_type;
  typedef typename super_t::task_type task_type;

  typedef concurrent_group<agent<grainsize>,blocksize> block_type;

  __host__ __device__
  void launch(block_type request, Closure c, cudaStream_t stream)
  {
    block_type b = configure(request);

    size_type block_size = b.size();
    size_type heap_size  = b.heap_size();

    if(block_size > 0)
    {
      task_type task(b, c);
      super_t::launch(1, block_size, heap_size, stream, task);
    } // end if
  } // end go()

  __host__ __device__
  block_type configure(block_type b)
  {
    size_type block_size = super_t::choose_group_size(b.size());
    size_type heap_size  = super_t::choose_heap_size(device_properties(), block_size, b.heap_size());
    return make_block<block_type>(block_size, heap_size);
  } // end configure()
}; // end cuda_launcher


template<std::size_t groupsize, std::size_t grainsize, typename Closure>
struct cuda_launcher<
  parallel_group<
    agent<grainsize>,
    groupsize
  >,
  Closure
>
  : public cuda_launcher_base<dynamic_group_size, parallel_group<agent<grainsize>,groupsize>,Closure>
{
  typedef cuda_launcher_base<dynamic_group_size, parallel_group<agent<grainsize>,groupsize>,Closure> super_t;
  typedef typename super_t::size_type size_type; 
  typedef typename super_t::task_type task_type;

  typedef parallel_group<agent<grainsize>,groupsize> group_type;

  __host__ __device__
  void launch(group_type g, Closure c, cudaStream_t stream)
  {
    size_type num_blocks, block_size;
    thrust::tie(num_blocks,block_size) = configure(g);

    if(num_blocks > 0 && block_size > 0)
    {
      task_type task(g, c);

      super_t::launch(num_blocks, block_size, 0, stream, task);
    } // end if
  } // end go()

  __host__ __device__
  thrust::tuple<size_type,size_type> configure(group_type g)
  {
    size_type block_size = thrust::min<size_type>(g.size(), super_t::choose_group_size(use_default));

    // don't ask for more than a reasonable number of blocks
    size_type max_blocks = super_t::choose_num_groups(bulk::use_default, block_size);

    // given no limits at all, how many blocks would we launch?
    size_type num_blocks = (block_size > 0) ? (g.size() + block_size - 1) / block_size : 0;

    // don't ask for more blocks than the limit we prescribed for ourself
    num_blocks = thrust::min<size_type>(num_blocks, max_blocks);

    return thrust::make_tuple(num_blocks, block_size);
  } // end configure()
}; // end cuda_launcher


} // end detail
} // end bulk
BULK_NAMESPACE_SUFFIX

