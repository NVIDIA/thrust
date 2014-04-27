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
#include <thrust/system/cuda/detail/bulk/uninitialized.hpp>
#include <thrust/system/cuda/detail/bulk/detail/cuda_task.hpp>
#include <thrust/system/cuda/detail/bulk/detail/throw_on_error.hpp>
#include <thrust/detail/minmax.h>
#include <thrust/system/cuda/detail/runtime_introspection.h>
#include <thrust/system/cuda/detail/synchronize.h>
#include <thrust/system/cuda/detail/execution_policy.h>
#include <thrust/system/cpp/detail/execution_policy.h>
#include <thrust/detail/temporary_array.h>
#include <thrust/pair.h>


namespace thrust
{
namespace detail
{


// XXX WAR circular inclusion problems with this forward declaration
// XXX consider not using temporary_array at all here to avoid these
//     issues
template<typename, typename> class temporary_array;


} // end detail
} // end thrust


BULK_NAMESPACE_PREFIX
namespace bulk
{
namespace detail
{


template<typename Function>
size_t maximum_potential_occupancy(Function kernel, size_t num_threads, size_t num_smem_bytes)
{
  namespace ns = thrust::system::cuda::detail;

  ns::function_attributes_t attr = ns::function_attributes(kernel);

  return ns::cuda_launch_config_detail::max_active_blocks_per_multiprocessor(ns::device_properties(),
                                                                             attr,
                                                                             num_threads,
                                                                             0);
}


#ifdef __CUDACC__
// if there are multiple versions of Bulk floating around, this may be #defined already
#  ifndef __bulk_launch_bounds__
#    define __bulk_launch_bounds__(num_threads_per_block, num_blocks_per_sm) __launch_bounds__(num_threads_per_block, num_blocks_per_sm)
#  endif
#else
#  ifndef __bulk_launch_bounds__
#    define __bulk_launch_bounds__(num_threads_per_block, num_blocks_per_sm)
#  endif
#endif // __CUDACC__


#if BULK_ASYNC_USE_UNINITIALIZED
// XXX uninitialized is a performance hazard
//     disable it for the moment
template<unsigned int block_size, typename Function>
__global__
__bulk_launch_bounds__(block_size, 0)
void launch_by_value(uninitialized<Function> f)
{
  f.get()();
}
#else
template<unsigned int block_size, typename Function>
__global__
__bulk_launch_bounds__(block_size, 0)
void launch_by_value(Function f)
{
  f();
}
#endif


template<unsigned int block_size, typename Function>
__global__
__bulk_launch_bounds__(block_size, 0)
void launch_by_pointer(const Function *f)
{
  // copy to registers
  Function f_reg = *f;
  f_reg();
}


// put this state in an anon namespace
namespace
{


bool verbose = false;


}


// sm_10 devices have 256 bytes of parameter space
template<unsigned int block_size_, typename Function, bool by_value = sizeof(Function) <= 256>
class triple_chevron_launcher
{
  public:
    typedef Function task_type;

#if BULK_ASYNC_USE_UNINITIALIZED
    typedef void (*global_function_t)(uninitialized<task_type>);
#else
    typedef void (*global_function_t)(task_type);
#endif


    static global_function_t get_global_function()
    {
      return launch_by_value<block_size_, task_type>;
    } // end get_launch_function()


    template<typename DerivedPolicy>
    void launch(thrust::cuda::execution_policy<DerivedPolicy> &, unsigned int num_blocks, unsigned int block_size, size_t num_dynamic_smem_bytes, cudaStream_t stream, task_type task)
    {
      // guard use of triple chevrons from foreign compilers
#ifdef __CUDACC__

#if BULK_ASYNC_USE_UNINITIALIZED
      uninitialized<task_type> wrapped_task;
      wrapped_task.construct(task);

      get_global_function()<<<static_cast<unsigned int>(num_blocks), static_cast<unsigned int>(block_size), static_cast<size_t>(num_dynamic_smem_bytes), stream>>>(wrapped_task);
#else
      get_global_function()<<<static_cast<unsigned int>(num_blocks), static_cast<unsigned int>(block_size), static_cast<size_t>(num_dynamic_smem_bytes), stream>>>(task);
#endif

#endif // __CUDACC__
    } // end launch()
};


template<unsigned int block_size_, typename Function>
class triple_chevron_launcher<block_size_,Function,false>
{
  public:
    typedef Function task_type;
    typedef void (*global_function_t)(const task_type*);

    static global_function_t get_global_function()
    {
      return launch_by_pointer<block_size_, task_type>;
    } // end get_launch_function()


    template<typename DerivedPolicy>
    void launch(thrust::cuda::execution_policy<DerivedPolicy> &exec, unsigned int num_blocks, unsigned int block_size, size_t num_dynamic_smem_bytes, cudaStream_t stream, task_type task)
    {
      // guard use of triple chevrons from foreign compilers
#ifdef __CUDACC__
      // use temporary storage for the task
      thrust::cpp::tag host_tag;
      thrust::detail::temporary_array<task_type,DerivedPolicy> task_storage(exec, host_tag, &task, &task + 1);

      get_global_function()<<<static_cast<unsigned int>(num_blocks), static_cast<unsigned int>(block_size), static_cast<size_t>(num_dynamic_smem_bytes), stream>>>((&task_storage[0]).get());
#endif // __CUDACC__
    } // end launch()
};


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


  void launch(size_type num_blocks, size_type block_size, size_type num_dynamic_smem_bytes, cudaStream_t stream, task_type task)
  {
    if(verbose)
    {
      cudaError_t error = cudaGetLastError();

      std::clog << "cuda_launcher_base::launch(): CUDA error before launch: " << cudaGetErrorString(error) << std::endl;
      std::clog << "cuda_launcher_base::launch(): num_blocks: " << num_blocks << std::endl;
      std::clog << "cuda_launcher_base::launch(): block_size: " << block_size << std::endl;
      std::clog << "cuda_launcher_base::launch(): num_dynamic_smem_bytes: " << num_dynamic_smem_bytes << std::endl;
      std::clog << "cuda_launcher_base::launch(): occupancy: " << maximum_potential_occupancy(super_t::get_global_function(), block_size, num_dynamic_smem_bytes) << std::endl;

      bulk::detail::throw_on_error(error, "before kernel launch in cuda_launcher_base::launch()");
    } // end if

    if(num_blocks > 0)
    {
      thrust::cuda::tag exec;
      super_t::launch(exec, num_blocks, block_size, num_dynamic_smem_bytes, stream, task);

      // check that the launch got off the ground
      bulk::detail::throw_on_error(cudaGetLastError(), "after kernel launch in cuda_launcher_base::launch()");

      thrust::system::cuda::detail::synchronize_if_enabled("bulk_kernel_by_value");
    } // end if
  } // end launch()


  typedef thrust::system::cuda::detail::function_attributes_t function_attributes_t;


  static function_attributes_t function_attributes()
  {
    return thrust::system::cuda::detail::function_attributes(super_t::get_global_function());
  } // end function_attributes()


  typedef thrust::system::cuda::detail::device_properties_t device_properties_t;

  static device_properties_t device_properties()
  {
    return thrust::system::cuda::detail::device_properties();
  } // end device_properties()


  static size_type max_active_blocks_per_multiprocessor(const device_properties_t &props,
                                                        const function_attributes_t &attr,
                                                        size_type num_threads_per_block,
                                                        size_type num_smem_bytes_per_block)
  {
    return thrust::system::cuda::detail::cuda_launch_config_detail::max_active_blocks_per_multiprocessor(props, attr, num_threads_per_block, num_smem_bytes_per_block);
  } // end max_active_blocks_per_multiprocessor()


  // returns
  // 1. maximum number of additional dynamic smem bytes that would not lower the kernel's occupancy
  // 2. kernel occupancy
  static thrust::pair<size_type,size_type> dynamic_smem_occupancy_limit(const device_properties_t &props, const function_attributes_t &attr, size_type num_threads_per_block, size_type num_smem_bytes_per_block)
  {
    // figure out the kernel's occupancy with 0 bytes of dynamic smem
    size_type occupancy = max_active_blocks_per_multiprocessor(props, attr, num_threads_per_block, num_smem_bytes_per_block);

    // if the kernel footprint is already too large, return (0,0)
    if(occupancy < 1) return thrust::make_pair(0,0);

    return thrust::make_pair(thrust::system::cuda::detail::proportional_smem_allocation(props, attr, occupancy), occupancy);
  } // end smem_occupancy_limit()


  static size_type choose_heap_size(size_type group_size, size_type requested_size)
  {
    function_attributes_t attr = function_attributes();

    // if the kernel's ptx version is < 200, we return 0 because there is no heap
    // if the user requested no heap, give him no heap
    if(attr.ptxVersion < 20 || requested_size == 0)
    {
      return 0;
    } // end if

    // how much smem could we allocate without reducing occupancy?
    device_properties_t props = device_properties();
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


  static size_type choose_group_size(size_type requested_size)
  {
    size_type result = requested_size;

    if(result == use_default)
    {
      function_attributes_t attr = function_attributes();

      return thrust::system::cuda::detail::block_size_with_maximum_potential_occupancy(attr, device_properties());
    } // end if

    return result;
  } // end choose_group_size()


  static size_type max_physical_grid_size()
  {
    // get the limit of the actual device
    int actual_limit = device_properties().maxGridSize[0];

    // get the limit of the PTX version of the kernel
    int ptx_version = function_attributes().ptxVersion;

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

          if(bulk::detail::verbose)
          {
            std::clog << "cuda_launcher::launch(): max_physical_grid_size: " << max_physical_grid_size << std::endl;
            std::clog << "cuda_launcher::launch(): requesting " << num_physical_blocks << " physical blocks" << std::endl;
          }

          super_t::launch(num_physical_blocks, block_size, heap_size, stream, task);

          num_remaining_physical_blocks -= num_physical_blocks;
        } // end for block_offset
      } // end if
    } // end if
  } // end go()

  static grid_type configure(grid_type g)
  {
    size_type block_size = super_t::choose_group_size(g.this_exec.size());
    size_type heap_size  = super_t::choose_heap_size(block_size, g.this_exec.heap_size());
    size_type num_blocks = g.size();

    return make_grid<grid_type>(num_blocks, make_block<block_type>(block_size, heap_size));
  } // end configure()
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

  static block_type configure(block_type b)
  {
    size_type block_size = super_t::choose_group_size(b.size());
    size_type heap_size  = super_t::choose_heap_size(block_size, b.heap_size());
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

  static thrust::tuple<size_type,size_type> configure(group_type g)
  {
    size_type block_size = thrust::min<size_type>(g.size(), super_t::choose_group_size(use_default));

    size_type num_blocks = (block_size > 0) ? (g.size() + block_size - 1) / block_size : 0;

    // don't request more blocks than we can physically launch
    size_type max_num_blocks = super_t::max_physical_grid_size();
    num_blocks = (num_blocks > max_num_blocks) ? max_num_blocks : num_blocks;

    return thrust::make_tuple(num_blocks, block_size);
  } // end configure()
}; // end cuda_launcher


} // end detail
} // end bulk
BULK_NAMESPACE_SUFFIX

