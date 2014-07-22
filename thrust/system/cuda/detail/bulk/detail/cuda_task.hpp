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
#include <thrust/system/cuda/detail/bulk/malloc.hpp>
#include <thrust/system/cuda/detail/bulk/execution_policy.hpp>
#include <thrust/system/cuda/detail/bulk/detail/tuple_transform.hpp>
#include <thrust/system/cuda/detail/bulk/detail/closure.hpp>

#include <thrust/detail/type_traits.h>


BULK_NAMESPACE_PREFIX
namespace bulk
{
namespace detail
{


template<typename ExecutionGroup, typename Closure>
class task_base
{
  public:
    typedef ExecutionGroup group_type;
    typedef Closure        closure_type;

    __host__ __device__
    task_base(group_type g, closure_type c)
      : c(c), g(g)
    {}

  protected:
    __host__ __device__
    static void substitute_placeholders_and_execute(group_type &g, closure_type &c)
    {
      // substitute placeholders with this_group
      substituted_arguments_type new_args = substitute_placeholders(g, c.arguments());

      // create a new closure with the new arguments
      closure<typename closure_type::function_type, substituted_arguments_type> new_c(c.function(), new_args);

      // execute the new closure
      new_c();
    }

    closure_type c;
    group_type g;

  private:
    template<typename T>
    struct substitutor_result
      : thrust::detail::eval_if<
          bulk::detail::is_cursor<T>::value,
          cursor_result<T,ExecutionGroup>,
          thrust::detail::identity_<T>
        >
    {};

    typedef typename bulk::detail::tuple_meta_transform<
      typename closure_type::arguments_type,
      substitutor_result
    >::type substituted_arguments_type;

    struct substitutor
    {
      group_type &g;

      __device__
      substitutor(group_type &g)
        : g(g)
      {}

      template<unsigned int depth>
      __device__
      typename bulk::detail::cursor_result<cursor<depth>,group_type>::type
      operator()(cursor<depth> c) const
      {
        return c.get(g);
      }

      template<typename T>
      __device__
      T &operator()(T &x) const
      {
        return x;
      }
    };

    __host__ __device__
    static substituted_arguments_type substitute_placeholders(group_type &g, typename closure_type::arguments_type args)
    {
      return bulk::detail::tuple_host_device_transform<substitutor_result>(args, substitutor(g));
    }
};


template<std::size_t blocksize, std::size_t grainsize>
struct cuda_block
{
  typedef concurrent_group<agent<grainsize>, blocksize> type;
};


template<std::size_t gridsize, std::size_t blocksize, std::size_t grainsize>
struct cuda_grid
{
  typedef parallel_group<
    typename cuda_block<blocksize,grainsize>::type
  > type;
};


template<typename Group, typename Closure> class cuda_task;


template<typename Grid>
struct grid_maker
{
  __host__ __device__
  static Grid make(typename Grid::size_type     size,
                   typename Grid::agent_type    block,
                   typename Grid::size_type     index)
  {
    return Grid(block, index);
  }
};


template<typename Block>
struct grid_maker<parallel_group<Block,dynamic_group_size> >
{
  __host__ __device__
  static parallel_group<Block,dynamic_group_size> make(typename parallel_group<Block,dynamic_group_size>::size_type size,
                                                       Block block,
                                                       typename parallel_group<Block,dynamic_group_size>::size_type index)
  {
    return parallel_group<Block,dynamic_group_size>(size, block, index);
  }
};


template<typename Block>
struct block_maker
{
  __host__ __device__
  static Block make(typename Block::size_type     size,
                    typename Block::size_type     heap_size,
                    typename Block::agent_type    thread,
                    typename Block::size_type     index)
  {
    return Block(heap_size, thread, index);
  }
};

template<typename Thread>
struct block_maker<concurrent_group<Thread,dynamic_group_size> >
{
  __host__ __device__
  static concurrent_group<Thread,dynamic_group_size> make(typename concurrent_group<Thread,dynamic_group_size>::size_type size,
                                                          typename concurrent_group<Thread,dynamic_group_size>::size_type heap_size,
                                                          Thread thread,
                                                          typename concurrent_group<Thread,dynamic_group_size>::size_type index)
  {
    return concurrent_group<Thread,dynamic_group_size>(size, heap_size, thread, index);
  }
};


template<typename Grid>
__host__ __device__
Grid make_grid(typename Grid::size_type size, typename Grid::agent_type block, typename Grid::size_type index = invalid_index)
{
  return grid_maker<Grid>::make(size, block, index);
}


template<typename Block>
__host__ __device__
Block make_block(typename Block::size_type size, typename Block::size_type heap_size, typename Block::agent_type thread = typename Block::agent_type(), typename Block::size_type index = invalid_index)
{
  return block_maker<Block>::make(size, heap_size, thread, index);
}


// specialize cuda_task for a CUDA grid
template<std::size_t gridsize, std::size_t blocksize, std::size_t grainsize, typename Closure>
class cuda_task<
  parallel_group<
    concurrent_group<
      agent<grainsize>,
      blocksize
    >,
    gridsize
  >,
  Closure
> : public task_base<typename cuda_grid<gridsize,blocksize,grainsize>::type,Closure>
{
  private:
    typedef task_base<typename cuda_grid<gridsize,blocksize,grainsize>::type,Closure> super_t;

  public:
    typedef typename super_t::group_type    grid_type;
    typedef typename grid_type::agent_type  block_type;
    typedef typename block_type::agent_type thread_type;
    typedef typename super_t::closure_type  closure_type;
    typedef typename grid_type::size_type   size_type;

  private:
    size_type block_offset;

  public:

    __host__ __device__
    cuda_task(grid_type g, closure_type c, size_type offset)
      : super_t(g,c),
        block_offset(offset)
    {}

    __device__
    void operator()()
    {
      // guard use of CUDA built-ins from foreign compilers
#ifdef __CUDA_ARCH__
      // instantiate a view of this grid
      grid_type this_grid =
        make_grid<grid_type>(
          super_t::g.size(),
          make_block<block_type>(
            blockDim.x,
            super_t::g.this_exec.heap_size(),
            thread_type(threadIdx.x),
            block_offset + blockIdx.x
          ),
          0
      );

#if __CUDA_ARCH__ >= 200
      // initialize shared storage
      if(this_grid.this_exec.this_exec.index() == 0)
      {
        bulk::detail::init_on_chip_malloc(this_grid.this_exec.heap_size());
      }
      this_grid.this_exec.wait();
#endif

      substitute_placeholders_and_execute(this_grid, super_t::c);
#endif
    } // end operator()
}; // end cuda_task


// specialize cuda_task for a single CUDA block
template<std::size_t blocksize, std::size_t grainsize, typename Closure>
class cuda_task<
  concurrent_group<
    agent<grainsize>,
    blocksize
  >,
  Closure
> : public task_base<typename cuda_block<blocksize,grainsize>::type,Closure>
{
  private:
    typedef task_base<typename cuda_block<blocksize,grainsize>::type,Closure> super_t;

  public:
    typedef typename super_t::group_type    block_type;
    typedef typename block_type::agent_type thread_type;
    typedef typename super_t::closure_type  closure_type;
    typedef typename block_type::size_type  size_type;

  public:
    __host__ __device__
    cuda_task(block_type b, closure_type c)
      : super_t(b,c)
    {}

    __device__
    void operator()()
    {
      // guard use of CUDA built-ins from foreign compilers
#ifdef __CUDA_ARCH__
      // instantiate a view of this block
      block_type this_block =
        make_block<block_type>(
          blockDim.x,
          super_t::g.heap_size(),
          thread_type(threadIdx.x),
          0
        );

#if __CUDA_ARCH__ >= 200
      // initialize shared storage
      if(this_block.this_exec.index() == 0)
      {
        bulk::detail::init_on_chip_malloc(this_block.heap_size());
      }
      this_block.wait();
#endif

      substitute_placeholders_and_execute(this_block, super_t::c);
#endif
    } // end operator()
}; // end cuda_task


// specialize cuda_task for a single big parallel group
template<std::size_t groupsize, std::size_t grainsize, typename Closure>
class cuda_task<parallel_group<agent<grainsize>,groupsize>,Closure>
  : public task_base<parallel_group<agent<grainsize>,groupsize>,Closure>
{
  private:
    typedef task_base<parallel_group<agent<grainsize>,groupsize>,Closure> super_t;

  public:
    typedef typename super_t::closure_type closure_type;
    typedef typename super_t::group_type   group_type;

    __host__ __device__
    cuda_task(group_type g, closure_type c)
      : super_t(g,c)
    {}

    __device__
    void operator()()
    {
      // guard use of CUDA built-ins from foreign compilers
#ifdef __CUDA_ARCH__
      typedef int size_type;

      const size_type grid_size = gridDim.x * blockDim.x;

      for(size_type tid = blockDim.x * blockIdx.x + threadIdx.x;
          tid < super_t::g.size();
          tid += grid_size)
      {
        // instantiate a view of the exec group
        parallel_group<agent<grainsize>,groupsize> this_group(
          1,
          agent<grainsize>(tid),
          0
        );

        substitute_placeholders_and_execute(this_group, super_t::c);
      } // end for
#endif
    } // end operator()
}; // end cuda_task


} // end detail
} // end bulk
BULK_NAMESPACE_SUFFIX

