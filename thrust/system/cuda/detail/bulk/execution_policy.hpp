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
#include <thrust/system/cuda/detail/bulk/future.hpp>
#include <thrust/detail/type_traits.h>
#include <thrust/system/cuda/detail/bulk/detail/cuda_launcher/runtime_introspection.hpp>
#include <cstddef>


BULK_NAMESPACE_PREFIX
namespace bulk
{

// ExecutionAgent requirements:
//
// template<typename T>
// concept bool ExecutionAgent()
// {
//   return requires(T t)
//   {
//     typename T::size_type;
//     {t.index()} -> typename T::size_type;
//   }
// };
//
// ExecutionGroup requirements:
//
// template<typename T>
// concept bool ExecutionGroup()
// {
//   return ExecutionAgent<T>
//       && requires(T g)
//   {
//     typename T::agent_type;
//     ExecutionAgent<typename T::agent_type>();
//     {g.size()} -> typename T::size_type;
//     {g.this_exec} -> typename T::agent_type &
//   }
// };


static const int invalid_index = INT_MAX;


// sequential execution with a grainsize hint and index within a group
// a light-weight (logical) thread
template<std::size_t grainsize_ = 1>
class agent
{
  public:
    typedef int size_type;

    static const size_type static_grainsize = grainsize_;

    __host__ __device__
    agent(size_type i = invalid_index)
      : m_index(i)
    {}

    __host__ __device__
    size_type index() const
    {
      return m_index;
    }

    __host__ __device__
    size_type grainsize() const
    {
      return static_grainsize;
    }

  private:
    const size_type m_index;
};


static const int use_default = INT_MAX;

static const int dynamic_group_size = 0;


namespace detail
{
namespace group_detail
{


template<typename ExecutionAgent, std::size_t size_>
class group_base
{
  public:
    typedef ExecutionAgent agent_type;

    typedef int size_type;

    static const size_type static_size = size_;

    __host__ __device__
    group_base(agent_type exec = agent_type(), size_type i = invalid_index)
      : this_exec(exec),
        m_index(i)
    {}

    __host__ __device__
    size_type index() const
    {
      return m_index;
    }

    __host__ __device__
    size_type size() const
    {
      return static_size;
    }

    __device__
    size_type global_index() const
    {
      return index() * size() + this_exec.index();
    }

    agent_type this_exec;

  private:
    const size_type m_index;
};


template<typename ExecutionAgent>
class group_base<ExecutionAgent,dynamic_group_size>
{
  public:
    typedef ExecutionAgent agent_type;

    typedef int size_type;

    __host__ __device__
    group_base(size_type sz, agent_type exec = agent_type(), size_type i = invalid_index)
      : this_exec(exec),
        m_size(sz),
        m_index(i)
    {}

    __host__ __device__
    size_type index() const
    {
      return m_index;
    }

    __host__ __device__
    size_type size() const
    {
      return m_size;
    }

    __host__ __device__
    size_type global_index() const
    {
      return index() * size() + this_exec.index();
    }

    agent_type this_exec;

  private:
    const size_type m_size;
    const size_type m_index;
};


} // end group_detail
} // end detail


// a group of independent ExecutionAgents
template<typename ExecutionAgent = agent<>,
         std::size_t size_ = dynamic_group_size>
class parallel_group
  : public detail::group_detail::group_base<ExecutionAgent,size_>
{
  private:
    typedef detail::group_detail::group_base<
      ExecutionAgent,
      size_
    > super_t;

  public:
    typedef typename super_t::agent_type agent_type;

    typedef typename super_t::size_type  size_type;

    // XXX the constructor taking an index should be made private
    __host__ __device__
    parallel_group(agent_type exec = agent_type(), size_type i = invalid_index)
      : super_t(exec,i)
    {}
};


template<typename ExecutionAgent>
class parallel_group<ExecutionAgent,dynamic_group_size>
  : public detail::group_detail::group_base<ExecutionAgent,dynamic_group_size>
{
  private:
    typedef detail::group_detail::group_base<
      ExecutionAgent,
      dynamic_group_size
    > super_t;

  public:
    typedef typename super_t::agent_type agent_type;

    typedef typename super_t::size_type  size_type;

    // XXX the constructor taking an index should be made private
    __host__ __device__
    parallel_group(size_type size, agent_type exec = agent_type(), size_type i = invalid_index)
      : super_t(size,exec,i)
    {}
};


// shorthand for creating a parallel_group of agents
inline __host__ __device__
parallel_group<> par(size_t size)
{
  return parallel_group<>(size);
}


// shorthand for creating a parallel_group of ExecutionAgents
template<typename ExecutionAgent>
__host__ __device__
parallel_group<ExecutionAgent> par(ExecutionAgent exec, size_t size)
{
  return parallel_group<ExecutionAgent>(size, exec);
}


template<typename ExecutionAgent>
class async_launch
{
  public:
    __host__ __device__
    async_launch(ExecutionAgent exec, cudaStream_t s, cudaEvent_t be = 0)
      : stream_valid(true),e(exec),s(s),be(be)
    {}

    __host__
    async_launch(ExecutionAgent exec, cudaEvent_t be)
      : stream_valid(false),e(exec),s(0),be(be)
    {}

    __host__ __device__
    ExecutionAgent exec() const
    {
      return e;
    }

    __host__ __device__
    cudaStream_t stream() const
    {
      return s;
    }

    __host__ __device__
    cudaEvent_t before_event() const
    {
      return be;
    }

    __host__ __device__
    bool is_stream_valid() const
    {
      return stream_valid;
    }

  private:
    bool stream_valid;
    ExecutionAgent e;
    cudaStream_t s;
    cudaEvent_t be;
};


inline __host__ __device__
async_launch<bulk::parallel_group<> > par(cudaStream_t s, size_t num_threads)
{
  return async_launch<bulk::parallel_group<> >(bulk::parallel_group<>(num_threads), s);
}


template<typename ExecutionAgent>
inline __host__ __device__
async_launch<bulk::parallel_group<ExecutionAgent> > par(cudaStream_t s, ExecutionAgent exec, size_t num_groups)
{
  return async_launch<bulk::parallel_group<ExecutionAgent> >(bulk::par(exec, num_groups), s);
}


inline async_launch<bulk::parallel_group<> > par(bulk::future<void> &before, size_t num_threads)
{
  cudaEvent_t before_event = bulk::detail::future_core_access::event(before);

  return async_launch<bulk::parallel_group<> >(bulk::parallel_group<>(num_threads), before_event);
}


// a group of concurrent ExecutionAgents which may synchronize
template<typename ExecutionAgent      = agent<>,
         std::size_t size_      = dynamic_group_size>
class concurrent_group
  : public parallel_group<ExecutionAgent,size_>
{
  private:
    typedef parallel_group<
      ExecutionAgent,
      size_
    > super_t;

  public:
    typedef typename super_t::agent_type agent_type;
    typedef typename super_t::size_type  size_type;

    // XXX the constructor taking an index should be made private
    __host__ __device__
    concurrent_group(size_type heap_size = use_default,
                     agent_type exec = agent_type(),
                     size_type i = invalid_index)
      : super_t(exec,i),
        m_heap_size(heap_size)
    {}

    __device__
    void wait() const
    {
      // guard use of __syncthreads from foreign compilers
#ifdef __CUDA_ARCH__
      __syncthreads();
#endif
    }

    __host__ __device__
    size_type heap_size() const
    {
      return m_heap_size;
    }

    // XXX this should go elsewhere
    __host__ __device__
    inline static size_type hardware_concurrency()
    {
#if __BULK_HAS_CUDART__
      return static_cast<size_type>(bulk::detail::device_properties().multiProcessorCount);
#else
      return 0;
#endif
    } // end hardware_concurrency()

  private:
    size_type m_heap_size;
};


template<typename ExecutionAgent>
class concurrent_group<ExecutionAgent,dynamic_group_size>
  : public parallel_group<ExecutionAgent,dynamic_group_size>
{
  private:
    typedef parallel_group<
      ExecutionAgent,
      dynamic_group_size
    > super_t;

  public:
    typedef typename super_t::agent_type agent_type;

    typedef typename super_t::size_type  size_type;

    // XXX the constructor taking an index should be made private
    __host__ __device__
    concurrent_group(size_type size,
                     size_type heap_size = use_default,
                     agent_type exec = agent_type(),
                     size_type i = invalid_index)
      : super_t(size,exec,i),
        m_heap_size(heap_size)
    {}

    __device__
    void wait()
    {
      // guard use of __syncthreads from foreign compilers
#ifdef __CUDA_ARCH__
      __syncthreads();
#endif
    }

    __host__ __device__
    size_type heap_size() const
    {
      return m_heap_size;
    }

    // XXX this should go elsewhere
    __host__ __device__
    inline static size_type hardware_concurrency()
    {
#if __BULK_HAS_CUDART__
      return static_cast<size_type>(bulk::detail::device_properties().multiProcessorCount);
#else
      return 0;
#endif
    } // end hardware_concurrency()

  private:
    size_type m_heap_size;
};


// shorthand for creating a concurrent_group of agents
inline __host__ __device__
concurrent_group<> con(size_t size, size_t heap_size = use_default)
{
  return concurrent_group<>(size,heap_size);
}


// shorthand for creating a concurrent_group of ExecutionAgents
template<typename ExecutionAgent>
__host__ __device__
concurrent_group<ExecutionAgent> con(ExecutionAgent exec, size_t size, size_t heap_size = use_default)
{
  return concurrent_group<ExecutionAgent>(size,heap_size,exec);
}


// shorthand for creating a concurrent_group of agents with static sizing
template<std::size_t groupsize, std::size_t grainsize>
__host__ __device__
concurrent_group<bulk::agent<grainsize>,groupsize>
con(size_t heap_size)
{
  return concurrent_group<bulk::agent<grainsize>,groupsize>(heap_size);
}


// a way to statically bound the size of an ExecutionAgent's work
template<std::size_t bound_, typename ExecutionAgent>
class bounded
  : public ExecutionAgent
{
  public:
    typedef typename ExecutionAgent::size_type size_type;

    static const size_type static_bound = bound_;

    __host__ __device__
    size_type bound() const
    {
      return static_bound;
    }


    __host__ __device__
    ExecutionAgent &unbound()
    {
      return *this;
    }


    __host__ __device__
    const ExecutionAgent &unbound() const
    {
      return *this;
    }


  private:
    // XXX delete these unless we find a need for them
    bounded();

    bounded(const bounded &);
};


template<std::size_t bound_, typename ExecutionAgent>
__host__ __device__
bounded<bound_, ExecutionAgent> &bound(ExecutionAgent &exec)
{
  return static_cast<bounded<bound_, ExecutionAgent>&>(exec);
}


template<std::size_t bound_, typename ExecutionAgent>
__host__ __device__
const bounded<bound_, ExecutionAgent> &bound(const ExecutionAgent &exec)
{
  return static_cast<const bounded<bound_, ExecutionAgent>&>(exec);
}


namespace detail
{


template<unsigned int depth, typename ExecutionAgent>
struct agent_at_depth
{
  typedef typename agent_at_depth<
    depth-1,ExecutionAgent
  >::type parent_agent_type;

  typedef typename parent_agent_type::agent_type type;
};


template<typename ExecutionAgent>
struct agent_at_depth<0,ExecutionAgent>
{
  typedef ExecutionAgent type;
};


template<typename Cursor, typename ExecutionGroup>
struct cursor_result
{
  typedef typename agent_at_depth<Cursor::depth,ExecutionGroup>::type & type;
};


template<unsigned int d> struct cursor;


template<unsigned int d>
struct cursor
{
  static const unsigned int depth = d;

  __host__ __device__ cursor() {}

  cursor<depth+1> this_exec;

  template<typename ExecutionGroup>
  static __host__ __device__
  typename cursor_result<cursor,ExecutionGroup>::type
  get(ExecutionGroup &root)
  {
    return cursor<depth-1>::get(root.this_exec);
  }
};


template<> struct cursor<3>
{
  static const unsigned int depth = 3;

  __host__ __device__ cursor() {}

  template<typename ExecutionGroup>
  static __host__ __device__
  typename cursor_result<cursor,ExecutionGroup>::type
  get(ExecutionGroup &root)
  {
    return cursor<depth-1>::get(root.this_exec);
  }
};


template<> struct cursor<0>
{
  static const unsigned int depth = 0;

  __host__ __device__ cursor() {}

  cursor<1> this_exec;

  // the root level cursor simply returns the root
  template<typename ExecutionAgent>
  static __host__ __device__
  ExecutionAgent &get(ExecutionAgent &root)
  {
    return root;
  }
};


template<typename T> struct is_cursor : thrust::detail::false_type {};


template<unsigned int d>
struct is_cursor<cursor<d> >
  : thrust::detail::true_type
{};


} // end detail


#ifdef __CUDA_ARCH__
static const __device__ detail::cursor<0> root;
#else
static const detail::cursor<0> root;
#endif


// shorthand for creating a parallel group of concurrent groups of agents
inline __host__ __device__
parallel_group<concurrent_group<> > grid(size_t num_groups = use_default, size_t group_size = use_default, size_t heap_size = use_default)
{
  return par(con(group_size,heap_size), num_groups);
}
               
  


inline __host__ __device__
async_launch<
  parallel_group<concurrent_group<> >
>
  grid(size_t num_groups, size_t group_size, size_t heap_size, cudaStream_t stream)
{
  return par(stream, con(group_size,heap_size), num_groups);
}


template<std::size_t groupsize, std::size_t grainsize>
__host__ __device__
parallel_group<
  concurrent_group<
    bulk::agent<grainsize>,
    groupsize
  >
>
  grid(size_t num_groups, size_t heap_size = use_default)
{
  return par(con<groupsize,grainsize>(heap_size), num_groups);
}


template<std::size_t groupsize, std::size_t grainsize>
__host__ __device__
async_launch<
  parallel_group<
    concurrent_group<
      bulk::agent<grainsize>,
      groupsize
    >
  >
>
  grid(size_t num_groups, size_t heap_size, cudaStream_t stream)
{
  return par(stream, con<groupsize,grainsize>(heap_size), num_groups);
}


} // end bulk
BULK_NAMESPACE_SUFFIX

