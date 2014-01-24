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
#include <thrust/system/cuda/detail/bulk/malloc.hpp>
#include <thrust/system/cuda/detail/bulk/algorithm/copy.hpp>
#include <thrust/system/cuda/detail/bulk/algorithm/accumulate.hpp>
#include <thrust/system/cuda/detail/bulk/uninitialized.hpp>
#include <thrust/detail/type_traits.h>
#include <thrust/detail/type_traits/function_traits.h>
#include <thrust/detail/type_traits/iterator/is_output_iterator.h>


BULK_NAMESPACE_PREFIX
namespace bulk
{


template<std::size_t bound, std::size_t grainsize, typename RandomAccessIterator1, typename RandomAccessIterator2, typename T, typename BinaryFunction>
__forceinline__ __device__
RandomAccessIterator2
  inclusive_scan(const bounded<bound, bulk::agent<grainsize> > &exec,
                 RandomAccessIterator1 first,
                 RandomAccessIterator1 last,
                 RandomAccessIterator2 result,
                 T init,
                 BinaryFunction binary_op)
{
  for(int i = 0; i < exec.bound(); ++i)
  {
    if(first + i < last)
    {
      init = binary_op(init, first[i]);
      result[i] = init;
    } // end if
  } // end for

  return result + (last - first);
} // end inclusive_scan


template<std::size_t bound, std::size_t grainsize, typename RandomAccessIterator1, typename RandomAccessIterator2, typename T, typename BinaryFunction>
__forceinline__ __device__
RandomAccessIterator2
  exclusive_scan(const bounded<bound, bulk::agent<grainsize> > &exec,
                 RandomAccessIterator1 first,
                 RandomAccessIterator1 last,
                 RandomAccessIterator2 result,
                 T init,
                 BinaryFunction binary_op)
{
  for(int i = 0; i < exec.bound(); ++i)
  {
    if(first + i < last)
    {
      result[i] = init;
      init = binary_op(init, first[i]);
    } // end if
  } // end for

  return result + (last - first);
} // end exclusive_scan


namespace detail
{
namespace scan_detail
{


template<typename InputIterator, typename OutputIterator, typename BinaryFunction>
struct scan_intermediate
  : thrust::detail::eval_if<
      thrust::detail::has_result_type<BinaryFunction>::value,
      thrust::detail::result_type<BinaryFunction>,
      thrust::detail::eval_if<
        thrust::detail::is_output_iterator<OutputIterator>::value,
        thrust::iterator_value<InputIterator>,
        thrust::iterator_value<OutputIterator>
      >
    >
{};


template<typename ConcurrentGroup, typename RandomAccessIterator, typename T, typename BinaryFunction>
__device__ T inplace_exclusive_scan(ConcurrentGroup &g, RandomAccessIterator first, T init, BinaryFunction binary_op)
{
  typedef typename ConcurrentGroup::size_type size_type;

  size_type tid = g.this_exec.index();

  if(tid == 0)
  {
    first[0] = binary_op(init, first[0]);
  }

  T x = first[tid];

  g.wait();

  for(size_type offset = 1; offset < g.size(); offset += offset)
  {
    if(tid >= offset)
    {
      x = binary_op(first[tid - offset], x);
    }

    g.wait();

    first[tid] = x;

    g.wait();
  }

  T result = first[g.size() - 1];

  if(tid == 0)
  {
    x = init;
  }
  else
  {
    x = first[tid - 1];
  }

  g.wait();

  first[tid] = x;

  g.wait();

  return result;
}


template<typename ConcurrentGroup, typename RandomAccessIterator, typename Size, typename T, typename BinaryFunction>
__device__ T small_inplace_exclusive_scan(ConcurrentGroup &g, RandomAccessIterator first, Size n, T init, BinaryFunction binary_op)
{
  typedef typename ConcurrentGroup::size_type size_type;

  size_type tid = g.this_exec.index();

  if(tid == 0)
  {
    first[0] = binary_op(init, first[0]);
  }

  T x = tid < n ? first[tid] : init;

  g.wait();

  for(size_type offset = 1; offset < g.size(); offset += offset)
  {
    if(tid >= offset && tid - offset < n)
    {
      x = binary_op(first[tid - offset], x);
    }

    g.wait();

    if(tid < n)
    {
      first[tid] = x;
    }

    g.wait();
  }

  T result = first[n - 1];

  if(tid < n)
  {
    if(tid == 0)
    {
      x = init;
    }
    else
    {
      x = first[tid - 1];
    }
  }

  g.wait();

  if(tid < n)
  {
    first[tid] = x;
  }

  g.wait();

  return result;
}


// the upper bound on n is g.size()
template<typename ConcurrentGroup, typename RandomAccessIterator, typename Size, typename T, typename BinaryFunction>
__device__ T bounded_inplace_exclusive_scan(ConcurrentGroup &g, RandomAccessIterator first, Size n, T init, BinaryFunction binary_op)
{
  return (n == g.size()) ?
    inplace_exclusive_scan(g, first, init, binary_op) :
    small_inplace_exclusive_scan(g, first, n, init, binary_op);
}


template<bool inclusive,
         std::size_t bound, std::size_t groupsize, std::size_t grainsize,
         typename RandomAccessIterator1,
         typename RandomAccessIterator2,
         typename T,
         typename BinaryFunction>
__device__
// XXX MSVC9 has trouble with this enable_if, so just don't bother with it
//typename thrust::detail::enable_if<
//  bound <= groupsize * grainsize,
//  T
//>::type
T
scan(bulk::bounded<
       bound,
       bulk::concurrent_group<bulk::agent<grainsize>,groupsize>
     > &g,
     RandomAccessIterator1 first, RandomAccessIterator1 last,
     RandomAccessIterator2 result,
     T carry_in,
     BinaryFunction binary_op)
{
  typedef typename thrust::iterator_value<RandomAccessIterator1>::type input_type;

  typedef typename scan_intermediate<
    RandomAccessIterator1,
    RandomAccessIterator2,
    BinaryFunction
  >::type intermediate_type;
  
  typedef typename bulk::bounded<
    bound,
    bulk::concurrent_group<bulk::agent<grainsize>,groupsize>
  >::size_type size_type;

  size_type tid = g.this_exec.index();
  size_type n = last - first;

  // make a local copy from the input
  input_type local_inputs[grainsize];
  
  size_type local_offset = grainsize * tid;
  size_type local_size = thrust::max<size_type>(0,thrust::min<size_type>(grainsize, n - grainsize * tid));
  
  bulk::copy_n(bulk::bound<grainsize>(g.this_exec), first + local_offset, local_size, local_inputs);
  
  // XXX this should be uninitialized<intermediate_type>
  intermediate_type x;
  
  if(local_size)
  {
    x = local_inputs[0];
    x = bulk::accumulate(bulk::bound<grainsize-1>(g.this_exec), local_inputs + 1, local_inputs + local_size, x, binary_op);
  } // end if
  
  g.wait();
  
  if(local_size)
  {
    result[tid] = x;
  } // end if
  
  g.wait();

  // count the number of spine elements
  const size_type spine_n = (n >= g.size() * g.this_exec.grainsize()) ? g.size() : (n + g.this_exec.grainsize() - 1) / g.this_exec.grainsize();
  
  // exclusive scan the array of per-thread sums
  // XXX this call is another bounded scan
  //     the bound is groupsize
  carry_in = bounded_inplace_exclusive_scan(g, result, spine_n, carry_in, binary_op);
  
  if(local_size)
  {
    x = result[tid];
  } // end if
  
  g.wait();
  
  if(inclusive)
  {
    bulk::inclusive_scan(bulk::bound<grainsize>(g.this_exec), local_inputs, local_inputs + local_size, result + local_offset, x, binary_op);
  } // end if
  else
  {
    bulk::exclusive_scan(bulk::bound<grainsize>(g.this_exec), local_inputs, local_inputs + local_size, result + local_offset, x, binary_op);
  } // end else
  
  g.wait();

  return carry_in;
} // end scan()


template<std::size_t groupsize, std::size_t grainsize, typename RandomAccessIterator1, typename RandomAccessIterator2, typename BinaryFunction>
struct scan_buffer
{
  typedef typename thrust::iterator_value<RandomAccessIterator1>::type  input_type;

  typedef typename scan_intermediate<
    RandomAccessIterator1,
    RandomAccessIterator2,
    BinaryFunction
  >::type intermediate_type;

  union
  {
    uninitialized_array<input_type, groupsize * grainsize>        inputs;
    uninitialized_array<intermediate_type, groupsize * grainsize> results;
  };
};


template<bool inclusive, std::size_t groupsize, std::size_t grainsize, typename RandomAccessIterator1, typename RandomAccessIterator2, typename T, typename BinaryFunction>
__device__ void scan_with_buffer(bulk::concurrent_group<bulk::agent<grainsize>,groupsize> &g,
                                 RandomAccessIterator1 first, RandomAccessIterator1 last,
                                 RandomAccessIterator2 result,
                                 T carry_in,
                                 BinaryFunction binary_op,
                                 scan_buffer<groupsize,grainsize,RandomAccessIterator1,RandomAccessIterator2,BinaryFunction> &buffer)
{
  typedef scan_buffer<
    groupsize,
    grainsize,
    RandomAccessIterator1,
    RandomAccessIterator2,
    BinaryFunction
  > buffer_type;

  typedef typename buffer_type::input_type        input_type;
  typedef typename buffer_type::intermediate_type intermediate_type;

  // XXX grabbing this pointer up front before the loop is noticeably
  //     faster than dereferencing inputs or results inside buffer
  //     in the loop below
  union {
    input_type        *inputs;
    intermediate_type *results;
  } stage;

  stage.inputs = buffer.inputs.data();

  typedef typename bulk::concurrent_group<bulk::agent<grainsize>,groupsize>::size_type size_type;

  size_type tid = g.this_exec.index();

  const size_type elements_per_group = groupsize * grainsize;

  for(; first < last; first += elements_per_group, result += elements_per_group)
  {
    size_type partition_size = thrust::min<size_type>(elements_per_group, last - first);
    
    // stage data through shared memory
    bulk::copy_n(g, first, partition_size, stage.inputs);

    carry_in = scan<inclusive>(bulk::bound<elements_per_group>(g),
                               stage.inputs, stage.inputs + partition_size,
                               stage.results,
                               carry_in,
                               binary_op);
    
    // copy to result 
    bulk::copy_n(g, stage.results, partition_size, result);
  } // end for
} // end scan_with_buffer()


} // end scan_detail
} // end detail


template<std::size_t bound,
         std::size_t groupsize,
         std::size_t grainsize,
         typename RandomAccessIterator1,
         typename RandomAccessIterator2,
         typename T,
         typename BinaryFunction>
__device__
typename thrust::detail::enable_if<
  bound <= groupsize * grainsize,
  RandomAccessIterator2
>::type
inclusive_scan(bulk::bounded<
                 bound,
                 bulk::concurrent_group<bulk::agent<grainsize>,groupsize>
               > &g,
               RandomAccessIterator1 first, RandomAccessIterator1 last,
               RandomAccessIterator2 result,
               T carry_in,
               BinaryFunction binary_op)
{
  detail::scan_detail::scan<true>(g, first, last, result, carry_in, binary_op);
  return result + (last - first);
} // end inclusive_scan()


template<std::size_t bound,
         std::size_t groupsize,
         std::size_t grainsize,
         typename RandomAccessIterator1,
         typename RandomAccessIterator2,
         typename BinaryFunction>
__device__
typename thrust::detail::enable_if<
  bound <= groupsize * grainsize,
  RandomAccessIterator2
>::type
inclusive_scan(bulk::bounded<
                 bound,
                 bulk::concurrent_group<bulk::agent<grainsize>,groupsize>
               > &g,
               RandomAccessIterator1 first, RandomAccessIterator1 last,
               RandomAccessIterator2 result,
               BinaryFunction binary_op)
{
  if(bound > 0 && first < last)
  {
    typename thrust::iterator_value<RandomAccessIterator1>::type init = *first;

    // we need to wait because first may be the same as result
    g.wait();

    if(g.this_exec.index() == 0)
    {
      *result = init;
    }

    detail::scan_detail::scan<true>(g, first + 1, last, result + 1, init, binary_op);
  }

  return result + (last - first);
} // end inclusive_scan()


template<std::size_t groupsize,
         std::size_t grainsize,
         typename RandomAccessIterator1,
         typename RandomAccessIterator2,
         typename T,
         typename BinaryFunction>
__device__ void inclusive_scan(bulk::concurrent_group<bulk::agent<grainsize>,groupsize> &g,
                               RandomAccessIterator1 first, RandomAccessIterator1 last,
                               RandomAccessIterator2 result,
                               T init,
                               BinaryFunction binary_op)
{
  typedef detail::scan_detail::scan_buffer<groupsize,grainsize,RandomAccessIterator1,RandomAccessIterator2,BinaryFunction> buffer_type;

#if __CUDA_ARCH__ >= 200
  buffer_type *buffer = reinterpret_cast<buffer_type*>(bulk::malloc(g, sizeof(buffer_type)));

  if(bulk::is_on_chip(buffer))
  {
    detail::scan_detail::scan_with_buffer<true>(g, first, last, result, init, binary_op, *bulk::on_chip_cast(buffer));
  } // end if
  else
  {
    detail::scan_detail::scan_with_buffer<true>(g, first, last, result, init, binary_op, *buffer);
  } // end else

  bulk::free(g, buffer);
#else
  __shared__ uninitialized<buffer_type> buffer;
  detail::scan_detail::scan_with_buffer<true>(g, first, last, result, init, binary_op, buffer.get());
#endif // __CUDA_ARCH__
} // end inclusive_scan()


template<std::size_t size,
         std::size_t grainsize,
         typename RandomAccessIterator1,
         typename RandomAccessIterator2,
         typename BinaryFunction>
__device__
RandomAccessIterator2
inclusive_scan(bulk::concurrent_group<bulk::agent<grainsize>,size> &this_group,
               RandomAccessIterator1 first,
               RandomAccessIterator1 last,
               RandomAccessIterator2 result,
               BinaryFunction binary_op)
{
  if(first < last)
  {
    // the first input becomes the init
    // XXX convert to the immediate type when passing init to respect Thrust's semantics
    //     when Thrust adopts the semantics of N3724, just forward along *first
    //typename thrust::iterator_value<RandomAccessIterator1>::type init = *first;
    typename detail::scan_detail::scan_intermediate<
      RandomAccessIterator1,
      RandomAccessIterator2,
      BinaryFunction
    >::type init = *first;

    // we need to wait because first may be the same as result
    this_group.wait();

    if(this_group.this_exec.index() == 0)
    {
      *result = init;
    } // end if

    bulk::inclusive_scan(this_group, first + 1, last, result + 1, init, binary_op);
  } // end if

  return result + (last - first);
} // end inclusive_scan()


template<std::size_t bound, std::size_t groupsize, std::size_t grainsize,
         typename RandomAccessIterator1,
         typename RandomAccessIterator2,
         typename T,
         typename BinaryFunction>
__device__
typename thrust::detail::enable_if<
  bound <= groupsize * grainsize,
  RandomAccessIterator2
>::type
exclusive_scan(bulk::bounded<
                 bound,
                 bulk::concurrent_group<bulk::agent<grainsize>,groupsize>
               > &g,
               RandomAccessIterator1 first, RandomAccessIterator1 last,
               RandomAccessIterator2 result,
               T carry_in,
               BinaryFunction binary_op)
{
  detail::scan_detail::scan<true>(g, first, last, result, carry_in, binary_op);
  return result + (last - first);
} // end exclusive_scan()


template<std::size_t groupsize,
         std::size_t grainsize,
         typename RandomAccessIterator1,
         typename RandomAccessIterator2,
         typename T,
         typename BinaryFunction>
__device__
typename thrust::detail::enable_if<
  (groupsize > 0),
  RandomAccessIterator2
>::type
exclusive_scan(bulk::concurrent_group<agent<grainsize>,groupsize> &g,
               RandomAccessIterator1 first, RandomAccessIterator1 last,
               RandomAccessIterator2 result,
               T init,
               BinaryFunction binary_op)
{
  typedef detail::scan_detail::scan_buffer<groupsize,grainsize,RandomAccessIterator1,RandomAccessIterator2,BinaryFunction> buffer_type;

#if __CUDA_ARCH__ >= 200
  buffer_type *buffer = reinterpret_cast<buffer_type*>(bulk::malloc(g, sizeof(buffer_type)));

  if(bulk::is_on_chip(buffer))
  {
    detail::scan_detail::scan_with_buffer<false>(g, first, last, result, init, binary_op, *bulk::on_chip_cast(buffer));
  } // end if
  else
  {
    detail::scan_detail::scan_with_buffer<false>(g, first, last, result, init, binary_op, *buffer);
  } // end else

  bulk::free(g, buffer);
#else
  __shared__ uninitialized<buffer_type> buffer;
  detail::scan_detail::scan_with_buffer<false>(g, first, last, result, init, binary_op, buffer.get());
#endif

  return result + (last - first);
} // end exclusive_scan()


} // end bulk
BULK_NAMESPACE_SUFFIX

