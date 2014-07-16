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
#include <thrust/system/cuda/detail/bulk/algorithm/copy.hpp>
#include <thrust/system/cuda/detail/bulk/algorithm/scan.hpp>
#include <thrust/system/cuda/detail/bulk/algorithm/scatter.hpp>
#include <thrust/system/cuda/detail/bulk/malloc.hpp>
#include <thrust/system/cuda/detail/bulk/detail/head_flags.hpp>
#include <thrust/system/cuda/detail/bulk/detail/tail_flags.hpp>
#include <thrust/detail/type_traits/function_traits.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/detail/minmax.h>


BULK_NAMESPACE_PREFIX
namespace bulk
{
namespace detail
{
namespace reduce_by_key_detail
{


template<typename FlagType, typename ValueType, typename BinaryFunction>
struct scan_head_flags_functor
{
  BinaryFunction binary_op;

  typedef thrust::tuple<FlagType,ValueType> result_type;
  typedef result_type first_argument_type;
  typedef result_type second_argument_type;

  __host__ __device__
  scan_head_flags_functor(BinaryFunction binary_op)
    : binary_op(binary_op)
  {}

  __host__ __device__
  result_type operator()(const first_argument_type &a, const second_argument_type &b)
  {
    ValueType val = thrust::get<0>(b) ? thrust::get<1>(b) : binary_op(thrust::get<1>(a), thrust::get<1>(b));
    FlagType flag = thrust::get<0>(a) + thrust::get<0>(b);
    return result_type(flag, val);
  }
};


template<typename ConcurrentGroup,
         typename InputIterator1,
         typename Size,
         typename InputIterator2,
         typename InputIterator3,
         typename OutputIterator1,
         typename OutputIterator2>
__device__
void scatter_tails_n(ConcurrentGroup &group,
                     InputIterator1 flags_first,
                     Size n,
                     InputIterator2 keys_first,
                     InputIterator3 values_first,
                     OutputIterator1 keys_result,
                     OutputIterator2 values_result)
{
  // for each tail element in [flags_first, flags_first + n)
  // scatter the key and value to that element's corresponding flag element - 1
  
  // the zip_iterators in this scatter_if can confuse nvcc's pointer space tracking for __CUDA_ARCH__ < 200
  // separate the scatters for __CUDA_ARCH__ < 200
#if __CUDA_ARCH__ >= 200
  bulk::scatter_if(group,
                   thrust::make_zip_iterator(thrust::make_tuple(values_first,         keys_first)),
                   thrust::make_zip_iterator(thrust::make_tuple(values_first + n - 1, keys_first)),
                   thrust::make_transform_iterator(flags_first, thrust::placeholders::_1 - 1),
                   bulk::detail::make_tail_flags(flags_first, flags_first + n).begin(),
                   thrust::make_zip_iterator(thrust::make_tuple(values_result, keys_result)));
#else
  bulk::scatter_if(group,
                   values_first, 
                   values_first + n - 1,
                   thrust::make_transform_iterator(flags_first, thrust::placeholders::_1 - 1),
                   bulk::detail::make_tail_flags(flags_first, flags_first + n).begin(),
                   values_result);

  bulk::scatter_if(group,
                   keys_first, 
                   keys_first + n - 1,
                   thrust::make_transform_iterator(flags_first, thrust::placeholders::_1 - 1),
                   bulk::detail::make_tail_flags(flags_first, flags_first + n).begin(),
                   keys_result);
#endif
} // end scatter_tails_n()


} // end reduce_by_key_detail
} // end detail


template<std::size_t groupsize,
         std::size_t grainsize,
         typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator1,
         typename OutputIterator2,
         typename T1,
         typename T2,
         typename BinaryPredicate,
         typename BinaryFunction>
thrust::tuple<
  OutputIterator1,
  OutputIterator2,
  typename thrust::iterator_value<InputIterator1>::type,
  typename thrust::iterator_value<OutputIterator2>::type
>
__device__
reduce_by_key(bulk::concurrent_group<bulk::agent<grainsize>,groupsize> &g,
              InputIterator1 keys_first, InputIterator1 keys_last,
              InputIterator2 values_first,
              OutputIterator1 keys_result,
              OutputIterator2 values_result,
              T1 init_key,
              T2 init_value,
              BinaryPredicate pred,
              BinaryFunction binary_op)
{
  typedef typename thrust::iterator_value<InputIterator2>::type value_type; // XXX this should be the type returned by BinaryFunction

  typedef typename bulk::concurrent_group<bulk::agent<grainsize>,groupsize>::size_type size_type;

  const size_type interval_size = groupsize * grainsize;

#if __CUDA_ARCH__ >= 200
  size_type *s_flags = reinterpret_cast<size_type*>(bulk::malloc(g, interval_size * sizeof(int)));
  value_type *s_values = reinterpret_cast<value_type*>(bulk::malloc(g, interval_size * sizeof(value_type)));
#else
  __shared__ uninitialized_array<size_type,interval_size> s_flags_impl;
  size_type *s_flags = s_flags_impl.data();

  __shared__ uninitialized_array<value_type,interval_size> s_values_impl;
  value_type *s_values = s_values_impl.data();
#endif

  for(; keys_first < keys_last; keys_first += interval_size, values_first += interval_size)
  {
    // upper bound on n is interval_size
    size_type n = thrust::min<size_type>(interval_size, keys_last - keys_first);

    bulk::detail::head_flags_with_init<
      InputIterator1,
      BinaryPredicate,
      size_type
    > flags(keys_first, keys_first + n, init_key, pred);

    detail::reduce_by_key_detail::scan_head_flags_functor<size_type, value_type, BinaryFunction> f(binary_op);

    // load input into smem
    bulk::copy_n(bulk::bound<interval_size>(g),
                 thrust::make_zip_iterator(thrust::make_tuple(flags.begin(), values_first)),
                 n,
                 thrust::make_zip_iterator(thrust::make_tuple(s_flags, s_values)));

    // scan in smem
    bulk::inclusive_scan(bulk::bound<interval_size>(g),
                         thrust::make_zip_iterator(thrust::make_tuple(s_flags,     s_values)),
                         thrust::make_zip_iterator(thrust::make_tuple(s_flags + n, s_values)),
                         thrust::make_zip_iterator(thrust::make_tuple(s_flags,     s_values)),
                         thrust::make_tuple(1, init_value),
                         f);

    // scatter tail results to the output
    detail::reduce_by_key_detail::scatter_tails_n(bulk::bound<interval_size>(g),
                                                  s_flags, n,
                                                  keys_first, s_values,
                                                  keys_result, values_result);


    // if the init was not a carry, we need to insert it at the beginning of the result
    if(g.this_exec.index() == 0 && s_flags[0] > 1)
    {
      keys_result[0]   = init_key;
      values_result[0] = init_value;
    }

    size_type result_size = s_flags[n - 1] - 1;

    keys_result    += result_size;
    values_result  += result_size;
    init_key        = keys_first[n-1];
    init_value      = s_values[n - 1];

    g.wait();
  } // end for

#if __CUDA_ARCH__ >= 200
  bulk::free(g, s_flags);
  bulk::free(g, s_values);
#endif

  return thrust::make_tuple(keys_result, values_result, init_key, init_value);
} // end reduce_by_key()


} // end bulk
BULK_NAMESPACE_SUFFIX

