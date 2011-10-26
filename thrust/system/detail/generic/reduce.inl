/*
 *  Copyright 2008-2011 NVIDIA Corporation
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

#include <thrust/reduce.h>
#include <thrust/distance.h>
#include <thrust/system/detail/generic/reduce.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/functional.h>

#include <thrust/detail/temporary_array.h>

#include <thrust/detail/backend/internal/reduce_intervals.h>
#include <thrust/detail/backend/default_decomposition.h>

namespace thrust
{
namespace system
{
namespace detail
{
namespace generic
{

template<typename InputIterator>
  typename thrust::iterator_traits<InputIterator>::value_type
    reduce(tag, InputIterator first, InputIterator last)
{
  typedef typename thrust::iterator_value<InputIterator>::type InputType;

  // use InputType(0) as init by default
  return thrust::reduce(first, last, InputType(0));
} // end reduce()


template<typename InputIterator, typename T>
  T reduce(tag, InputIterator first, InputIterator last, T init)
{
  // use plus<T> by default
  return thrust::reduce(first, last, init, thrust::plus<T>());
} // end reduce()

namespace detail
{

// TODO move this into /detail/backend
// this metafunction passes through a type unless it's any_space_tag,
// in which case it returns device_space_tag
template<typename Space>
  struct any_space_to_device_space_tag
{
  typedef Space type;
}; // end any_space_to_device_space_tag


template<>
  struct any_space_to_device_space_tag<thrust::any_space_tag>
{
  typedef thrust::device_space_tag type;
}; // end any_space_to_device_space_tag


} // end namespace detail


template<typename RandomAccessIterator,
         typename OutputType,
         typename BinaryFunction>
  OutputType reduce(tag,
                    RandomAccessIterator first,
                    RandomAccessIterator last,
                    OutputType init,
                    BinaryFunction binary_op)
{
  typedef typename thrust::iterator_space<RandomAccessIterator>::type PossiblyAnySpace;
  typedef typename detail::any_space_to_device_space_tag<PossiblyAnySpace>::type Space;

  typedef typename thrust::iterator_difference<RandomAccessIterator>::type difference_type;

  const difference_type n = thrust::distance(first,last);

  // determine first and second level decomposition
  thrust::detail::backend::uniform_decomposition<difference_type> decomp1 = thrust::detail::backend::default_decomposition<Space>(n);
  thrust::detail::backend::uniform_decomposition<difference_type> decomp2(decomp1.size() + 1, 1, 1);

  // allocate storage for the initializer and partial sums
  thrust::detail::temporary_array<OutputType,Space> partial_sums(decomp1.size() + 1);
  
  // set first element of temp array to init
  partial_sums[0] = init;
  
  // accumulate partial sums (first level reduction)
  thrust::detail::backend::internal::reduce_intervals(first, partial_sums.begin() + 1, binary_op, decomp1);

  // reduce partial sums (second level reduction)
  thrust::detail::backend::internal::reduce_intervals(partial_sums.begin(), partial_sums.begin(), binary_op, decomp2);

  return partial_sums[0];
} // end reduce()


} // end namespace generic
} // end namespace detail
} // end namespace system
} // end namespace thrust

