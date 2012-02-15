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

#pragma once

#include <thrust/iterator/iterator_traits.h>
#include <thrust/iterator/detail/backend_iterator_spaces.h>

#include <thrust/detail/uninitialized_array.h>

#include <thrust/detail/backend/reduce_intervals.h>
#include <thrust/detail/backend/default_decomposition.h>

namespace thrust
{
namespace detail
{
namespace backend
{
namespace generic
{
namespace detail
{

// TODO move this into /detail/backend
// this metafunction passes through a type unless it's any_space_tag,
// in which case it returns default_device_space_tag
template<typename Space>
  struct any_space_to_default_device_space_tag
{
  typedef Space type;
}; // end any_space_to_default_device_space_tag

template<>
  struct any_space_to_default_device_space_tag<thrust::any_space_tag>
{
  typedef thrust::detail::default_device_space_tag type;
}; // end any_space_to_default_device_space_tag

} // end namespace detail

template<typename RandomAccessIterator,
         typename SizeType,
         typename OutputType,
         typename BinaryFunction>
  OutputType reduce_n(RandomAccessIterator first,
                      SizeType n,
                      OutputType init,
                      BinaryFunction binary_op)
{
  typedef typename thrust::iterator_space<RandomAccessIterator>::type PossiblyAnySpace;
  typedef typename detail::any_space_to_default_device_space_tag<PossiblyAnySpace>::type Space;

  // determine first and second level decomposition
  thrust::detail::backend::uniform_decomposition<SizeType> decomp1 = thrust::detail::backend::default_decomposition<Space>(n);
  thrust::detail::backend::uniform_decomposition<SizeType> decomp2(decomp1.size() + 1, 1, 1);

  // allocate storage for the initializer and partial sums
  thrust::detail::uninitialized_array<OutputType,Space> partial_sums(decomp1.size() + 1);
  
  // set first element of temp array to init
  partial_sums[0] = init;
  
  // accumulate partial sums (first level reduction)
  thrust::detail::backend::reduce_intervals(first, partial_sums.begin() + 1, binary_op, decomp1);

  // reduce partial sums (second level reduction)
  thrust::detail::backend::reduce_intervals(partial_sums.begin(), partial_sums.begin(), binary_op, decomp2);

  return partial_sums[0];
} // end reduce_n()

} // end namespace generic
} // end namespace backend
} // end namespace detail
} // end namespace thrust

