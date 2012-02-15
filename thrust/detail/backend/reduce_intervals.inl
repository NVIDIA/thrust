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

#include <thrust/iterator/iterator_traits.h>
#include <thrust/iterator/detail/backend_iterator_spaces.h>

#include <thrust/detail/backend/cpp/reduce_intervals.h>
#include <thrust/detail/backend/omp/reduce_intervals.h>
#include <thrust/detail/backend/cuda/reduce_intervals.h>

namespace thrust
{
namespace detail
{
namespace backend
{
namespace dispatch
{

template <typename InputIterator,
          typename OutputIterator,
          typename BinaryFunction,
          typename Decomposition>
void reduce_intervals(InputIterator input,
                      OutputIterator output,
                      BinaryFunction binary_op,
                      Decomposition decomp,
                      thrust::host_space_tag)
{
  return thrust::detail::backend::cpp::reduce_intervals(input, output, binary_op, decomp);
}

template <typename InputIterator,
          typename OutputIterator,
          typename BinaryFunction,
          typename Decomposition>
void reduce_intervals(InputIterator input,
                      OutputIterator output,
                      BinaryFunction binary_op,
                      Decomposition decomp,
                      thrust::detail::cuda_device_space_tag)
{
  return thrust::detail::backend::cuda::reduce_intervals(input, output, binary_op, decomp);
}

template <typename InputIterator,
          typename OutputIterator,
          typename BinaryFunction,
          typename Decomposition>
void reduce_intervals(InputIterator input,
                      OutputIterator output,
                      BinaryFunction binary_op,
                      Decomposition decomp,
                      thrust::detail::omp_device_space_tag)
{
  return thrust::detail::backend::omp::reduce_intervals(input, output, binary_op, decomp);
}

template <typename InputIterator,
          typename OutputIterator,
          typename BinaryFunction,
          typename Decomposition>
void reduce_intervals(InputIterator input,
                      OutputIterator output,
                      BinaryFunction binary_op,
                      Decomposition decomp,
                      thrust::any_space_tag)
{
  return thrust::detail::backend::dispatch::reduce_intervals(input, output, binary_op, decomp,
      thrust::detail::default_device_space_tag());
}

} // end dispatch


template <typename InputIterator,
          typename OutputIterator,
          typename BinaryFunction,
          typename Decomposition>
void reduce_intervals(InputIterator input,
                      OutputIterator output,
                      BinaryFunction binary_op,
                      Decomposition decomp)
{
  return thrust::detail::backend::dispatch::reduce_intervals
    (input, output, binary_op, decomp,
      typename thrust::detail::minimum_space<
        typename thrust::iterator_space<InputIterator>::type,
        typename thrust::iterator_space<OutputIterator>::type
      >::type());
}

} // end backend
} // end detail
} // end thrust

