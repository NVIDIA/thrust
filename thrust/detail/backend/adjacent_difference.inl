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
#include <thrust/iterator/detail/minimum_space.h>

#include <thrust/detail/backend/cpp/adjacent_difference.h>
#include <thrust/detail/backend/cuda/adjacent_difference.h>
#include <thrust/detail/backend/generic/adjacent_difference.h>

namespace thrust
{
namespace detail
{
namespace backend
{
namespace dispatch
{

template <typename InputIterator, typename OutputIterator, typename BinaryFunction, typename Space>
OutputIterator adjacent_difference(InputIterator first, InputIterator last,
                                   OutputIterator result,
                                   BinaryFunction binary_op,
                                   Space)
{
  return thrust::detail::backend::generic::adjacent_difference(first, last, result, binary_op);
}

template <typename InputIterator, typename OutputIterator, typename BinaryFunction>
OutputIterator adjacent_difference(InputIterator first, InputIterator last,
                                   OutputIterator result,
                                   BinaryFunction binary_op,
                                   thrust::detail::cuda_device_space_tag)
{
  return thrust::detail::backend::cuda::adjacent_difference(first, last, result, binary_op);
}

template <typename InputIterator, typename OutputIterator, typename BinaryFunction>
OutputIterator adjacent_difference(InputIterator first, InputIterator last,
                                   OutputIterator result,
                                   BinaryFunction binary_op,
                                   thrust::host_space_tag)
{
  return thrust::detail::backend::cpp::adjacent_difference(first, last, result, binary_op);
}

} // end namespace dispatch

template <typename InputIterator, typename OutputIterator, typename BinaryFunction>
OutputIterator adjacent_difference(InputIterator first, InputIterator last,
                                   OutputIterator result,
                                   BinaryFunction binary_op)
{
  return thrust::detail::backend::dispatch::adjacent_difference(first, last, result, binary_op,
      typename thrust::detail::minimum_space<
        typename thrust::iterator_space<InputIterator>::type,
        typename thrust::iterator_space<OutputIterator>::type
      >::type());
}

} // end namespace backend
} // end namespace detail
} // end namespace thrust

