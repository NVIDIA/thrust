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

#include <thrust/detail/backend/for_each.h>
#include <thrust/detail/backend/cpp/for_each.h>
#include <thrust/detail/backend/cuda/for_each.h>
#include <thrust/detail/backend/omp/for_each.h>
#include <thrust/detail/backend/generic/for_each.h>

namespace thrust
{
namespace detail
{
namespace backend
{
namespace dispatch
{


template<typename OutputIterator,
         typename Size,
         typename UnaryFunction>
OutputIterator for_each_n(OutputIterator first,
                          Size n,
                          UnaryFunction f,
                          thrust::detail::omp_device_space_tag)
{
  return thrust::detail::backend::omp::for_each_n(first, n, f);
}

template<typename OutputIterator,
         typename Size,
         typename UnaryFunction>
OutputIterator for_each_n(OutputIterator first,
                          Size n,
                          UnaryFunction f,
                          thrust::detail::cuda_device_space_tag)
{
  return thrust::detail::backend::cuda::for_each_n(first, n, f);
}

template<typename OutputIterator,
         typename Size,
         typename UnaryFunction>
OutputIterator for_each_n(OutputIterator first,
                          Size n,
                          UnaryFunction f,
                          thrust::host_space_tag)
{
  return thrust::detail::backend::cpp::for_each_n(first, n, f);
}

template<typename OutputIterator,
         typename Size,
         typename UnaryFunction>
OutputIterator for_each_n(OutputIterator first,
                          Size n,
                          UnaryFunction f,
                          thrust::any_space_tag)
{
  return thrust::detail::backend::dispatch::for_each_n(first, n, f,
    thrust::detail::default_device_space_tag());
}


template<typename InputIterator,
         typename UnaryFunction,
         typename Space>
InputIterator for_each(InputIterator first,
                       InputIterator last,
                       UnaryFunction f,
                       Space)
{
  return thrust::detail::backend::generic::for_each(first, last, f);
}

template<typename InputIterator,
         typename UnaryFunction>
InputIterator for_each(InputIterator first,
                       InputIterator last,
                       UnaryFunction f,
                       thrust::host_space_tag)
{
  return thrust::detail::backend::cpp::for_each(first, last, f);
}

template<typename InputIterator,
         typename UnaryFunction>
InputIterator for_each(InputIterator first,
                       InputIterator last,
                       UnaryFunction f,
                       thrust::any_space_tag)
{
  return thrust::detail::backend::dispatch::for_each(first, last, f,
    thrust::detail::default_device_space_tag());
}


} // end namespace dispatch

template<typename OutputIterator,
         typename Size,
         typename UnaryFunction>
OutputIterator for_each_n(OutputIterator first,
                          Size n,
                          UnaryFunction f)
{
  return thrust::detail::backend::dispatch::for_each_n(first, n, f,
      typename thrust::iterator_space<OutputIterator>::type());
}

template<typename InputIterator,
         typename UnaryFunction>
InputIterator for_each(InputIterator first,
                       InputIterator last,
                       UnaryFunction f)
{
  return thrust::detail::backend::dispatch::for_each(first, last, f,
      typename thrust::iterator_space<InputIterator>::type());
}

} // end namespace backend
} // end namespace detail
} // end namespace thrust

