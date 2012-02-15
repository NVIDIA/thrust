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


/*! \file default_decomposition.h
 *  \brief Return an appropriate decomposition for a given backend.
 */

#pragma once

#include <thrust/iterator/detail/backend_iterator_spaces.h>

#include <thrust/detail/backend/cpp/default_decomposition.h>
#include <thrust/detail/backend/omp/default_decomposition.h>
#include <thrust/detail/backend/cuda/default_decomposition.h>

namespace thrust
{
namespace detail
{
namespace backend
{
namespace dispatch
{

template <typename IndexType>
uniform_decomposition<IndexType> default_decomposition(IndexType n, thrust::host_space_tag)
{
  return thrust::detail::backend::cpp::default_decomposition(n);
}

template <typename IndexType>
uniform_decomposition<IndexType> default_decomposition(IndexType n, thrust::detail::omp_device_space_tag)
{
  return thrust::detail::backend::omp::default_decomposition(n);
}

template <typename IndexType>
uniform_decomposition<IndexType> default_decomposition(IndexType n, thrust::detail::cuda_device_space_tag)
{
  return thrust::detail::backend::cuda::default_decomposition(n);
}

template <typename IndexType>
uniform_decomposition<IndexType> default_decomposition(IndexType n, thrust::any_space_tag)
{
  return thrust::detail::backend::dispatch::default_decomposition(n, thrust::detail::default_device_space_tag());
}

} // end namespace dispatch

template <typename Space, typename IndexType>
uniform_decomposition<IndexType> default_decomposition(IndexType n)
{
  return thrust::detail::backend::dispatch::default_decomposition(n, Space());
}

} // end namespace backend
} // end namespace detail
} // end namespace thrust

