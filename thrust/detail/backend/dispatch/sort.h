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

#include <thrust/iterator/iterator_traits.h>

#include <thrust/detail/backend/cuda/sort.h>
#include <thrust/detail/backend/omp/sort.h>

namespace thrust
{
namespace detail
{
namespace backend
{
namespace dispatch
{

template<typename RandomAccessKeyIterator,
         typename RandomAccessValueIterator,
         typename StrictWeakOrdering>
  void stable_sort_by_key(RandomAccessKeyIterator keys_first,
                          RandomAccessKeyIterator keys_last,
                          RandomAccessValueIterator values_first,
                          StrictWeakOrdering comp,
                          thrust::detail::omp_device_space_tag,
                          thrust::detail::omp_device_space_tag)
{
    // OpenMP implementation
    thrust::detail::backend::omp::stable_sort_by_key(keys_first, keys_last, values_first, comp);
}

template<typename RandomAccessKeyIterator,
         typename RandomAccessValueIterator,
         typename StrictWeakOrdering>
  void stable_sort_by_key(RandomAccessKeyIterator keys_first,
                          RandomAccessKeyIterator keys_last,
                          RandomAccessValueIterator values_first,
                          StrictWeakOrdering comp,
                          thrust::detail::cuda_device_space_tag,
                          thrust::detail::cuda_device_space_tag)
{
    // CUDA implementation
    thrust::detail::backend::cuda::stable_sort_by_key(keys_first, keys_last, values_first, comp);
}

} // end namespace dispatch
} // end namespace backend
} // end namespace detail
} // end namespace thrust

