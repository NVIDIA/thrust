/*
 *  Copyright 2008-2010 NVIDIA Corporation
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
#include <thrust/detail/device/cuda/set_operations.h>
#include <thrust/detail/device/generic/set_operations.h>

namespace thrust
{

namespace detail
{

namespace device
{

namespace dispatch
{

template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator,
         typename StrictWeakOrdering,
         typename Space1,
         typename Space2,
         typename Space3>
  OutputIterator set_intersection(InputIterator1 first1,
                                  InputIterator1 last1,
                                  InputIterator2 first2,
                                  InputIterator2 last2,
                                  OutputIterator result,
                                  StrictWeakOrdering comp,
                                  Space1,
                                  Space2,
                                  Space3)
{
  // generic backend
  return thrust::detail::device::generic::set_intersection(first1,last1,first2,last2,result,comp);
} // end set_intersection()


template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator,
         typename StrictWeakOrdering>
  OutputIterator set_intersection(InputIterator1 first1,
                                  InputIterator1 last1,
                                  InputIterator2 first2,
                                  InputIterator2 last2,
                                  OutputIterator result,
                                  StrictWeakOrdering comp,
                                  thrust::detail::cuda_device_space_tag,
                                  thrust::detail::cuda_device_space_tag,
                                  thrust::detail::cuda_device_space_tag)
{
  // refinement for the CUDA backend
  return thrust::detail::device::cuda::set_intersection(first1,last1,first2,last2,result,comp);
} // end set_intersection()


} // end dispatch

} // end device

} // end detail

} // end thrust

