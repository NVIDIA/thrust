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

#include <thrust/detail/config.h>
#include <thrust/iterator/detail/backend_iterator_categories.h>

namespace thrust
{

template<typename> class device_ptr;

namespace detail
{


// XXX device_ptr_category needs to be f(default_device_space_tag, random_access_traversal_tag)
#if   THRUST_DEVICE_BACKEND == THRUST_DEVICE_BACKEND_CUDA
typedef thrust::detail::random_access_cuda_device_iterator_tag device_ptr_category;
#elif THRUST_DEVICE_BACKEND == THRUST_DEVICE_BACKEND_OMP
typedef thrust::detail::random_access_omp_device_iterator_tag device_ptr_category;
#else
#error "Unknown device backend."
#endif // THRUST_DEVICE_BACKEND


template<typename T>
  struct device_ptr_traits
{
  typedef thrust::detail::device_ptr_category        iterator_category;
  typedef typename detail::remove_const<T>::type     value_type;
  typedef std::ptrdiff_t                             difference_type;
  typedef device_ptr<T>                              pointer;
  typedef device_reference<T>                        reference;
}; // end device_ptr_traits


template<>
  struct device_ptr_traits<void>
{
  typedef thrust::detail::device_ptr_category        iterator_category;
  typedef void                                       value_type;
  typedef std::ptrdiff_t                             difference_type;
  typedef device_ptr<void>                           pointer;
  typedef void                                       reference;
}; // end device_ptr_traits


} // end detail
} // end thrust

