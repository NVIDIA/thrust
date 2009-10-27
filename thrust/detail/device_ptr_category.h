/*
 *  Copyright 2008-2009 NVIDIA Corporation
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
#include <thrust/iterator/iterator_traits.h>

namespace thrust
{

namespace detail
{

#if THRUST_DEVICE_BACKEND == CUDA
typedef thrust::detail::random_access_cuda_device_iterator_tag device_ptr_category;
#elif THRUST_DEVICE_BACKEND == OMP
typedef thrust::detail::random_access_omp_device_iterator_tag device_ptr_category;
#else
#error "Unknown device backend."
#endif // THRUST_DEVICE_BACKEND

} // end detail

} // end thrust

