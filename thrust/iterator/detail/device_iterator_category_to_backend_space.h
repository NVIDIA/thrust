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

#include <thrust/iterator/detail/backend_iterator_categories.h>
#include <thrust/iterator/detail/backend_iterator_spaces.h>
#include <thrust/detail/type_traits.h>

namespace thrust
{

namespace detail
{

template<typename DeviceCategory>
  struct device_iterator_category_to_backend_space
      // convertible to cuda?
    : eval_if<
        is_convertible<DeviceCategory, thrust::detail::cuda_device_iterator_tag>::value,

        detail::identity_<thrust::detail::cuda_device_space_tag>,

        // convertible to omp?
        eval_if<
          is_convertible<DeviceCategory, thrust::detail::omp_device_iterator_tag>::value,

          detail::identity_<thrust::detail::omp_device_space_tag>,

          // convertible to device_space_tag?
          eval_if<
            is_convertible<DeviceCategory, thrust::device_space_tag>::value,
            
            detail::identity_<thrust::device_space_tag>,

            // unknown space
            detail::identity_<void>
          >
        >
      >
{};

} // end detail

} // end thrust

