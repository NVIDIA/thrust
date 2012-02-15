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

namespace thrust
{

struct any_space_tag
{
  // use conversion operators instead of inheritance to avoid ambiguous conversion errors
  operator host_space_tag () {return host_space_tag();};

  operator device_space_tag () {return device_space_tag();};

  operator detail::cuda_device_space_tag () {return detail::cuda_device_space_tag();};

  operator detail::omp_device_space_tag () {return detail::omp_device_space_tag();};
};

} // end thrust

