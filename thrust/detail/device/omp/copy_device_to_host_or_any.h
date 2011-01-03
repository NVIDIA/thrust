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

namespace thrust
{

namespace detail
{

namespace device
{

namespace omp
{

template<typename InputIterator,
         typename OutputIterator>
OutputIterator copy_device_to_host_or_any(InputIterator first,
                                          InputIterator last,
                                          OutputIterator result);

} // end omp

} // end device

} // end detail

} // end thrust

#include <thrust/detail/device/omp/copy_device_to_host_or_any.inl>

