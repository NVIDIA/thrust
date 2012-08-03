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


/*! \file runtime_introspection.h
 *  \brief Defines the interface to functions
 *         providing introspection into the architecture
 *         of CUDA devices.
 */

#pragma once

#include <thrust/detail/config.h>

// #include this for device_properties_t and function_attributes_t
#include <thrust/system/cuda/detail/cuda_launch_config.h>

// #include this for size_t
#include <cstddef>

namespace thrust
{
namespace system
{
namespace cuda
{
namespace detail
{


/*! Returns the current device ordinal.
 */
inline int current_device();

/*! Returns a copy of the device_properties_t structure
 *  that is associated with a given device.
 */
inline device_properties_t device_properties(int device_id);

/*! Returns a copy of the device_properties_t structure
 *  that is associated with the current device.
 */
inline device_properties_t device_properties(void);

/*! Returns a copy of the function_attributes_t structure
 *  that is associated with a given __global__ function
 */
template <typename KernelFunction>
inline function_attributes_t function_attributes(KernelFunction kernel);

/*! Returns the compute capability of a device in integer format.
 *  For example, returns 10 for sm_10 and 21 for sm_21
 *  \return The compute capability as an integer
 */
inline size_t compute_capability(const device_properties_t &properties);
inline size_t compute_capability(void);


} // end namespace detail
} // end namespace cuda
} // end namespace system
} // end namespace thrust

#include <thrust/system/cuda/detail/runtime_introspection.inl>

