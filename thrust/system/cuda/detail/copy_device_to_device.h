/*
 *  Copyright 2008-2013 NVIDIA Corporation
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


/*! \file copy_device_to_device.h
 *  \brief Device implementations for copying on the device.
 */

#pragma once

#include <thrust/detail/config.h>
#include <thrust/system/cuda/detail/execution_policy.h>

namespace thrust
{
namespace system
{
namespace cuda
{
namespace detail
{


template<typename DerivedPolicy,
         typename InputIterator,
         typename OutputIterator>
__host__ __device__
OutputIterator copy_device_to_device(execution_policy<DerivedPolicy> &exec,
                                     InputIterator begin, 
                                     InputIterator end, 
                                     OutputIterator result);

} // end namespace detail
} // end namespace cuda
} // end namespace system
} // end namespace thrust

#include <thrust/system/cuda/detail/copy_device_to_device.inl>

