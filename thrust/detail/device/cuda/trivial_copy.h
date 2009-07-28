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


/*! \file trivial_copy.h
 *  \brief Device implementations for copying memory between host and device.
 */

#pragma once

#include <cuda_runtime_api.h> // for cudaMemcpy
#include <stdexcept>          // for std::runtime_error
#include <string>

#include <thrust/iterator/iterator_categories.h>
#include <thrust/iterator/iterator_traits.h>

namespace thrust
{

namespace detail
{

namespace device
{

namespace cuda
{

namespace detail
{

inline void checked_cudaMemcpy(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind)
{
    cudaError_t error = cudaMemcpy(dst,src,count,kind);
    if(error)
    {
        throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(error));
    }
} // end checked_cudaMemcpy()

} // end namespace detail

inline void trivial_copy_host_to_device(void *dst, const void *src, size_t count)
{
    detail::checked_cudaMemcpy(dst, src, count, cudaMemcpyHostToDevice);
}

inline void trivial_copy_device_to_host(void *dst, const void *src, size_t count)
{
    detail::checked_cudaMemcpy(dst, src, count, cudaMemcpyDeviceToHost);
}

inline void trivial_copy_device_to_device(void *dst, const void *src, size_t count)
{
    detail::checked_cudaMemcpy(dst, src, count, cudaMemcpyDeviceToDevice);
}

} // end namespace cuda

} // end namespace device

} // end namespace detail

} // end namespace thrust

