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


/*! \file scan.inl
 *  \brief Inline file for scan.h.
 */

#include <thrust/detail/config.h>
#include <thrust/detail/device/cuda/dispatch/scan.h>
#include <thrust/detail/static_assert.h>

namespace thrust
{
namespace detail
{
namespace device
{
namespace cuda
{

template<typename InputIterator,
         typename OutputIterator,
         typename AssociativeOperator>
  OutputIterator inclusive_scan(InputIterator first,
                                InputIterator last,
                                OutputIterator result,
                                AssociativeOperator binary_op)
{
    // we're attempting to launch a kernel, assert we're compiling with nvcc
    // ========================================================================
    // X Note to the user: If you've found this line due to a compiler error, X
    // X you need to compile your code using nvcc, rather than g++ or cl.exe  X
    // ========================================================================
    THRUST_STATIC_ASSERT( (depend_on_instantiation<InputIterator, THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_NVCC>::value) );

    typedef typename thrust::iterator_value<OutputIterator>::type OutputType;

    // whether to use fast_scan or safe_scan
    // TODO profile this threshold
#if THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_NVCC && CUDA_VERSION >= 3010
    // CUDA 3.1 and higher support non-pod types in statically-allocated __shared__ memory
    static const bool use_fast_scan = sizeof(OutputType) <= 16;
#else    
    // CUDA 3.0 and earlier must use safe_scan for non-pod types
    static const bool use_fast_scan = sizeof(OutputType) <= 16 && thrust::detail::is_pod<OutputType>::value;
#endif

    // XXX WAR nvcc unused variable warning
    (void) use_fast_scan;

    return thrust::detail::device::cuda::dispatch::inclusive_scan
        (first, last, result, binary_op,
         thrust::detail::integral_constant<bool, use_fast_scan>());
}

template<typename InputIterator,
         typename OutputIterator,
         typename T,
         typename AssociativeOperator>
  OutputIterator exclusive_scan(InputIterator first,
                                InputIterator last,
                                OutputIterator result,
                                T init,
                                AssociativeOperator binary_op)
{
    // we're attempting to launch a kernel, assert we're compiling with nvcc
    // ========================================================================
    // X Note to the user: If you've found this line due to a compiler error, X
    // X you need to compile your code using nvcc, rather than g++ or cl.exe  X
    // ========================================================================
    THRUST_STATIC_ASSERT( (depend_on_instantiation<InputIterator, THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_NVCC>::value) );

    typedef typename thrust::iterator_value<OutputIterator>::type OutputType;

    // whether to use fast_scan or safe_scan
    // TODO profile this threshold
#if THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_NVCC && CUDA_VERSION >= 3010
    // CUDA 3.1 and higher support non-pod types in statically-allocated __shared__ memory
    static const bool use_fast_scan = sizeof(OutputType) <= 16;
#else    
    // CUDA 3.0 and earlier must use safe_scan for non-pod types
    static const bool use_fast_scan = sizeof(OutputType) <= 16 && thrust::detail::is_pod<OutputType>::value;
#endif

    // XXX WAR nvcc 3.0 unused variable warning
    (void) use_fast_scan;

    return thrust::detail::device::cuda::dispatch::exclusive_scan
        (first, last, result, init, binary_op,
         thrust::detail::integral_constant<bool, use_fast_scan>());
}

} // end namespace cuda
} // end namespace device
} // end namespace detail
} // end namespace thrust

