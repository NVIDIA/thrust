
/******************************************************************************
 * Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/
#pragma once


#if THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_NVCC

#include <thrust/advance.h>
#include <thrust/detail/raw_pointer_cast.h>
#include <thrust/distance.h>
#include <thrust/system/cuda/config.h>
#include <thrust/system/cuda/detail/execution_policy.h>
#include <thrust/system/cuda/detail/transform.h>
#include <thrust/functional.h>

THRUST_BEGIN_NS
namespace cuda_cub {

namespace __copy {

template <typename Derived, typename InputIt, typename OutputIt>
OutputIt THRUST_RUNTIME_FUNCTION
device_to_device(execution_policy<Derived>& policy,
                 InputIt first,
                 InputIt last,
                 OutputIt result,
                 thrust::false_type /* is trivial */)
{
  typedef typename thrust::iterator_traits<InputIt>::value_type InputTy;
  return cuda_cub::transform(policy,
                             first,
                             last,
                             result,
                             thrust::identity<InputTy>());
}

template <typename Derived, typename InputIt, typename OutputIt>
OutputIt THRUST_RUNTIME_FUNCTION
device_to_device(execution_policy<Derived>& policy,
                 InputIt first,
                 InputIt last,
                 OutputIt result,
                 thrust::true_type /* is trivial */)
{
  typedef typename thrust::iterator_traits<InputIt>::difference_type diff_t;
  const diff_t size = std::distance(first, last);
  const cudaError_t status = thrust::cuda_cub::trivial_copy_device_to_device(
    policy,
    thrust::raw_pointer_cast(&*result),
    thrust::raw_pointer_cast(&*first),
    size);
  thrust::cuda_cub::throw_on_error(status);
  thrust::advance(result, size);
  return result;
}

template <class Derived, class InputIt, class OutputIt>
OutputIt THRUST_RUNTIME_FUNCTION
device_to_device(execution_policy<Derived>& policy,
                 InputIt first,
                 InputIt last,
                 OutputIt result)
{
  typedef
    typename thrust::is_indirectly_trivially_relocatable_to<InputIt,
                                                            OutputIt>::type
      is_trivial;
  return cuda_cub::__copy::device_to_device(policy,
                                            first,
                                            last,
                                            result,
                                            is_trivial());
}

}    // namespace __copy

}    // namespace cuda_cub
THRUST_END_NS
#endif
