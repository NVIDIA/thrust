/******************************************************************************
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

#include <thrust/detail/config.h>

#if THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_NVCC
#include <thrust/system/cuda/detail/copy.h>
#include <thrust/detail/temporary_array.h>

THRUST_NAMESPACE_BEGIN

namespace cuda_cub {


template<typename Derived,
         typename ForwardIterator>
__host__ __device__
  ForwardIterator shift_left(execution_policy<Derived> &policy,
                             ForwardIterator first,
                             ForwardIterator last,
                             typename thrust::iterator_traits<ForwardIterator>::difference_type n)
{
  if (n <= 0) {
    return last;
  }

  const auto len = thrust::distance(first, last);
  if (n >= len) {
    return first;
  }

  ForwardIterator new_first = first;
  thrust::advance(new_first, n)

  thrust::detail::temporary_array<thrust::value_t<ForwardIterator>, Derived> tmp{policy, new_first, last};
  return cuda_cub::copy(policy, tmp.begin(), tmp.end(), first);
}

template<typename Derived,
         typename ForwardIterator>
__host__ __device__
  ForwardIterator shift_right(execution_policy<Derived> &policy,
                             ForwardIterator first,
                             ForwardIterator last,
                             typename thrust::iterator_traits<ForwardIterator>::difference_type n)
{
  if (n <= 0) {
    return first;
  }

  const auto len = thrust::distance(first, last);
  if (n >= len) {
    return last;
  }

  ForwardIterator new_last = first;
  thrust::advance(new_last, len - n)

  thrust::detail::temporary_array<thrust::value_t<ForwardIterator>, Derived> tmp{policy, first, new_last};

  thrust::advance(first, n)
  cuda_cub::copy(policy, tmp.begin(), tmp.end(), first);

  return first;
}

} // namespace cuda_cub

THRUST_NAMESPACE_END
#endif
