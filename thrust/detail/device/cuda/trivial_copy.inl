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

#include <thrust/detail/config.h>

// do not attempt to compile this code, which relies on 
// CUDART, without system support
#if THRUST_DEVICE_BACKEND == THRUST_DEVICE_BACKEND_CUDA

#include <thrust/detail/device/cuda/trivial_copy.h>
#include <cuda_runtime_api.h>
#include <thrust/system_error.h>
#include <thrust/system/cuda_error.h>

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
    throw thrust::system_error(error, thrust::cuda_category());
  } // end error
} // end checked_cudaMemcpy()


template<typename SrcSpace,
         typename DstSpace>
  struct is_host_to_device
    : integral_constant<
        bool,
        thrust::detail::is_convertible<SrcSpace, thrust::host_space_tag>::value &&
        thrust::detail::is_convertible<DstSpace, thrust::device_space_tag>::value
      >
{};


template<typename SrcSpace,
         typename DstSpace>
  struct is_device_to_host
    : integral_constant<
        bool,
        thrust::detail::is_convertible<SrcSpace, thrust::device_space_tag>::value &&
        thrust::detail::is_convertible<DstSpace, thrust::host_space_tag>::value
      >
{};


template<typename SrcSpace,
         typename DstSpace>
  struct is_device_to_device
    : integral_constant<
        bool,
        thrust::detail::is_convertible<SrcSpace, thrust::device_space_tag>::value &&
        thrust::detail::is_convertible<DstSpace, thrust::device_space_tag>::value
      >
{};


template<typename SrcSpace,
         typename DstSpace>
  struct cuda_memcpy_kind
    : thrust::detail::eval_if<
        is_host_to_device<SrcSpace,DstSpace>::value,
        thrust::detail::integral_constant<cudaMemcpyKind, cudaMemcpyHostToDevice>,

        eval_if<
          is_device_to_host<SrcSpace,DstSpace>::value,
          thrust::detail::integral_constant<cudaMemcpyKind, cudaMemcpyDeviceToHost>,

          eval_if<
            is_device_to_device<SrcSpace,DstSpace>::value,
            thrust::detail::integral_constant<cudaMemcpyKind, cudaMemcpyDeviceToDevice>,
            void
          >
        >
      >::type
{};

} // end namespace detail


template<typename RandomAccessIterator1,
         typename Size,
         typename RandomAccessIterator2>
  void trivial_copy_n(RandomAccessIterator1 first,
                      Size n,
                      RandomAccessIterator2 result)
{
  typedef typename thrust::iterator_value<RandomAccessIterator1>::type T;

  typedef typename thrust::iterator_space<RandomAccessIterator1>::type SrcSpace;
  typedef typename thrust::iterator_space<RandomAccessIterator2>::type DstSpace;

  void *dst = thrust::raw_pointer_cast(&*result);
  const void *src = thrust::raw_pointer_cast(&*first);

  detail::checked_cudaMemcpy(dst, src, n * sizeof(T), detail::cuda_memcpy_kind<SrcSpace, DstSpace>::value);
}


} // end namespace cuda

} // end namespace device

} // end namespace detail

} // end namespace thrust

#endif // THRUST_DEVICE_BACKEND

