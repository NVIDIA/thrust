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

#include <thrust/detail/config.h>
#include <thrust/system/cuda/detail/trivial_copy.h>
#include <thrust/system/cuda/detail/guarded_cuda_runtime_api.h>
#include <thrust/system_error.h>
#include <thrust/system/cuda/error.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/system/cpp/detail/execution_policy.h>
#include <thrust/detail/raw_pointer_cast.h>

namespace thrust
{
namespace system
{
namespace cuda
{
namespace detail
{

namespace trivial_copy_detail
{

inline void checked_cudaMemcpy(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind)
{
  cudaError_t error = cudaMemcpy(dst,src,count,kind);
  if(error)
  {
    throw thrust::system_error(error, thrust::cuda_category());
  } // end error
} // end checked_cudaMemcpy()


template<typename System1,
         typename System2>
  cudaMemcpyKind cuda_memcpy_kind(const thrust::cuda::execution_policy<System1> &,
                                  const thrust::cpp::execution_policy<System2> &)
{
  return cudaMemcpyDeviceToHost;
} // end cuda_memcpy_kind()


template<typename System1,
         typename System2>
  cudaMemcpyKind cuda_memcpy_kind(const thrust::cpp::execution_policy<System1> &,
                                  const thrust::cuda::execution_policy<System2> &)
{
  return cudaMemcpyHostToDevice;
} // end cuda_memcpy_kind()


} // end namespace trivial_copy_detail


template<typename DerivedPolicy,
         typename RandomAccessIterator1,
         typename Size,
         typename RandomAccessIterator2>
  void trivial_copy_n(execution_policy<DerivedPolicy> &exec,
                      RandomAccessIterator1 first,
                      Size n,
                      RandomAccessIterator2 result)
{
  typedef typename thrust::iterator_value<RandomAccessIterator1>::type T;

  void *dst = thrust::raw_pointer_cast(&*result);
  const void *src = thrust::raw_pointer_cast(&*first);

  trivial_copy_detail::checked_cudaMemcpy(dst, src, n * sizeof(T), cudaMemcpyDeviceToDevice);
}


template<typename System1,
         typename System2,
         typename RandomAccessIterator1,
         typename Size,
         typename RandomAccessIterator2>
  void trivial_copy_n(cross_system<System1,System2> &systems,
                      RandomAccessIterator1 first,
                      Size n,
                      RandomAccessIterator2 result)
{
  typedef typename thrust::iterator_value<RandomAccessIterator1>::type T;

  void *dst = thrust::raw_pointer_cast(&*result);
  const void *src = thrust::raw_pointer_cast(&*first);

  cudaMemcpyKind kind = trivial_copy_detail::cuda_memcpy_kind(thrust::detail::derived_cast(systems.system1), thrust::detail::derived_cast(systems.system2));

  trivial_copy_detail::checked_cudaMemcpy(dst, src, n * sizeof(T), kind);
}


} // end namespace detail
} // end namespace cuda
} // end namespace system
} // end namespace thrust

