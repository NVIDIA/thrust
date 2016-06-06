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

#pragma once

#include <thrust/detail/config.h>
#include <thrust/system/cuda/detail/trivial_copy.h>
#include <thrust/system/cuda/detail/guarded_cuda_runtime_api.h>
#include <thrust/system_error.h>
#include <thrust/system/cuda/error.h>
#include <thrust/system/cuda/detail/throw_on_error.h>
#include <thrust/system/cuda/detail/synchronize.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/system/cpp/detail/execution_policy.h>
#include <thrust/detail/raw_pointer_cast.h>
#include <thrust/functional.h>
#include <thrust/system/cuda/detail/execute_on_stream.h>

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

inline void checked_cudaMemcpyAsync(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream)
{
  cudaError_t error = cudaMemcpyAsync(dst,src,count,kind,stream);
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

template<typename System>
cudaMemcpyKind cuda_memcpy_kind(const thrust::cuda::execution_policy<System> &,
                                const thrust::cuda::execution_policy<System> &)
{
#if defined(_WIN32) && !defined(_WIN64)
  // On Win32 we assume cudaMemcpyDeviceToDevice on copy with cuda::par
  // and raw pointers. This is the only legal option in Win32 with cuda::par policy.
  return cudaMemcpyDeviceToDevice;
#else
  // In 64-bit mode copy with cuda::par can legally accept both host and device raw pointers
  // the memcopy kind will be decided by the CUDA runtime based on UVA space of the pointer.
  return cudaMemcpyDefault;
#endif
} // end cuda_memcpy_kind()

namespace {
// XXX: WAR for clang++ >= 3.7.0
//      (a) warnings (nvbug 200202717) &  (b) errors (nvbug 200204101)
//      (a) Clang issues a warning when the address of a reference is tested for null
//      (b) With -O2 & -O3 clang assumes that the address of a reference is not a null
//      and optimizes conditional stmt as "true", which segfaults when the reference
//      is actually bound to nullptr (for example thrust/detail/reference.inl:155)
template<class T> 
bool is_valid_policy(T const& t)
{
  volatile size_t value = reinterpret_cast<size_t>(&t);
  if (value)
  {
    if (value == 0)
    {
      fprintf(stderr, " clang WAR failed. Terminate.\n");
      std::terminate();
    }
    return true;
  }
  return false;
}
}

template<typename System1,
         typename System2>
cudaStream_t cuda_memcpy_stream(const thrust::cuda::execution_policy<System1> &exec,
                                const thrust::cpp::execution_policy<System2> &)
{
  if (is_valid_policy(exec))
    return stream(derived_cast(exec));
  return legacy_stream();
} // end cuda_memcpy_stream()

template<typename System1,
         typename System2>
cudaStream_t cuda_memcpy_stream(const thrust::cpp::execution_policy<System1> &,
                                const thrust::cuda::execution_policy<System2> &exec)
{
  if (is_valid_policy(exec))
    return stream(derived_cast(exec));
  return legacy_stream();
} // end cuda_memcpy_stream()


template<typename System>
cudaStream_t cuda_memcpy_stream(const thrust::cuda::execution_policy<System> &,
                                const thrust::cuda::execution_policy<System> &exec)
{
  if (is_valid_policy(exec))
    return stream(derived_cast(exec));
  return legacy_stream();
} // end cuda_memcpy_stream()



template<class System>
cudaStream_t cuda_memcpy_stream(const thrust::system::cuda::detail::execute_on_stream &exec,
                                const thrust::cuda::execution_policy<System> &)
{
  if (is_valid_policy(exec))
    return stream(exec);
  return legacy_stream();
} // end cuda_memcpy_stream()





} // end namespace trivial_copy_detail


template<typename DerivedPolicy,
         typename RandomAccessIterator1,
         typename Size,
         typename RandomAccessIterator2>
__host__ __device__
void trivial_copy_n(execution_policy<DerivedPolicy> &exec,
                    RandomAccessIterator1 first,
                    Size n,
                    RandomAccessIterator2 result)
{
  typedef typename thrust::iterator_value<RandomAccessIterator1>::type T;

#ifndef __CUDA_ARCH__
  void *dst = thrust::raw_pointer_cast(&*result);
  const void *src = thrust::raw_pointer_cast(&*first);

  // since the user may have given thrust::cuda::par to thrust::copy explicitly,
  // this copy may be a cross-space copy that has bypassed system dispatch
  // we need to have cudaMemcpyAsync figure out the directionality of the copy dynamically
  // using cudaMemcpyDefault

  cudaMemcpyKind kind = trivial_copy_detail::cuda_memcpy_kind(thrust::detail::derived_cast(exec), thrust::detail::derived_cast(exec));
  trivial_copy_detail::checked_cudaMemcpyAsync(dst, src, n * sizeof(T), kind, stream(thrust::detail::derived_cast(exec)));
#else
  thrust::transform(exec, first, first + n, result, thrust::identity<T>());
#endif
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


  // async host <-> device copy , but synchronize on a user provided stream
  cudaStream_t s = trivial_copy_detail::cuda_memcpy_stream(derived_cast(systems.system1), derived_cast(systems.system2));
  trivial_copy_detail::checked_cudaMemcpyAsync(dst, src, n * sizeof(T), kind, s);
  synchronize(s, "failed synchronize in thrust::system::cuda::detail::trivial_copy_n");
}


} // end namespace detail
} // end namespace cuda
} // end namespace system
} // end namespace thrust

