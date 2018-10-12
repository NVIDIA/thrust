// Copyright (c) 2018 NVIDIA Corporation
// Author: Bryce Adelstein Lelbach <brycelelbach@gmail.com>
//
// Distributed under the Boost Software License v1.0 (boost.org/LICENSE_1_0.txt)

#pragma once

#include <thrust/detail/config.h>
#include <thrust/detail/cpp11_required.h>

#if THRUST_CPP_DIALECT >= 2011

#include <thrust/system/cuda/pointer.h>
#include <thrust/system/cuda/detail/execution_policy.h>

#include <thrust/future.h>

THRUST_BEGIN_NS

namespace system { namespace cuda
{

template <typename T>
struct ready_future;

template <typename T, typename Pointer = pointer<T>>
struct unique_eager_future;

}} // namespace system::cuda

namespace cuda
{

template <typename T>
using ready_future = thrust::system::cuda::ready_future<T>;

template <typename T, typename Pointer = thrust::system::cuda::pointer<T>>
using unique_eager_future = thrust::system::cuda::unique_eager_future<T, Pointer>;

} // namespace cuda

template <typename T, typename Pointer, typename DerivedPolicy>
__host__ __device__
thrust::system::cuda::unique_eager_future<T, Pointer>
unique_eager_future_type(thrust::cuda_cub::execution_policy<DerivedPolicy> const&);

THRUST_END_NS

#include <thrust/system/cuda/detail/future.inl>

#endif // THRUST_CPP_DIALECT >= 2011

