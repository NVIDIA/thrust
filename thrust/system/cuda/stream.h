//===----------------------------------------------------------------------===//
//
// Part of Thrust, released under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Author: Bryce Adelstein Lelbach <brycelelbach@gmail.com>
//
//===----------------------------------------------------------------------===//

#pragma once

#include <thrust/detail/config.h>
#include <thrust/detail/type_deduction.h>
#include <thrust/detail/tuple_algorithms.h>
#include <thrust/detail/execution_policy.h>
#include <thrust/system/cuda/error.h>
#include <tuple>
#include <memory>

namespace thrust { namespace system { namespace cuda
{

///////////////////////////////////////////////////////////////////////////////

// TODO: Move me to a new header.
struct nonowning_t final {};

THRUST_INLINE_CONSTANT nonowning_t nonowning{};

///////////////////////////////////////////////////////////////////////////////

struct marker_deleter final
{
  __host__
  void operator()(CUevent_st* e) const
  {
    if (nullptr != e)
      throw_on_error(cudaEventDestroy(e));
  }
};

///////////////////////////////////////////////////////////////////////////////

struct unique_marker final
{
  using native_handle_type = CUevent_st*;

private:
  std::unique_ptr<CUevent_st, marker_deleter> handle_;

public:
  /// \brief Create a new stream and construct a handle to it. When the handle
  ///        is destroyed, the stream is destroyed.
  __host__
  unique_marker()
    : handle_(nullptr, marker_deleter())
  {
    native_handle_type e;
    throw_on_error(
      cudaEventCreateWithFlags(&e, cudaEventDisableTiming)
    );
    handle_.reset(e);
  }

  __thrust_exec_check_disable__
  unique_marker(unique_marker const&) = delete;
  __thrust_exec_check_disable__
  unique_marker(unique_marker&&) = default;
  __thrust_exec_check_disable__
  unique_marker& operator=(unique_marker const&) = delete;
  __thrust_exec_check_disable__
  unique_marker& operator=(unique_marker&&) = default;

  __thrust_exec_check_disable__
  ~unique_marker() = default;

  __host__
  auto get() const
  THRUST_DECLTYPE_RETURNS(native_handle_type(handle_.get()));
  __host__
  auto native_handle() const
  THRUST_DECLTYPE_RETURNS(native_handle_type(handle_.get()));

  __host__
  bool valid() const noexcept { return bool(handle_); }

  __host__
  bool ready() const
  {
    cudaError_t const err = cudaEventQuery(handle_.get());

    if (cudaErrorNotReady == err)
      return false;

    // Throw on any other error.
    throw_on_error(err);

    return true;
  }

  __host__
  void wait() const
  {
    throw_on_error(cudaEventSynchronize(handle_.get()));
  }

  __host__
  bool operator==(unique_marker const& other) const
  {
    return other.handle_ == handle_;
  }

  __host__
  bool operator!=(unique_marker const& other) const
  {
    return !(other == *this);
  }
};

///////////////////////////////////////////////////////////////////////////////

struct stream_deleter final
{
  __host__
  void operator()(CUstream_st* s) const
  {
    if (nullptr != s)
      throw_on_error(cudaStreamDestroy(s));
  }
};

struct stream_conditional_deleter final
{
private:
  bool const cond_;

public:
  __host__
  constexpr stream_conditional_deleter() noexcept
    : cond_(true) {}

  __host__
  explicit constexpr stream_conditional_deleter(nonowning_t) noexcept
    : cond_(false) {}

  __host__
  void operator()(CUstream_st* s) const
  {
    if (cond_ && nullptr != s)
    {
      throw_on_error(cudaStreamDestroy(s));
    }
  }
};

///////////////////////////////////////////////////////////////////////////////

struct unique_stream final
{
  using native_handle_type = CUstream_st*;

private:
  std::unique_ptr<CUstream_st, stream_conditional_deleter> handle_;

public:
  /// \brief Create a new stream and construct a handle to it. When the handle
  ///        is destroyed, the stream is destroyed.
  __host__
  unique_stream()
    : handle_(nullptr, stream_conditional_deleter())
  {
    native_handle_type s;
    throw_on_error(
      cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking)
    );
    handle_.reset(s);
  }

  /// \brief Construct a non-owning handle to an existing stream. When the
  ///        handle is destroyed, the stream is not destroyed.
  __host__
  explicit unique_stream(nonowning_t, native_handle_type handle)
    : handle_(handle, stream_conditional_deleter(nonowning))
  {}

  __thrust_exec_check_disable__
  unique_stream(unique_stream const&) = delete;
  __thrust_exec_check_disable__
  unique_stream(unique_stream&&) = default;
  __thrust_exec_check_disable__
  unique_stream& operator=(unique_stream const&) = delete;
  __thrust_exec_check_disable__
  unique_stream& operator=(unique_stream&&) = default;

  __thrust_exec_check_disable__
  ~unique_stream() = default;

  __host__
  auto get() const
  THRUST_DECLTYPE_RETURNS(native_handle_type(handle_.get()));
  __host__
  auto native_handle() const
  THRUST_DECLTYPE_RETURNS(native_handle_type(handle_.get()));

  __host__
  bool valid() const noexcept { return bool(handle_); }

  __host__
  bool ready() const
  {
    cudaError_t const err = cudaStreamQuery(handle_.get());

    if (cudaErrorNotReady == err)
      return false;

    // Throw on any other error.
    throw_on_error(err);

    return true;
  }

  __host__
  void wait() const
  {
    throw_on_error(
      cudaStreamSynchronize(handle_.get())
    );
  }

  __host__
  void depend_on(unique_marker& e)
  {
    throw_on_error(
      cudaStreamWaitEvent(handle_.get(), e.get(), 0)
    );
  }

  __host__
  void depend_on(unique_stream& s)
  {
    if (s != *this)
    {
      unique_marker e;
      s.record(e);
      depend_on(e);
    }
  }

  __host__
  void record(unique_marker& e)
  {
    throw_on_error(cudaEventRecord(e.get(), handle_.get()));
  }

  __host__
  bool operator==(unique_stream const& other) const
  {
    return other.handle_ == handle_;
  }

  __host__
  bool operator!=(unique_stream const& other) const
  {
    return !(other == *this);
  }
};

///////////////////////////////////////////////////////////////////////////////

__host__ __device__ inline cudaStream_t get_default_raw_stream()
{
#ifdef CUDA_API_PER_THREAD_DEFAULT_STREAM
  return cudaStreamPerThread;
#else
  return cudaStreamLegacy;
#endif
}

///////////////////////////////////////////////////////////////////////////////

namespace detail
{

struct raw_stream_applier
{
  template <typename Head, typename... Tail>
  __host__ cudaStream_t operator()(Head&& head, Tail&&... tail) const
  {
    auto stm = dispatch_get_raw_stream(head);
    if (get_default_raw_stream() != stm)
      return stm;
    return (*this)(THRUST_FORWARD(tail)...);
  }

  __host__ cudaStream_t operator()() const
  {
    return get_default_raw_stream();
  }
};

} // namespace detail

template <typename... Ts>
__host__ cudaStream_t dispatch_get_raw_stream(std::tuple<Ts...> const& t)
{
  return tuple_apply(t, detail::raw_stream_applier{});
}

// Fallback implementation.
template <typename T>
__host__ __device__ cudaStream_t
get_raw_stream(T&&)
{
  return get_default_raw_stream();
}

template <typename Derived>
__host__ __device__ cudaStream_t
get_raw_stream(thrust::detail::execution_policy_base<Derived> const& policy)
{
  return dispatch_get_raw_stream(derived_cast(policy));
}

///////////////////////////////////////////////////////////////////////////////

template <typename Derived>
__host__ __device__ void
synchronize(thrust::detail::execution_policy_base<Derived> const& policy)
{
  auto const stm = get_raw_stream(policy);

  if (THRUST_IS_HOST_CODE) {
    #if THRUST_INCLUDE_HOST_CODE
      throw_on_error(cudaStreamSynchronize(stm));
    #endif
  } else {
    #if THRUST_INCLUDE_DEVICE_CODE && THRUST_HAS_CUDART
      throw_on_error(cudaDeviceSynchronize());
    #endif
  }
}

template <typename Derived>
__host__ __device__ void
synchronize(
  thrust::detail::execution_policy_base<Derived> const& policy, char const* msg
) {
  auto const stm = get_raw_stream(derived_cast(policy));

  if (THRUST_IS_HOST_CODE) {
    #if THRUST_INCLUDE_HOST_CODE
      throw_on_error(cudaStreamSynchronize(stm), msg);
    #endif
  } else {
    #if THRUST_INCLUDE_DEVICE_CODE && THRUST_HAS_CUDART
      throw_on_error(cudaDeviceSynchronize(), msg);
    #endif
  }
}

}} // namespace system::cuda

namespace cuda
{

using system::cuda::nonowning;

using system::cuda::marker_deleter;
using system::cuda::unique_marker;

using system::cuda::stream_deleter;
using system::cuda::stream_conditional_deleter;
using system::cuda::unique_stream;

using system::cuda::get_default_raw_stream;
using system::cuda::get_raw_stream;

using system::cuda::synchronize;

}

namespace cuda_cub
{

using system::cuda::nonowning;

using system::cuda::marker_deleter;
using system::cuda::unique_marker;

using system::cuda::stream_deleter;
using system::cuda::stream_conditional_deleter;
using system::cuda::unique_stream;

using system::cuda::get_default_raw_stream;
using system::cuda::get_raw_stream;

using system::cuda::synchronize;

}

} // namespace thrust


