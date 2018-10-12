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

// TODO: Optimize for thrust::plus

// TODO: Move into system::cuda

#pragma once

#include <thrust/detail/config.h>
#include <thrust/detail/cpp11_required.h>

#if THRUST_CPP_DIALECT >= 2011

#if THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_NVCC

#include <thrust/system/cuda/config.h>

#include <thrust/system/cuda/detail/async/customization.h>
#include <thrust/system/cuda/detail/reduce.h>
#include <thrust/system/cuda/future.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/distance.h>

#include <type_traits>

THRUST_BEGIN_NS

namespace system { namespace cuda { namespace detail
{

template <
  typename DerivedPolicy
, typename ForwardIt, typename Size, typename T, typename BinaryOp
>
THRUST_RUNTIME_FUNCTION
auto async_reduce_n(
  execution_policy<DerivedPolicy>& policy,
  ForwardIt                        first,
  Size                             n,
  T                                init,
  BinaryOp                         op
) ->
  unique_eager_future<
    T
  , typename thrust::detail::allocator_traits<
      decltype(get_async_device_allocator(policy))
    >::template rebind_traits<T>::pointer
  >
{
  auto const device_alloc = get_async_device_allocator(policy);

  using pointer
    = typename thrust::detail::allocator_traits<decltype(device_alloc)>::
      template rebind_traits<T>::pointer;

  unique_eager_future_promise_pair<T, pointer> fp;

  // Determine temporary device storage requirements.

  size_t tmp_size = 0;
  thrust::cuda_cub::throw_on_error(
    thrust::cuda_cub::cub::DeviceReduce::Reduce(
      NULL
    , tmp_size
    , first
    , reinterpret_cast<T*>(NULL)
    , n
    , op
    , init
    , NULL // Null stream, just for sizing.
    , THRUST_DEBUG_SYNC_FLAG
    )
  , "after reduction sizing"
  );

  // Allocate temporary storage.

  auto content = uninitialized_allocate_unique_n<thrust::detail::uint8_t>(
    device_alloc, sizeof(T) + tmp_size
  );

  // The array was dynamically allocated, so we assume that it's suitably
  // aligned for any type of data. `malloc`/`cudaMalloc`/`new`/`std::allocator`
  // make this guarantee.
  auto const content_ptr = content.get();
  T* const ret_ptr = thrust::detail::aligned_reinterpret_cast<T*>(
    raw_pointer_cast(content_ptr)
  );
  void* const tmp_ptr = static_cast<void*>(
    thrust::raw_pointer_cast(content_ptr + sizeof(T))
  );

  // Set up stream with dependencies.

  cudaStream_t const user_raw_stream = thrust::cuda_cub::stream(policy);

  if (thrust::cuda_cub::default_stream() != user_raw_stream)
  {
    fp = depend_on<T, pointer>(
      [] (decltype(content) const& c)
      {
        return pointer(
          thrust::detail::aligned_reinterpret_cast<T*>(
            raw_pointer_cast(c.get())
          )
        );
      }
    , std::tuple_cat(
        std::make_tuple(
          std::move(content)
        , unique_stream(nonowning, user_raw_stream)
        )
      , extract_dependencies(
          std::move(thrust::detail::derived_cast(policy))
        )
      )
    );
  }
  else
  {
    fp = depend_on<T, pointer>(
      [] (decltype(content) const& c)
      {
        return pointer(
          thrust::detail::aligned_reinterpret_cast<T*>(
            raw_pointer_cast(c.get())
          )
        );
      }
    , std::tuple_cat(
        std::make_tuple(
          std::move(content)
        )
      , extract_dependencies(
          std::move(thrust::detail::derived_cast(policy))
        )
      )
    );
  }

  // Run reduction.
 
  thrust::cuda_cub::throw_on_error(
    thrust::cuda_cub::cub::DeviceReduce::Reduce(
      tmp_ptr
    , tmp_size
    , first
    , ret_ptr
    , n
    , op
    , init
    , fp.future.stream()
    , THRUST_DEBUG_SYNC_FLAG
    )
  , "after reduction launch"
  );

  return std::move(fp.future);
}

}}} // namespace system::cuda::detail

namespace cuda_cub
{

// ADL entry point.
template <
  typename DerivedPolicy
, typename ForwardIt, typename Sentinel, typename T, typename BinaryOp
>
THRUST_RUNTIME_FUNCTION
auto async_reduce(
  execution_policy<DerivedPolicy>& policy,
  ForwardIt                        first,
  Sentinel                         last,
  T                                init,
  BinaryOp                         op
)
THRUST_DECLTYPE_RETURNS(
  thrust::system::cuda::detail::async_reduce_n(
    policy, first, thrust::distance(first, last), init, op
  )
)

} // cuda_cub

THRUST_END_NS

#endif // THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_NVCC

#endif // THRUST_CPP_DIALECT >= 2011

