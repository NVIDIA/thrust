/*
 *  Copyright 2008-2018 NVIDIA Corporation
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

/*! \file async/reduce.h
 *  \brief Functions for asynchronously reducing a range to a single value.
 */

#pragma once

#include <thrust/detail/config.h>
#include <thrust/detail/cpp11_required.h>

#if THRUST_CPP_DIALECT >= 2011

#include <thrust/detail/static_assert.h>
#include <thrust/detail/select_system.h>
#include <thrust/type_traits/logical_metafunctions.h>
#include <thrust/type_traits/remove_cvref.h>
#include <thrust/type_traits/is_execution_policy.h>
#include <thrust/system/detail/adl/async/reduce.h>

#include <thrust/future.h>

THRUST_BEGIN_NS

namespace async
{

namespace unimplemented
{

template <
  typename DerivedPolicy
, typename ForwardIt, typename Sentinel, typename T, typename BinaryOp
>
__host__ __device__
future<T, DerivedPolicy>
async_reduce(
  thrust::execution_policy<DerivedPolicy>&, ForwardIt, Sentinel, T, BinaryOp
)
{
  THRUST_STATIC_ASSERT_MSG(
    (thrust::detail::depend_on_instantiation<ForwardIt, false>::value)
  , "this algorithm is not implemented for the specified system"
  );
  return {};
} 

} // namespace unimplemented

struct reduce_fn final
{
  __thrust_exec_check_disable__
  template <
    typename DerivedPolicy
  , typename ForwardIt, typename Sentinel, typename T, typename BinaryOp
  >
  __host__ __device__
  static future<remove_cvref_t<T>, DerivedPolicy>
  call(
    thrust::detail::execution_policy_base<DerivedPolicy> const& exec
  , ForwardIt&& first, Sentinel&& last
  , T&& init
  , BinaryOp&& op
  )
  {
    // ADL dispatch.
    using thrust::async::unimplemented::async_reduce;
    return async_reduce(
      thrust::detail::derived_cast(thrust::detail::strip_const(exec))
    , THRUST_FWD(first), THRUST_FWD(last)
    , THRUST_FWD(init)
    , THRUST_FWD(op)
    );
  } 

  __thrust_exec_check_disable__
  template <
    typename DerivedPolicy
  , typename ForwardIt, typename Sentinel, typename T
  >
  __host__ __device__
  static future<remove_cvref_t<T>, DerivedPolicy> call(
    thrust::detail::execution_policy_base<DerivedPolicy> const& exec
  , ForwardIt&& first, Sentinel&& last
  , T&& init
  )
  {
    return call(
      exec
    , THRUST_FWD(first), THRUST_FWD(last)
    , THRUST_FWD(init)
    , thrust::plus<remove_cvref_t<T>>{}
    );
  }

  __thrust_exec_check_disable__
  template <
    typename DerivedPolicy
  , typename ForwardIt, typename Sentinel
  >
  __host__ __device__
  static future<
    typename iterator_traits<remove_cvref_t<ForwardIt>>::value_type, DerivedPolicy
  >
  call(
    thrust::detail::execution_policy_base<DerivedPolicy> const& exec
  , ForwardIt&& first, Sentinel&& last
  )
  {
    return call(
      exec
    , THRUST_FWD(first), THRUST_FWD(last)
    , typename iterator_traits<remove_cvref_t<ForwardIt>>::value_type{}
    );
  }

  __thrust_exec_check_disable__
  template <typename ForwardIt, typename Sentinel, typename T, typename BinaryOp>
  __host__ __device__
  static auto call(ForwardIt&& first, Sentinel&& last, T&& init, BinaryOp&& op)
  THRUST_DECLTYPE_RETURNS_WITH_SFINAE_CONDITION(
    (negation<is_execution_policy<remove_cvref_t<ForwardIt>>>::value)
  , call(
      thrust::detail::select_system(
        typename iterator_system<remove_cvref_t<ForwardIt>>::type{}
      )
    , THRUST_FWD(first), THRUST_FWD(last)
    , THRUST_FWD(init)
    , THRUST_FWD(op)
    )
  )

  __thrust_exec_check_disable__
  template <typename ForwardIt, typename Sentinel, typename T>
  __host__ __device__
  static auto call(ForwardIt&& first, Sentinel&& last, T&& init)
  THRUST_DECLTYPE_RETURNS_WITH_SFINAE_CONDITION(
    (negation<is_execution_policy<remove_cvref_t<ForwardIt>>>::value)
  , call(
      THRUST_FWD(first), THRUST_FWD(last)
    , THRUST_FWD(init)
    , thrust::plus<remove_cvref_t<T>>{}
    )
  )

  __thrust_exec_check_disable__
  template <typename ForwardIt, typename Sentinel>
  __host__ __device__
  static auto call(ForwardIt&& first, Sentinel&& last)
  THRUST_DECLTYPE_RETURNS(
    call(
      THRUST_FWD(first), THRUST_FWD(last)
    , typename iterator_traits<remove_cvref_t<ForwardIt>>::value_type{}
    )
  )

  template <typename... Args>
  auto operator()(Args&&... args) const
  THRUST_DECLTYPE_RETURNS(
    call(THRUST_FWD(args)...)
  )
};

THRUST_INLINE_CONSTANT reduce_fn reduce{};

} // namespace async

THRUST_END_NS

#endif // THRUST_CPP_DIALECT >= 2011

