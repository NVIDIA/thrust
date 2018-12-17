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

/*! \file async/sort.h
 *  \brief Functions for asynchronously sorting a range.
 */

#pragma once

#include <thrust/detail/config.h>
#include <thrust/detail/cpp11_required.h>
#include <thrust/detail/modern_gcc_required.h>

#if THRUST_CPP_DIALECT >= 2011 && !defined(THRUST_LEGACY_GCC)

#include <thrust/detail/static_assert.h>
#include <thrust/detail/select_system.h>
#include <thrust/type_traits/logical_metafunctions.h>
#include <thrust/type_traits/remove_cvref.h>
#include <thrust/type_traits/is_execution_policy.h>
#include <thrust/system/detail/adl/async/sort.h>

#include <thrust/event.h>

THRUST_BEGIN_NS

namespace async
{

namespace unimplemented
{

template <
  typename DerivedPolicy
, typename ForwardIt, typename Sentinel, typename StrictWeakOrdering
>
__host__ 
event<DerivedPolicy>
async_stable_sort(
  thrust::execution_policy<DerivedPolicy>& 
, ForwardIt, Sentinel, StrictWeakOrdering
)
{
  THRUST_STATIC_ASSERT_MSG(
    (thrust::detail::depend_on_instantiation<ForwardIt, false>::value)
  , "this algorithm is not implemented for the specified system"
  );
  return {};
} 

} // namespace unimplemented

namespace stable_sort_detail
{

using thrust::async::unimplemented::async_stable_sort;

struct stable_sort_fn final
{
  template <
    typename DerivedPolicy
  , typename ForwardIt, typename Sentinel, typename StrictWeakOrdering
  >
  __host__ 
  static auto call(
    thrust::detail::execution_policy_base<DerivedPolicy> const& exec
  , ForwardIt&& first, Sentinel&& last
  , StrictWeakOrdering&& comp
  )
  // ADL dispatch.
  THRUST_DECLTYPE_RETURNS(
    async_stable_sort(
      thrust::detail::derived_cast(thrust::detail::strip_const(exec))
    , THRUST_FWD(first), THRUST_FWD(last)
    , THRUST_FWD(comp)
    )
  )

  template <
    typename DerivedPolicy
  , typename ForwardIt, typename Sentinel
  >
  __host__ 
  static auto call(
    thrust::detail::execution_policy_base<DerivedPolicy> const& exec
  , ForwardIt&& first, Sentinel&& last
  )
  // ADL dispatch.
  THRUST_DECLTYPE_RETURNS(
    async_stable_sort(
      thrust::detail::derived_cast(thrust::detail::strip_const(exec))
    , THRUST_FWD(first), THRUST_FWD(last)
    , thrust::less<
        typename iterator_traits<remove_cvref_t<ForwardIt>>::value_type
      >{}
    )
  )

  template <typename ForwardIt, typename Sentinel, typename StrictWeakOrdering>
  __host__ 
  static auto call(ForwardIt&& first, Sentinel&& last, StrictWeakOrdering&& comp) 
  THRUST_DECLTYPE_RETURNS(
    stable_sort_fn::call(
      thrust::detail::select_system(
        typename iterator_system<remove_cvref_t<ForwardIt>>::type{}
      )
    , THRUST_FWD(first), THRUST_FWD(last)
    , THRUST_FWD(comp)
    )
  )

  template <typename ForwardIt, typename Sentinel>
  __host__ 
  static auto call(ForwardIt&& first, Sentinel&& last) 
  THRUST_DECLTYPE_RETURNS(
    stable_sort_fn::call(
      THRUST_FWD(first), THRUST_FWD(last)
    , thrust::less<
        typename iterator_traits<remove_cvref_t<ForwardIt>>::value_type
      >{}
    )
  )

  template <typename... Args>
  THRUST_NODISCARD __host__ 
  auto operator()(Args&&... args) const
  THRUST_DECLTYPE_RETURNS(
    call(THRUST_FWD(args)...)
  )
};

} // namespace stable_sort_detail

THRUST_INLINE_CONSTANT stable_sort_detail::stable_sort_fn stable_sort{};

namespace fallback
{

template <
  typename DerivedPolicy
, typename ForwardIt, typename Sentinel, typename StrictWeakOrdering
>
__host__ 
event<DerivedPolicy>
async_sort(
  thrust::execution_policy<DerivedPolicy>& exec
, ForwardIt&& first, Sentinel&& last, StrictWeakOrdering&& comp
)
{
  return async_stable_sort(
    thrust::detail::derived_cast(exec)
  , THRUST_FWD(first), THRUST_FWD(last), THRUST_FWD(comp)
  );
} 

} // namespace fallback

namespace sort_detail
{

using thrust::async::fallback::async_sort;

struct sort_fn final
{
  template <
    typename DerivedPolicy
  , typename ForwardIt, typename Sentinel, typename StrictWeakOrdering
  >
  __host__ 
  static auto call(
    thrust::detail::execution_policy_base<DerivedPolicy> const& exec
  , ForwardIt&& first, Sentinel&& last
  , StrictWeakOrdering&& comp
  )
  // ADL dispatch.
  THRUST_DECLTYPE_RETURNS(
    async_sort(
      thrust::detail::derived_cast(thrust::detail::strip_const(exec))
    , THRUST_FWD(first), THRUST_FWD(last)
    , THRUST_FWD(comp)
    )
  )

  template <
    typename DerivedPolicy
  , typename ForwardIt, typename Sentinel
  >
  __host__ 
  static auto call(
    thrust::detail::execution_policy_base<DerivedPolicy> const& exec
  , ForwardIt&& first, Sentinel&& last
  )
  THRUST_DECLTYPE_RETURNS(
    sort_fn::call(
      exec
    , THRUST_FWD(first), THRUST_FWD(last)
    , thrust::less<
        typename iterator_traits<remove_cvref_t<ForwardIt>>::value_type
      >{}
    )
  )

  template <typename ForwardIt, typename Sentinel, typename StrictWeakOrdering>
  __host__ 
  static auto call(ForwardIt&& first, Sentinel&& last, StrictWeakOrdering&& comp) 
  THRUST_DECLTYPE_RETURNS_WITH_SFINAE_CONDITION(
    (negation<is_execution_policy<remove_cvref_t<ForwardIt>>>::value)
  , sort_fn::call(
      thrust::detail::select_system(
        typename iterator_system<remove_cvref_t<ForwardIt>>::type{}
      )
    , THRUST_FWD(first), THRUST_FWD(last)
    , THRUST_FWD(comp)
    )
  )

  template <typename ForwardIt, typename Sentinel>
  __host__ 
  static auto call(ForwardIt&& first, Sentinel&& last) 
  THRUST_DECLTYPE_RETURNS(
    sort_fn::call(
      thrust::detail::select_system(
        typename iterator_system<remove_cvref_t<ForwardIt>>::type{}
      )
    , THRUST_FWD(first), THRUST_FWD(last)
    , thrust::less<
        typename iterator_traits<remove_cvref_t<ForwardIt>>::value_type
      >{}
    )
  )

  template <typename... Args>
  THRUST_NODISCARD __host__ 
  auto operator()(Args&&... args) const
  THRUST_DECLTYPE_RETURNS(
    call(THRUST_FWD(args)...)
  )
};

} // namespace sort_detail

THRUST_INLINE_CONSTANT sort_detail::sort_fn sort{};

} // namespace async

THRUST_END_NS

#endif

