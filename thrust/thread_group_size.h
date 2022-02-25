/*
 *  Copyright 2022 NVIDIA Corporation
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
#include <thrust/detail/cpp11_required.h>

#include <thrust/detail/type_traits/has_nested_type.h>

#include <type_traits>

THRUST_NAMESPACE_BEGIN

/*! \brief \c callable_with_thread_group_size is a
 *  <a href="https://en.cppreference.com/w/cpp/named_req/Callable">Callable</a>
 *  adaptor which adds compile-time information about the thread group size
 *  that should be used when parallelizing invocations of the underlying
 *  <tt>Callable</tt>.
 */
template <int ThreadGroupSize, typename Callable>
struct callable_with_thread_group_size
{
  /*! \brief Construct a \c callable_with_thread_group_size object that will
   *  invoke \c c.
   */
  __host__ __device__
  explicit callable_with_thread_group_size(Callable&& c) : call((Callable&&)c){}

  /*! \brief Equivalent to <tt>call(args...)</tt>.
   */
  __thrust_exec_check_disable__
  template <typename... Args>
  __host__ __device__
  bool operator()(Args&&... args) { return call((Args&&)args...); }


  using thread_group_size = std::integral_constant<int, ThreadGroupSize>;

  /*! \cond
   */
  Callable call;
  /*! \endcond
   */
};

/*! \brief Create a
 *  <a href="https://en.cppreference.com/w/cpp/named_req/Callable">Callable</a>
 *  that wraps another <tt>Callable</tt> (\c c) and adds compile-time
 *  information about the thread group size that should be used when
 *  parallelizing invocations of \c c.
 */
template <int ThreadGroupSize, typename Callable>
__host__ __device__
callable_with_thread_group_size<ThreadGroupSize, Callable>
with_thread_group_size(Callable&& c)
{
  return callable_with_thread_group_size<ThreadGroupSize, Callable>((Callable&&)c);
}

/*! \cond
 */

namespace detail {

__THRUST_DEFINE_HAS_NESTED_TYPE(has_thread_group_size, thread_group_size)

}

/*! \endcond
 */

/*! \brief <a href="https://en.cppreference.com/w/cpp/types/integral_constant"><tt>std::integral_constant</tt></a>
/*! whose value is the thread group size that should be used when parallelizing
 *  invocations of the specified
 *  <a href="https://en.cppreference.com/w/cpp/named_req/Callable">Callable</a>
 */
template <typename Callable, typename = void>
struct thread_group_size : std::integral_constant<int, 256> {};

template <typename Callable>
struct thread_group_size<
  Callable
, typename std::enable_if<detail::has_thread_group_size<Callable>::value, void>::type
> : std::integral_constant<int, Callable::thread_group_size::value> {};

THRUST_NAMESPACE_END

