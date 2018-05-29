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
#include <thrust/detail/type_traits.h>
#include <thrust/detail/type_traits/function_traits.h>

#if __cplusplus >= 201103L || defined(__cpp_lib_result_of_sfinae)
// necessary for std::result_of
#include <type_traits>
#endif

namespace thrust
{
namespace detail
{

#if __cplusplus >= 201103L || defined(__cpp_lib_result_of_sfinae)

template<typename Signature>
  struct result_of
{
  using type = typename std::result_of<Signature>::type;
};

#else

template<typename Signature, typename Enable = void> struct result_of;

// specialization for unary invocations of things which have result_type
template<typename Functor, typename Arg1>
  struct result_of_adaptable_function<
    Functor(Arg1),
    typename thrust::detail::enable_if<thrust::detail::has_result_type<Functor>::value>::type
  >
{
  typedef typename Functor::result_type type;
}; // end result_of

// specialization for binary invocations of things which have result_type
template<typename Functor, typename Arg1, typename Arg2>
  struct result_of_adaptable_function<
    Functor(Arg1,Arg2),
    typename thrust::detail::enable_if<thrust::detail::has_result_type<Functor>::value>::type
  >
{
  typedef typename Functor::result_type type;
};

#endif // __cplusplus >= 201103L

} // end detail
} // end thrust

