/*
 *  Copyright 2008-2009 NVIDIA Corporation
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

#include <thrust/detail/type_traits.h>
#include <thrust/iterator/iterator_adaptor.h>

namespace thrust
{

namespace experimental
{

namespace detail
{

template <bool, typename Then, typename Else>
  struct eval_if
{
}; // end eval_if

template<typename Then, typename Else>
  struct eval_if<true, Then, Else>
{
  typedef typename Then::type type;
}; // end eval_if

template<typename Then, typename Else>
  struct eval_if<false, Then, Else>
{
  typedef typename Else::type type;
}; // end eval_if

template<typename T>
  struct identity
{
  typedef T type;
}; // end identity

// If T is use_default, return the result of invoking
// DefaultNullaryFn, otherwise return T.
template <class T, class DefaultNullaryFn>
struct ia_dflt_help
  : eval_if<
        thrust::detail::is_same<T, use_default>::value
      , DefaultNullaryFn
      , identity<T>
    >
{
};

} // end detail

} // end experimental

} // end thrust

