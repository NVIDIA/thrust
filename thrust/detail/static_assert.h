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

/*
 * (C) Copyright John Maddock 2000.
 *
 * Distributed under the Boost Software License, Version 1.0.
 * (See accompanying NOTICE file for the complete license)
 *
 * For more information, see http://www.boost.org
 */

#pragma once

#include <thrust/detail/config.h>
#include <thrust/detail/type_traits.h>
#include <thrust/detail/preprocessor.h>

namespace thrust
{

namespace detail
{

template <typename, bool x>
struct depend_on_instantiation
{
  THRUST_INLINE_INTEGRAL_MEMBER_CONSTANT bool value = x;
};

#if THRUST_CPP_DIALECT >= 2017
#  define THRUST_STATIC_ASSERT(B)        static_assert(B)
#else
#  define THRUST_STATIC_ASSERT(B)        static_assert(B, "static assertion failed")
#endif

#define THRUST_STATIC_ASSERT_MSG(B, msg) static_assert(B, msg)

} // namespace detail

} // end namespace thrust


