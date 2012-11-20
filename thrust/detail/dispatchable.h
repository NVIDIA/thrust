/*
 *  Copyright 2008-2012 NVIDIA Corporation
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

namespace thrust
{
namespace detail
{


// dispatchable_base serves as a guard against
// inifinite recursion in thrust entry points:
//
// template<typename System>
// void foo(const thrust::detail::dispatchable_base<System> &s)
// {
//   using thrust::system::detail::generic::foo;
//
//   foo(thrust::detail::derived_cast(thrust::detail::strip_const(s));
// }
//
// foo is not recursive when
// 1. System is derived from thrust::dispatchable below
// 2. generic::foo takes thrust::dispatchable as a parameter
template<typename Derived> struct dispatchable_base {};


template<typename Derived>
__host__ __device__
inline dispatchable_base<Derived> &strip_const(const dispatchable_base<Derived> &x)
{
  return const_cast<dispatchable_base<Derived>&>(x);
}


template<typename Derived>
__host__ __device__
inline Derived &derived_cast(dispatchable_base<Derived> &x)
{
  return static_cast<Derived&>(x);
}


template<typename Derived>
__host__ __device__
inline const Derived &derived_cast(const dispatchable_base<Derived> &x)
{
  return static_cast<const Derived&>(x);
}


} // end detail


template<typename Derived>
  struct dispatchable
    : thrust::detail::dispatchable_base<Derived>
{};


} // end thrust

