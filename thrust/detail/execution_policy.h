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


// execution_policy_base serves as a guard against
// inifinite recursion in thrust entry points:
//
// template<typename DerivedPolicy>
// void foo(const thrust::detail::execution_policy_base<DerivedPolicy> &s)
// {
//   using thrust::system::detail::generic::foo;
//
//   foo(thrust::detail::derived_cast(thrust::detail::strip_const(s));
// }
//
// foo is not recursive when
// 1. DerivedPolicy is derived from thrust::execution_policy below
// 2. generic::foo takes thrust::execution_policy as a parameter
template<typename DerivedPolicy> struct execution_policy_base {};


template<typename DerivedPolicy>
__host__ __device__
inline execution_policy_base<DerivedPolicy> &strip_const(const execution_policy_base<DerivedPolicy> &x)
{
  return const_cast<execution_policy_base<DerivedPolicy>&>(x);
}


template<typename DerivedPolicy>
__host__ __device__
inline DerivedPolicy &derived_cast(execution_policy_base<DerivedPolicy> &x)
{
  return static_cast<DerivedPolicy&>(x);
}


template<typename DerivedPolicy>
__host__ __device__
inline const DerivedPolicy &derived_cast(const execution_policy_base<DerivedPolicy> &x)
{
  return static_cast<const DerivedPolicy&>(x);
}


} // end detail


template<typename DerivedPolicy>
  struct execution_policy
    : thrust::detail::execution_policy_base<DerivedPolicy>
{};


} // end thrust

