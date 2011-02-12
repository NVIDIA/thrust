/*
 *  Copyright 2008-2011 NVIDIA Corporation
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
#include <thrust/detail/functional/actor.h>
#include <thrust/detail/functional/composite.h>
#include <thrust/detail/functional/operators/operator_adaptors.h>
#include <thrust/functional.h>

namespace thrust
{
namespace detail
{

template<typename Eval>
__host__ __device__
actor<
  composite<
    unary_operator<thrust::negate>,
    actor<Eval>
  >
>
__host__ __device__
operator-(const actor<Eval> &_1)
{
  return compose(unary_operator<thrust::negate>(), _1);
} // end operator-()

// there's no standard unary_plus functional, so roll an ad hoc one here
template<typename T>
  struct unary_plus
    : public thrust::unary_function<T,T>
{
  __host__ __device__ T operator()(const T &x) const {return +x;}
}; // end unary_plus

template<typename Eval>
__host__ __device__
actor<
  composite<
    unary_operator<unary_plus>,
    actor<Eval>
  >
>
operator+(const actor<Eval> &_1)
{
  return compose(unary_operator<unary_plus>(), _1);
} // end operator+()

template<typename Eval1, typename Eval2>
__host__ __device__
actor<
  composite<
    binary_operator<thrust::plus>,
    actor<Eval1>,
    actor<Eval2>
  >
>
operator+(const actor<Eval1> &_1, const actor<Eval2> &_2)
{
  return compose(binary_operator<thrust::plus>(), _1, _2);
} // end operator+()

template<typename Eval1, typename Eval2>
__host__ __device__
actor<
  composite<
    binary_operator<thrust::minus>,
    actor<Eval1>,
    actor<Eval2>
  >
>
operator-(const actor<Eval1> &_1, const actor<Eval2> &_2)
{
  return compose(binary_operator<thrust::minus>(), _1, _2);
} // end operator-()

template<typename Eval1, typename Eval2>
__host__ __device__
actor<
  composite<
    binary_operator<thrust::multiplies>,
    actor<Eval1>,
    actor<Eval2>
  >
>
operator*(const actor<Eval1> &_1, const actor<Eval2> &_2)
{
  return compose(binary_operator<thrust::multiplies>(), _1, _2);
} // end operator*()

template<typename Eval1, typename Eval2>
__host__ __device__
actor<
  composite<
    binary_operator<thrust::divides>,
    actor<Eval1>,
    actor<Eval2>
  >
>
operator/(const actor<Eval1> &_1, const actor<Eval2> &_2)
{
  return compose(binary_operator<thrust::divides>(), _1, _2);
} // end operator/()

template<typename Eval1, typename Eval2>
__host__ __device__
actor<
  composite<
    binary_operator<thrust::modulus>,
    actor<Eval1>,
    actor<Eval2>
  >
>
operator%(const actor<Eval1> &_1, const actor<Eval2> &_2)
{
  return compose(binary_operator<thrust::modulus>(), _1, _2);
} // end operator%()

} // end detail
} // end thrust

