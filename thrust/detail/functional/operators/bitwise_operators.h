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
namespace functional
{

template<typename T1, typename T2>
__host__ __device__
actor<
  composite<
    binary_operator<thrust::bit_and>,
    typename as_actor<T1>::type,
    typename as_actor<T2>::type
  >
>
operator&(const T1 &_1, const T2 &_2)
{
  return compose(binary_operator<thrust::bit_and>(),
                 as_actor<T1>::convert(_1),
                 as_actor<T2>::convert(_2));
} // end operator&()

template<typename T1, typename T2>
__host__ __device__
actor<
  composite<
    binary_operator<thrust::bit_or>,
    typename as_actor<T1>::type,
    typename as_actor<T2>::type
  >
>
operator|(const T1 &_1, const T2 &_2)
{
  return compose(binary_operator<thrust::bit_or>(),
                 as_actor<T1>::convert(_1),
                 as_actor<T2>::convert(_2));
} // end operator|()

template<typename T1, typename T2>
__host__ __device__
actor<
  composite<
    binary_operator<thrust::bit_xor>,
    typename as_actor<T1>::type,
    typename as_actor<T2>::type
  >
>
operator^(const T1 &_1, const T2 &_2)
{
  return compose(binary_operator<thrust::bit_xor>(),
                 as_actor<T1>::convert(_1),
                 as_actor<T2>::convert(_2));
} // end operator^()

// there's no standard bit_not functional, so roll an ad hoc one here
template<typename T>
  struct bit_not
    : public thrust::unary_function<T,T>
{
  __host__ __device__ T operator()(const T &x) const {return ~x;}
}; // end bit_not

template<typename Eval>
__host__ __device__
actor<
  composite<
    unary_operator<bit_not>,
    actor<Eval>
  >
>
__host__ __device__
operator~(const actor<Eval> &_1)
{
  return compose(unary_operator<bit_not>(), _1);
} // end operator~()

// there's no standard bit_lshift functional, so roll an ad hoc one here
template<typename T>
  struct bit_lshift
    : public thrust::binary_function<T,T,T>
{
  __host__ __device__ T operator()(const T &lhs, const T &rhs) const {return lhs << rhs;}
}; // end bit_lshift

template<typename T1, typename T2>
__host__ __device__
actor<
  composite<
    binary_operator<bit_lshift>,
    typename as_actor<T1>::type,
    typename as_actor<T2>::type
  >
>
operator<<(const T1 &_1, const T2 &_2)
{
  return compose(binary_operator<bit_lshift>(),
                 as_actor<T1>::convert(_1),
                 as_actor<T2>::convert(_2));
} // end operator<<()

// there's no standard bit_rshift functional, so roll an ad hoc one here
template<typename T>
  struct bit_rshift
    : public thrust::binary_function<T,T,T>
{
  __host__ __device__ T operator()(const T &lhs, const T &rhs) const {return lhs >> rhs;}
}; // end bit_rshift

template<typename T1, typename T2>
__host__ __device__
actor<
  composite<
    binary_operator<bit_rshift>,
    typename as_actor<T1>::type,
    typename as_actor<T2>::type
  >
>
operator>>(const T1 &_1, const T2 &_2)
{
  return compose(binary_operator<bit_rshift>(),
                 as_actor<T1>::convert(_1),
                 as_actor<T2>::convert(_2));
} // end operator>>()

} // end functional
} // end detail
} // end thrust


