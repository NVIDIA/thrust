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
#include <thrust/detail/functional/actor.h>
#include <thrust/detail/functional/composite.h>
#include <thrust/detail/functional/operators/operator_adaptors.h>
#include <thrust/functional.h>
#include <thrust/detail/type_traits.h>
#include <thrust/tuple.h>

namespace thrust
{
namespace detail
{
namespace functional
{

// we could use binary_operator<thrust::project2nd>, except
// that project2nd has two template parameters (rather than one)

template<template<typename,typename> class BinaryOperatorWithTwoParms>
  struct binary_operator_with_two_parms
{
  template<typename Env>
    struct first_argument
      : thrust::detail::eval_if<
          (thrust::tuple_size<Env>::value == 0),
          thrust::detail::identity_<thrust::null_type>,
          thrust::tuple_element<0,Env>
        >
  {};

  template<typename Env>
    struct second_argument
      : thrust::detail::eval_if<
          (thrust::tuple_size<Env>::value == 0),
          thrust::detail::identity_<thrust::null_type>,
          thrust::tuple_element<1,Env>
        >
  {};

  template<typename Env>
    struct operator_type
  {
    typedef BinaryOperatorWithTwoParms<
      typename thrust::detail::remove_reference<
        typename first_argument<Env>::type
      >::type,
      typename thrust::detail::remove_reference<
        typename second_argument<Env>::type
      >::type
    > type;
  };

  template<typename Env>
    struct result
  {
    typedef typename operator_type<Env>::type op_type;
    typedef typename op_type::result_type type;
  };

  template<typename Env>
  __host__ __device__
  typename result<Env>::type eval(const Env &e) const
  {
    typename operator_type<Env>::type op;
    return op(thrust::get<0>(e), thrust::get<1>(e));
  } // end eval()
}; // end binary_operator_with_two_parms

template<typename T1, typename T2>
__host__ __device__
actor<
  composite<
    binary_operator_with_two_parms<thrust::project2nd>,
    actor<T1>,
    typename as_actor<T2>::type
  >
>
operator,(const actor<T1> &_1, const T2 &_2)
{
  return compose(binary_operator_with_two_parms<thrust::project2nd>(),
                 make_actor(_1),
                 make_actor(_2));
} // end operator,()

template<typename T1, typename T2>
__host__ __device__
actor<
  composite<
    binary_operator_with_two_parms<thrust::project2nd>,
    typename as_actor<T1>::type,
    actor<T2>
  >
>
operator,(const T1 &_1, const actor<T2> &_2)
{
  return compose(binary_operator_with_two_parms<thrust::project2nd>(),
                 make_actor(_1),
                 make_actor(_2));
} // end operator,()

template<typename T1, typename T2>
__host__ __device__
actor<
  composite<
    binary_operator_with_two_parms<thrust::project2nd>,
    actor<T1>,
    actor<T2>
  >
>
operator,(const T1 &_1, const T2 &_2)
{
  return compose(binary_operator_with_two_parms<thrust::project2nd>(),
                 make_actor(_1),
                 make_actor(_2));
} // end operator,()
  
} // end functional
} // end detail
} // end thrust

