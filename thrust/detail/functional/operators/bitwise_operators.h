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

template<typename Eval1, typename Eval2>
__host__ __device__
actor<
  composite<
    binary_operator<thrust::bit_and>,
    actor<Eval1>,
    actor<Eval2>
  >
>
operator&(const actor<Eval1> &_1, const actor<Eval2> &_2)
{
  return compose(binary_operator<thrust::bit_and>(), _1, _2);
} // end operator&()

template<typename Eval1, typename Eval2>
__host__ __device__
actor<
  composite<
    binary_operator<thrust::bit_or>,
    actor<Eval1>,
    actor<Eval2>
  >
>
operator|(const actor<Eval1> &_1, const actor<Eval2> &_2)
{
  return compose(binary_operator<thrust::bit_or>(), _1, _2);
} // end operator|()

template<typename Eval1, typename Eval2>
__host__ __device__
actor<
  composite<
    binary_operator<thrust::bit_xor>,
    actor<Eval1>,
    actor<Eval2>
  >
>
operator^(const actor<Eval1> &_1, const actor<Eval2> &_2)
{
  return compose(binary_operator<thrust::bit_xor>(), _1, _2);
} // end operator^()

} // end detail
} // end thrust


