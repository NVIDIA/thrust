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

#include <thrust/detail/functional/actor.h>
#include <thrust/detail/functional/value.h>

namespace thrust
{
namespace detail
{
namespace functional
{

template<typename T>
  struct as_actor
{
  typedef value<T> type;

  static inline __host__ __device__ type convert(const T &x)
  {
    return val(x);
  } // end convert()
}; // end as_actor

template<typename T> struct as_actor;

template<typename Eval>
  struct as_actor<actor<Eval> >
{
  typedef actor<Eval> type;

  static inline __host__ __device__ const type &convert(const actor<Eval> &x)
  {
    return x;
  } // end convert()
}; // end as_actor

} // end functional
} // end detail
} // end thrust

