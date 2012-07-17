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

#include <thrust/reduce.h>
#include <thrust/system/detail/generic/reduce.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/functional.h>
#include <thrust/detail/static_assert.h>

namespace thrust
{
namespace system
{
namespace detail
{
namespace generic
{


template<typename System, typename InputIterator>
  typename thrust::iterator_traits<InputIterator>::value_type
    reduce(thrust::dispatchable<System> &system, InputIterator first, InputIterator last)
{
  typedef typename thrust::iterator_value<InputIterator>::type InputType;

  // use InputType(0) as init by default
  return thrust::reduce(system, first, last, InputType(0));
} // end reduce()


template<typename System, typename InputIterator, typename T>
  T reduce(thrust::dispatchable<System> &system, InputIterator first, InputIterator last, T init)
{
  // use plus<T> by default
  return thrust::reduce(system, first, last, init, thrust::plus<T>());
} // end reduce()


template<typename System,
         typename RandomAccessIterator,
         typename OutputType,
         typename BinaryFunction>
  OutputType reduce(thrust::dispatchable<System> &system,
                    RandomAccessIterator first,
                    RandomAccessIterator last,
                    OutputType init,
                    BinaryFunction binary_op)
{
  // unimplemented
  THRUST_STATIC_ASSERT( (thrust::detail::depend_on_instantiation<RandomAccessIterator, false>::value) );
  return OutputType();
} // end reduce()


} // end namespace generic
} // end namespace detail
} // end namespace system
} // end namespace thrust

