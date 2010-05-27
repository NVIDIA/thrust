/*
 *  Copyright 2008-2010 NVIDIA Corporation
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

#include <thrust/range/algorithm/reduce.h>
#include <thrust/functional.h>
#include <thrust/reduce.h>
#include <thrust/range/detail/value_type.h>
#include <thrust/range/begin.h>
#include <thrust/range/end.h>

namespace thrust
{

namespace experimental
{

namespace range
{

template<typename SinglePassRange>
  typename thrust::experimental::range_value<SinglePassRange>::type
    reduce(const SinglePassRange &rng)
{
  return thrust::experimental::range::reduce(rng, typename range_value<SinglePassRange>::type(0));
} // end reduce()


template<typename SinglePassRange, typename T>
  T reduce(const SinglePassRange &rng,
           T init)
{
  return thrust::experimental::range::reduce(rng, init, thrust::plus<T>());
} // end reduce()


template<typename SinglePassRange, typename T, typename BinaryFunction>
  T reduce(const SinglePassRange &rng,
           T init,
           BinaryFunction binary_op)
{
  return thrust::reduce(begin(rng), end(rng), init, binary_op);
} // end reduce()

} // end range

} // end experimental

} // end thrust

