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

#include <thrust/range/detail/value_type.h>

namespace thrust
{

namespace experimental
{

namespace range
{


template<typename SinglePassRange>
  typename thrust::experimental::range_value<SinglePassRange>::type
    reduce(const SinglePassRange &rng);


template<typename SinglePassRange, typename T>
  T reduce(const SinglePassRange &rng,
           T init);


template<typename SinglePassRange, typename T, typename BinaryFunction>
  T reduce(const SinglePassRange &rng,
           T init,
           BinaryFunction binary_op);


} // end range

} // end experimental

} // end thrust

#include <thrust/range/algorithm/detail/reduce.inl>

