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

#include <thrust/range/algorithm/count.h>
#include <thrust/count.h>
#include <thrust/range/begin.h>
#include <thrust/range/end.h>


namespace thrust
{

namespace experimental
{

namespace range
{


template<typename SinglePassRange, typename EqualityComparable>
  typename thrust::experimental::range_difference<SinglePassRange>::type
    count(const SinglePassRange &rng, const EqualityComparable &value)
{
  return thrust::count(begin(rng), end(rng), value);
} // end count()


template<typename SinglePassRange, typename Predicate>
  typename thrust::experimental::range_difference<SinglePassRange>::type
    count_if(const SinglePassRange &rng, Predicate pred)
{
  return thrust::count_if(begin(rng), end(rng), pred);
} // end count_if()


} // end range

} // end experimental

} // end thrust

