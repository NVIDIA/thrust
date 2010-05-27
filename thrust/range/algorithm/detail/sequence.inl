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

#include <thrust/sequence.h>
#include <thrust/range/begin.h>
#include <thrust/range/end.h>
#include <thrust/range/algorithm/detail/sequence_result.h>
#include <thrust/iterator/counting_iterator.h>
#include <limits>

namespace thrust
{

namespace experimental
{

namespace range
{


template<typename ForwardRange>
  inline typename detail::sequence_result<ForwardRange>::type
    sequence(ForwardRange &rng)
{
  return thrust::sequence(begin(rng), end(rng));
} // end sequence()


template<typename ForwardRange>
  inline typename detail::sequence_result<const ForwardRange>::type
    sequence(const ForwardRange &rng)
{
  return thrust::sequence(begin(rng), end(rng));
} // end sequence()


template<typename ForwardRange, typename T>
  inline typename detail::sequence_result<ForwardRange>::type
    sequence(ForwardRange &rng, T init)
{
  return thrust::sequence(begin(rng), end(rng), init);
} // end sequence()


template<typename ForwardRange, typename T>
  inline typename detail::sequence_result<ForwardRange>::type
    sequence(const ForwardRange &rng, T init)
{
  return thrust::sequence(begin(rng), end(rng), init);
} // end sequence()


template<typename ForwardRange, typename T>
  inline typename detail::sequence_result<ForwardRange>::type
    sequence(ForwardRange &rng, T init, T step)
{
  return thrust::sequence(begin(rng), end(rng), init, step);
} // end sequence()


template<typename ForwardRange, typename T>
  inline typename detail::sequence_result<const ForwardRange>::type
    sequence(const ForwardRange &rng, T init, T step)
{
  return thrust::sequence(begin(rng), end(rng), init, step);
} // end sequence()


template<typename Space, typename T>
  inline typename detail::lazy_sequence_result<T,Space>::type
    sequence(T first)
{
  return sequence<Space>(first, std::numeric_limits<T>::max());
} // end sequence()


template<typename T>
  inline typename detail::lazy_sequence_result<T>::type
    sequence(T first)
{
  return sequence(first, std::numeric_limits<T>::max());
} // end sequence()


template<typename Space, typename T>
  inline typename detail::lazy_sequence_result<T,Space>::type
    sequence(T first, T last)
{
  typedef typename detail::lazy_sequence_result<T,Space>::type Result;
  return Result(thrust::counting_iterator<T,Space>(first), thrust::counting_iterator<T,Space>(last));
} // end sequence()


template<typename T>
  inline typename detail::lazy_sequence_result<T>::type
    sequence(T first, T last)
{
  return sequence<thrust::any_space_tag>(first, last);
} // end sequence()


} // end range

} // end experimental

} // end thrust

