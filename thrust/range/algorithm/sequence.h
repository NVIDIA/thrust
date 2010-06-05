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

#include <thrust/detail/config.h>
#include <thrust/range/algorithm/detail/sequence_result.h>

namespace thrust
{

namespace experimental
{

namespace range
{


template<typename ForwardRange>
  inline typename detail::sequence_result<ForwardRange>::type
    sequence(ForwardRange &rng);


// add a second overload to accept temporary ranges from things like zip()
// XXX change
//
// const ForwardRange &rng
//
// to
//
// ForwardRange &&rng
//
// upon addition of rvalue references
template<typename ForwardRange>
  inline typename detail::sequence_result<const ForwardRange>::type
    sequence(const ForwardRange &rng);


template<typename ForwardRange, typename T>
  inline typename detail::sequence_result<ForwardRange>::type
    sequence(ForwardRange &rng, T init);


// add a second overload to accept temporary ranges from things like zip()
// XXX change
//
// const ForwardRange &rng
//
// to
//
// ForwardRange &&rng
//
// upon addition of rvalue references
template<typename ForwardRange, typename T>
  inline typename detail::sequence_result<const ForwardRange>::type
    sequence(const ForwardRange &rng, T init);


template<typename ForwardRange, typename T>
  inline typename detail::sequence_result<ForwardRange>::type
    sequence(ForwardRange &rng, T init, T step);


// add a second overload to accept temporary ranges from things like zip()
// XXX change
//
// const ForwardRange &rng
//
// to
//
// ForwardRange &&rng
//
// upon addition of rvalue references
template<typename ForwardRange, typename T>
  inline typename detail::sequence_result<const ForwardRange>::type
    sequence(const ForwardRange &rng, T init, T step);


// XXX replace these two overloads with one with an optional parameter when we get constexpr
template<typename T>
  inline typename detail::lazy_sequence_result<T>::type
    sequence(T init);


template<typename Space, typename T>
  inline typename detail::lazy_sequence_result<T,Space>::type
    sequence(T init);


template<typename T>
  inline typename detail::lazy_sequence_result<T>::type
    sequence(T first, T last);


template<typename Space, typename T>
  inline typename detail::lazy_sequence_result<T,Space>::type
    sequence(T first, T last);


} // end range

} // end experimental

} // end thrust

#include <thrust/range/algorithm/detail/sequence.inl>

