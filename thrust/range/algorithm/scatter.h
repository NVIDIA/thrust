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

namespace thrust
{

namespace experimental
{

namespace range
{


template<typename SinglePassRange1,
         typename SinglePassRange2,
         typename RandomAccessRange>
  void scatter(const SinglePassRange1 &input,
               const SinglePassRange2 &map,
               RandomAccessRange      &result);


// add a second overload to accept temporary ranges for the third parameter from things like zip()
//
// XXX change
//
//     const RandomAccessRange &result
//
//     to
//
//     RandomAccessRange &&result
//
//     upon addition of rvalue references
template<typename SinglePassRange1,
         typename SinglePassRange2,
         typename RandomAccessRange>
  void scatter(const SinglePassRange1  &input,
               const SinglePassRange2  &map,
               const RandomAccessRange &result);


template<typename SinglePassRange1,
         typename SinglePassRange2,
         typename SinglePassRange3,
         typename RandomAccessRange>
  void scatter_if(const SinglePassRange1 &input,
                  const SinglePassRange2 &map,
                  const SinglePassRange3 &stencil,
                  RandomAccessRange      &result);


// add a second overload to accept temporary ranges for the mutable parameter from things like zip()
//
// XXX change
//
//     const RandomAccessRange &result
//
//     to
//
//     RandomAccessRange &&result
//
//     upon addition of rvalue references
template<typename SinglePassRange1,
         typename SinglePassRange2,
         typename SinglePassRange3,
         typename RandomAccessRange>
  void scatter_if(const SinglePassRange1  &input,
                  const SinglePassRange2  &map,
                  const SinglePassRange3  &stencil,
                  const RandomAccessRange &result);


template<typename SinglePassRange1,
         typename SinglePassRange2,
         typename SinglePassRange3,
         typename RandomAccessRange,
         typename Predicate>
  void scatter_if(const SinglePassRange1 &input,
                  const SinglePassRange2 &map,
                  const SinglePassRange3 &stencil,
                  RandomAccessRange      &result,
                  Predicate pred);


template<typename SinglePassRange1,
         typename SinglePassRange2,
         typename SinglePassRange3,
         typename RandomAccessRange,
         typename Predicate>
  void scatter_if(const SinglePassRange1  &input,
                  const SinglePassRange2  &map,
                  const SinglePassRange3  &stencil,
                  const RandomAccessRange &result,
                  Predicate pred);


} // end range

} // end experimental

} // end thrust

#include <thrust/range/algorithm/detail/scatter.inl>

