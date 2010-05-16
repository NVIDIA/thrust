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

//  Copyright Neil Groves 2009. Use, modification and
//  distribution is subject to the Boost Software License, Version
//  1.0. (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)
//
//
// For more information, see http://www.boost.org/libs/range/

#pragma once

#include <thrust/range/begin.h>
#include <thrust/range/end.h>
#include <thrust/transform.h>
#include <thrust/range/iterator_range.h>
#include <thrust/range/detail/iterator.h>

namespace thrust
{

namespace experimental
{

namespace range
{

namespace detail
{

template<typename Range>
  struct transform_result
{
  typedef iterator_range<typename range_iterator<Range>::type> type;
};

} // end detail


// XXX these overloads differ from Boost.Range's transform in that they take their arguments
//     as ranges and return a range of the unconsumed portion of result 
template<typename SinglePassRange1, typename SinglePassRange2, typename UnaryFunction>
  inline typename detail::transform_result<SinglePassRange2>::type
    transform(const SinglePassRange1 &rng,
              SinglePassRange2 &result,
              UnaryFunction f)
{
  typedef typename detail::transform_result<SinglePassRange2>::type Result;

  return Result(thrust::transform(begin(rng), end(rng), begin(result), f), end(result));
} // end transform()


template<typename SinglePassRange1, typename SinglePassRange2, typename SinglePassRange3, typename BinaryFunction>
  inline typename detail::transform_result<SinglePassRange3>::type
    transform(const SinglePassRange1 &rng1,
              const SinglePassRange2 &rng2,
              SinglePassRange3 &result,
              BinaryFunction f)
{
  typedef typename detail::transform_result<SinglePassRange3>::type Result;

  return Result(thrust::transform(begin(rng1), end(rng1), begin(rng2), begin(result), f), end(result));
} // end for_each()


} // end range

} // end experimental

} // end thrust


