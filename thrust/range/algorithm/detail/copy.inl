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
#include <thrust/range/algorithm/copy.h>
#include <thrust/copy.h>
#include <thrust/range/begin.h>
#include <thrust/range/end.h>

namespace thrust
{

namespace experimental
{

namespace range
{


template<typename SinglePassRange1, typename SinglePassRange2>
  inline typename detail::copy_result<SinglePassRange2>::type
    copy(const SinglePassRange1 &input,
         SinglePassRange2       &result)
{
  typedef typename detail::copy_result<SinglePassRange2>::type Result;
  return Result(thrust::copy(begin(input), end(input), begin(result)), end(result));
} // end copy()


template<typename SinglePassRange1, typename SinglePassRange2>
  inline typename detail::copy_result<const SinglePassRange2>::type
    copy(const SinglePassRange1 &input,
         const SinglePassRange2 &result)
{
  typedef typename detail::copy_result<SinglePassRange2>::type Result;
  return Result(thrust::copy(begin(input), end(input), begin(result)), end(result));
} // end copy()


template<typename SinglePassRange1, typename SinglePassRange2, typename Predicate>
  inline typename detail::copy_result<SinglePassRange2>::type
    copy_if(const SinglePassRange1 &input,
            SinglePassRange2 &result,
            Predicate pred)
{
  typedef typename detail::copy_result<SinglePassRange2>::type Result;
  return Result(thrust::copy_if(begin(input), end(input), begin(result), pred), end(result));
} // end copy_if()


template<typename SinglePassRange1, typename SinglePassRange2, typename Predicate>
  inline typename detail::copy_result<const SinglePassRange2>::type
    copy_if(const SinglePassRange1 &input,
            const SinglePassRange2 &result,
            Predicate pred)
{
  typedef typename detail::copy_result<const SinglePassRange2>::type Result;
  return Result(thrust::copy_if(begin(input), end(input), begin(result), pred), end(result));
} // end copy_if()


template<typename SinglePassRange1, typename SinglePassRange2, typename SinglePassRange3, typename Predicate>
  inline typename detail::copy_result<SinglePassRange3>::type
    copy_if(const SinglePassRange1 &input,
            const SinglePassRange2 &stencil,
            SinglePassRange3 &result,
            Predicate pred)
{
  typedef typename detail::copy_result<SinglePassRange3>::type Result;
  return Result(thrust::copy_if(begin(input), end(input), begin(stencil), begin(result), pred), end(result));
} // end copy_if()


template<typename SinglePassRange1, typename SinglePassRange2, typename SinglePassRange3, typename Predicate>
  inline typename detail::copy_result<const SinglePassRange3>::type
    copy_if(const SinglePassRange1 &input,
            const SinglePassRange2 &stencil,
            const SinglePassRange3 &result,
            Predicate pred)
{
  typedef typename detail::copy_result<const SinglePassRange3>::type Result;
  return Result(thrust::copy_if(begin(input), end(input), begin(stencil), begin(result), pred), end(result));
} // end copy_if()


} // end range

} // end experimental

} // end thrust

