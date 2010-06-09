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
#include <thrust/range/algorithm/gather.h>
#include <thrust/range/begin.h>
#include <thrust/range/end.h>
#include <thrust/gather.h>
#include <thrust/iterator/permutation_iterator.h>

namespace thrust
{

namespace experimental
{

namespace range
{


template<typename SinglePassRange1, typename RandomAccessRange, typename SinglePassRange2>
  inline typename detail::gather_result<SinglePassRange2>::type
    gather(const SinglePassRange1  &map,
           const RandomAccessRange &input,
           SinglePassRange2 &result)
{
  typedef typename detail::gather_result<SinglePassRange2>::type Result;

  return Result(thrust::gather(begin(map), end(map), begin(input), begin(result)), end(result));
} // end gather()


template<typename SinglePassRange1, typename RandomAccessRange, typename SinglePassRange2>
  inline typename detail::gather_result<const SinglePassRange2>::type
    gather(const SinglePassRange1  &map,
           const RandomAccessRange &input,
           const SinglePassRange2 &result)
{
  typedef typename detail::gather_result<const SinglePassRange2>::type Result;

  return Result(thrust::gather(begin(map), end(map), begin(input), begin(result)), end(result));
} // end gather()


template<typename SinglePassRange1, typename SinglePassRange2, typename RandomAccessRange, typename SinglePassRange3>
  inline typename detail::gather_result<SinglePassRange3>::type
    gather_if(const SinglePassRange1  &map,
              const SinglePassRange2  &stencil,
              const RandomAccessRange &input,
              SinglePassRange3        &result)
{
  typedef typename detail::gather_result<SinglePassRange3>::type Result;

  return Result(thrust::gather_if(begin(map), end(map), begin(stencil), begin(input), begin(result)), end(result));
} // end gather_if()


template<typename SinglePassRange1, typename SinglePassRange2, typename RandomAccessRange, typename SinglePassRange3>
  inline typename detail::gather_result<const SinglePassRange3>::type
    gather_if(const SinglePassRange1  &map,
              const SinglePassRange2  &stencil,
              const RandomAccessRange &input,
              const SinglePassRange3  &result)
{
  typedef typename detail::gather_result<const SinglePassRange3>::type Result;

  return Result(thrust::gather_if(begin(map), end(map), begin(stencil), begin(input), begin(result)), end(result));
} // end gather_if()


template<typename SinglePassRange1, typename SinglePassRange2, typename RandomAccessRange, typename SinglePassRange3, typename Predicate>
  inline typename detail::gather_result<SinglePassRange3>::type
    gather_if(const SinglePassRange1  &map,
              const SinglePassRange2  &stencil,
              const RandomAccessRange &input,
              SinglePassRange3        &result,
              Predicate               pred)
{
  typedef typename detail::gather_result<SinglePassRange3>::type Result;

  return Result(thrust::gather_if(begin(map), end(map), begin(stencil), begin(input), begin(result), pred), end(result));
} // end gather_if()


template<typename SinglePassRange1, typename SinglePassRange2, typename RandomAccessRange, typename SinglePassRange3, typename Predicate>
  inline typename detail::gather_result<const SinglePassRange3>::type
    gather_if(const SinglePassRange1  &map,
              const SinglePassRange2  &stencil,
              const RandomAccessRange &input,
              const SinglePassRange3  &result,
              Predicate               pred)
{
  typedef typename detail::gather_result<const SinglePassRange3>::type Result;

  return Result(thrust::gather_if(begin(map), end(map), begin(stencil), begin(input), begin(result), pred), end(result));
} // end gather_if()


template<typename SinglePassRange, typename RandomAccessRange>
  inline typename detail::lazy_gather_result<const SinglePassRange, const RandomAccessRange>
    gather(const SinglePassRange &map,
           const RandomAccessRange &input)
{
  typedef typename detail::lazy_gather_result<const SinglePassRange, const RandomAccessRange>::type Result;

  return Result(thrust::make_permutation_iterator(begin(input), begin(map)),
                thrust::make_permutation_iterator(begin(input), end(map)));
} // end gather()

} // end range

} // end experimental

} // end thrust

