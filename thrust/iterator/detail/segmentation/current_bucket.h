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

#include <thrust/iterator/detail/segmentation/segmented_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/detail/segmentation/bucket_iterator.h>
#include <thrust/range/algorithm/transform.h>

namespace thrust
{

namespace detail
{


template<typename Iterator>
  typename thrust::detail::bucket_iterator<
    thrust::detail::segmented_iterator<Iterator>
  >::type
    current_bucket(thrust::detail::segmented_iterator<Iterator> iter)
{
  return iter.current_bucket();
}


template<typename UnaryFunc, typename SegmentedIterator, typename Reference, typename Value>
  typename thrust::detail::bucket_iterator<
    thrust::transform_iterator<
      UnaryFunc, SegmentedIterator, Reference, Value
    >
  >::type
    current_bucket(thrust::transform_iterator<UnaryFunc,SegmentedIterator,Reference,Value> iter)
{
  typedef typename thrust::detail::bucket_iterator<
    thrust::transform_iterator<
      UnaryFunc, SegmentedIterator, Reference, Value
    >
  >::type result_type;

  return result_type(thrust::detail::current_bucket(iter.base()), iter.functor());
}


} // end detail

} // end thrust

