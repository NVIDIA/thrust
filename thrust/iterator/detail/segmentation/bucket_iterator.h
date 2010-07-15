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

#include <thrust/iterator/transform_iterator.h>
#include <thrust/range/algorithm/transform.h>
#include <thrust/range/detail/iterator.h>
#include <thrust/detail/type_traits.h>

namespace thrust
{

namespace detail
{


template<typename Iterator> class segmented_iterator;

// forward declaration to WAR circular #inclusion
template<typename> struct is_segmented;

template<typename Iterator, typename Enable = void> struct bucket_iterator {};

template<typename Iterator>
  struct bucket_iterator<thrust::detail::segmented_iterator<Iterator> >
{
  typedef typename thrust::detail::segmented_iterator<Iterator>::bucket_iterator type;
};


template<typename UnaryFunction, typename Range>
struct transform_range_functor
  : thrust::unary_function<
      Range, 
      typename thrust::experimental::range::detail::lazy_unary_transform_result<
        Range,
        UnaryFunction
      >::type
    >
{
  transform_range_functor(UnaryFunction f)
    : m_f(f) {}

  typedef typename thrust::experimental::range::detail::lazy_unary_transform_result<
    Range,
    UnaryFunction
  >::type result_type;

  // define this as __host__ __device__ to allow it to work with transform_iterator
  // we will only ever use it from __host__ code
  __host__ __device__
  result_type operator()(Range &r)
  {
#ifndef __CUDA_ARCH__
    return thrust::experimental::range::transform(r, m_f);
#else
    return result_type();
#endif
  }

  // add a second overload to accept temporaries
  // XXX change this to an rvalue reference upon arrival of c++0x
  __host__ __device__
  result_type operator()(const Range &r)
  {
#ifndef __CUDA_ARCH__
    return thrust::experimental::range::transform(r, m_f);
#else
    return result_type();
#endif
  }

  UnaryFunction m_f;
}; // end transform_range_functor



template<typename UnaryFunc, typename Iterator, typename Reference, typename Value>
  struct bucket_iterator<
    thrust::transform_iterator<UnaryFunc,Iterator,Reference,Value>,
    typename thrust::detail::enable_if<
      is_segmented<Iterator>
    >::type
  >
{
  private:
    // get the SegmentedIterator's bucket_iterator
    typedef typename bucket_iterator<Iterator>::type                       base_range_iterator;

    // get the value_type of the bucket_iterator -- this is the range we're going to transform
    typedef typename thrust::iterator_value<base_range_iterator>::type     base_range;

    // name a transform_range_functor
    typedef transform_range_functor<UnaryFunc,base_range>                  xfrm_functor;

  public:
    // name a transform_iterator which will transform a base_range_iterator using the xfrm_functor
    typedef thrust::transform_iterator<xfrm_functor,base_range_iterator>   type;
};


} // end detail

} // end thrust

