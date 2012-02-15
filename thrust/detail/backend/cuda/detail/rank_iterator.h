/*
 *  Copyright 2008-2012 NVIDIA Corporation
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
#include <thrust/functional.h>
#include <thrust/iterator/transform_iterator.h>

namespace thrust
{
namespace detail
{
namespace backend
{
namespace cuda
{
namespace detail
{


// this functor keeps an iterator pointing to a sorted range and a Compare
// operator() takes an index as an argument, looks up x = first[index]
// and returns x's rank in the segment of elements equivalent to x
template<typename RandomAccessIterator, typename Compare>
  struct nth_occurrence_functor
    : thrust::unary_function<
        typename thrust::iterator_difference<RandomAccessIterator>::type,
        typename thrust::iterator_difference<RandomAccessIterator>::type
      >
{
  nth_occurrence_functor(RandomAccessIterator f, Compare c)
    : first(f), comp(c) {}

  template<typename Index>
  __host__ __device__ __forceinline__
  typename thrust::iterator_difference<RandomAccessIterator>::type operator()(Index index)
  {
    RandomAccessIterator x = first;
    x += index;

    return x - thrust::detail::backend::generic::scalar::lower_bound(first,x,dereference(x),comp);
  }

  RandomAccessIterator first;
  Compare comp;
}; // end nth_occurrence_functor


template<typename RandomAccessIterator, typename Compare>
  class rank_iterator
{
  typedef typename thrust::iterator_difference<RandomAccessIterator>::type difference;
  typedef thrust::counting_iterator<difference> counter;

  public:
    typedef thrust::transform_iterator<
      nth_occurrence_functor<RandomAccessIterator,Compare>,
      counter
    > type;
}; // end rank_iterator


template<typename RandomAccessIterator, typename Compare>
  typename rank_iterator<RandomAccessIterator,Compare>::type
    make_rank_iterator(RandomAccessIterator iter, Compare comp)
{
  typedef typename thrust::iterator_difference<RandomAccessIterator>::type difference;
  typedef thrust::counting_iterator<difference> CountingIterator;

  nth_occurrence_functor<RandomAccessIterator,Compare> f(iter,comp);

  return thrust::make_transform_iterator(CountingIterator(0), f);
} // end make_rank_iterator()


} // end detail
} // end cuda
} // end backend
} // end detail
} // end thrust

