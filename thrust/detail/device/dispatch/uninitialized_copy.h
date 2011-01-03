/*
 *  Copyright 2008-2011 NVIDIA Corporation
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
#include <thrust/for_each.h>
#include <thrust/copy.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>
#include <thrust/distance.h>
#include <thrust/advance.h>

namespace thrust
{

namespace detail
{

namespace device
{

namespace dispatch
{


// trivial copy constructor path
template<typename InputIterator,
         typename OutputIterator>
  OutputIterator uninitialized_copy(InputIterator first,
                                    InputIterator last,
                                    OutputIterator result,
                                    thrust::detail::true_type) // has_trivial_copy_constructor
{
  return thrust::copy(first, last, result);
} // end uninitialized_copy()


namespace detail
{

template<typename InputType,
         typename OutputType>
  struct uninitialized_copy_functor
{
  __host__ __device__
  void operator()(thrust::tuple<const InputType&,OutputType&> t)
  {
    const InputType &in = thrust::get<0>(t);
    OutputType &out = thrust::get<1>(t);

    ::new(static_cast<void*>(&out)) OutputType(in);
  } // end operator()()
}; // end uninitialized_copy_functor

} // end detail


// non-trivial copy constructor path
template<typename InputIterator,
         typename OutputIterator>
  OutputIterator uninitialized_copy(InputIterator first,
                                    InputIterator last,
                                    OutputIterator result,
                                    thrust::detail::false_type) // has_trivial_copy_constructor
{
  // zip up the iterators
  typedef thrust::tuple<InputIterator,OutputIterator> IteratorTuple;
  typedef thrust::zip_iterator<IteratorTuple> ZipIterator;

  ZipIterator begin = thrust::make_zip_iterator(thrust::make_tuple(first,result));
  ZipIterator end = begin;

  // get a zip_iterator pointing to the end
  const typename thrust::iterator_difference<InputIterator>::type n = thrust::distance(first,last);
  thrust::advance(end, n);

  // create a functor
  typedef typename iterator_traits<InputIterator>::value_type InputType;
  typedef typename iterator_traits<OutputIterator>::value_type OutputType;

  detail::uninitialized_copy_functor<InputType, OutputType> f;

  // do the for_each
  thrust::for_each(begin, end, f);

  // return the end of the output range
  return get<1>(end.get_iterator_tuple());
} // end uninitialized_fill()


} // end dispatch

} // end device

} // end detail

} // end thrust

