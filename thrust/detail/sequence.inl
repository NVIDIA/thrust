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


/*! \file sequence.inl
 *  \brief Inline file for sequence.h.
 */

#include <thrust/sequence.h>

#include <thrust/transform.h>
#include <thrust/distance.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/iterator/counting_iterator.h>

namespace thrust
{

template<typename ForwardIterator>
  void sequence(ForwardIterator first,
                ForwardIterator last)
{
  typedef typename thrust::iterator_traits<ForwardIterator>::value_type T;
  thrust::sequence(first, last, T(0));
} // end sequence()


template<typename ForwardIterator, typename T>
  void sequence(ForwardIterator first,
                ForwardIterator last,
                T init)
{
  thrust::sequence(first, last, init, T(1));
} // end sequence()



namespace detail
{

template <typename T>
struct sequence_functor
{
  const T init;
  const T step;

  sequence_functor(T _init, T _step) 
      : init(_init), step(_step) {}
  
  template <typename IntegerType>
      __host__ __device__
  T operator()(const IntegerType i) const
  {
    return init + step * i;
  }
}; // end sequence_functor

} // end namespace detail


template<typename ForwardIterator, typename T>
  void sequence(ForwardIterator first,
                ForwardIterator last,
                T init,
                T step)
{
    typedef typename thrust::iterator_traits<ForwardIterator>::difference_type difference_type;

    detail::sequence_functor<T> func(init, step);

    // by default, counting_iterator uses a 64b difference_type on 32b platforms to avoid overflowing its counter.
    // this causes problems when a zip_iterator is created in transform's implementation -- ForwardIterator is
    // incremented by a 64b difference_type and some compilers warn
    // to avoid this, specify the counting_iterator's difference_type to be the same as ForwardIterator's.
    thrust::counting_iterator<difference_type, thrust::use_default, thrust::use_default, difference_type> iter(0);

    thrust::transform(iter, iter + thrust::distance(first, last), first, func);
} // end sequence()

} // end namespace thrust

