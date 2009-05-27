/*
 *  Copyright 2008-2009 NVIDIA Corporation
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
#include <thrust/iterator/iterator_traits.h>
#include <thrust/detail/dispatch/sequence.h>

namespace thrust
{

template<typename ForwardIterator>
  void sequence(ForwardIterator first,
                ForwardIterator last)
{
  typedef typename thrust::iterator_traits<ForwardIterator>::value_type OutputType;
  thrust::sequence(first, last, OutputType(0), OutputType(1));
} // end sequence()


template<typename ForwardIterator, typename T>
  void sequence(ForwardIterator first,
                ForwardIterator last,
                T init)
{
  thrust::sequence(first, last, init, T(1));
} // end sequence()


template<typename ForwardIterator, typename T>
  void sequence(ForwardIterator first,
                ForwardIterator last,
                T init,
                T step)
{
  thrust::detail::dispatch::sequence(first, last, init, step,
    typename thrust::iterator_traits<ForwardIterator>::iterator_category());
} // end sequence()

} // end namespace thrust

