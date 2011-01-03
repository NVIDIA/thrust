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

#include <thrust/iterator/iterator_traits.h>
#include <thrust/detail/device/dispatch/copy.h>

namespace thrust
{
namespace detail
{
namespace device
{

template<typename InputIterator,
         typename OutputIterator>
  OutputIterator copy(InputIterator  first, 
                      InputIterator  last, 
                      OutputIterator result)
{
  return thrust::detail::device::dispatch::copy(first, last, result,
    typename thrust::iterator_space<InputIterator>::type(),
    typename thrust::iterator_space<OutputIterator>::type());
}

template<typename InputIterator,
         typename Size,
         typename OutputIterator>
  OutputIterator copy_n(InputIterator  first, 
                        Size n, 
                        OutputIterator result)
{
  return thrust::detail::device::dispatch::copy_n(first, n, result,
    typename thrust::iterator_space<InputIterator>::type(),
    typename thrust::iterator_space<OutputIterator>::type());
}


template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator,
         typename Predicate>
  OutputIterator copy_if(InputIterator1 first,
                         InputIterator1 last,
                         InputIterator2 stencil,
                         OutputIterator result,
                         Predicate pred)
{
  return thrust::detail::device::dispatch::copy_if(first, last, stencil, result, pred,
    typename thrust::iterator_space<InputIterator1>::type(),
    typename thrust::iterator_space<InputIterator2>::type(),
    typename thrust::iterator_space<OutputIterator>::type());
}

} // end namespace device
} // end namespace detail
} // end namespace thrust

