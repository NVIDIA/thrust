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


/*! \file remove.inl
 *  \brief Inline file for remove.h
 */

#include <thrust/iterator/iterator_traits.h>

#include <thrust/detail/device/copy.h>

#include <thrust/detail/internal_functional.h>
#include <thrust/detail/raw_buffer.h>

namespace thrust
{
namespace detail
{
namespace device
{
namespace generic
{

template<typename ForwardIterator,
         typename Predicate>
  ForwardIterator remove_if(ForwardIterator first,
                            ForwardIterator last,
                            Predicate pred)
{
  typedef typename thrust::iterator_traits<ForwardIterator>::value_type InputType;
  typedef typename thrust::iterator_space<ForwardIterator>::type Space;

  // create temporary storage for an intermediate result
  thrust::detail::raw_buffer<InputType,Space> temp(first, last);

  // remove into temp
  return thrust::detail::device::generic::remove_copy_if(temp.begin(), temp.end(), temp.begin(), first, pred);
}

template<typename ForwardIterator,
         typename InputIterator,
         typename Predicate>
  ForwardIterator remove_if(ForwardIterator first,
                            ForwardIterator last,
                            InputIterator stencil,
                            Predicate pred)
{
  typedef typename thrust::iterator_traits<ForwardIterator>::value_type InputType;
  typedef typename thrust::iterator_space<ForwardIterator>::type Space;

  // create temporary storage for an intermediate result
  thrust::detail::raw_buffer<InputType,Space> temp(first, last);

  // remove into temp
  return thrust::detail::device::generic::remove_copy_if(temp.begin(), temp.end(), stencil, first, pred);
} 

template<typename InputIterator,
         typename OutputIterator,
         typename Predicate>
  OutputIterator remove_copy_if(InputIterator first,
                                InputIterator last,
                                OutputIterator result,
                                Predicate pred)
{
    return thrust::detail::device::generic::remove_copy_if(first, last, first, result, pred);
}


template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator,
         typename Predicate>
  OutputIterator remove_copy_if(InputIterator1 first,
                                InputIterator1 last,
                                InputIterator2 stencil,
                                OutputIterator result,
                                Predicate pred)
{
    return thrust::detail::device::copy_if(first, last, stencil, result, thrust::detail::not1(pred));
}

} // end namespace generic
} // end namespace device
} // end namespace detail
} // end namespace thrust

