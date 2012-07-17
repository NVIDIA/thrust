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


/*! \file remove.inl
 *  \brief Inline file for remove.h
 */

#include <thrust/detail/config.h>
#include <thrust/system/detail/generic/remove.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/detail/copy_if.h>
#include <thrust/detail/internal_functional.h>
#include <thrust/detail/temporary_array.h>
#include <thrust/remove.h>

namespace thrust
{
namespace system
{
namespace detail
{
namespace generic
{


template<typename System,
         typename ForwardIterator,
         typename T>
  ForwardIterator remove(thrust::dispatchable<System> &system,
                         ForwardIterator first,
                         ForwardIterator last,
                         const T &value)
{
  thrust::detail::equal_to_value<T> pred(value);

  // XXX consider using a placeholder here
  return thrust::remove_if(system, first, last, pred);
} // end remove()


template<typename System,
         typename InputIterator,
         typename OutputIterator,
         typename T>
  OutputIterator remove_copy(thrust::dispatchable<System> &system,
                             InputIterator first,
                             InputIterator last,
                             OutputIterator result,
                             const T &value)
{
  thrust::detail::equal_to_value<T> pred(value);

  // XXX consider using a placeholder here
  return thrust::remove_copy_if(system, first, last, result, pred);
} // end remove_copy()


template<typename System,
         typename ForwardIterator,
         typename Predicate>
  ForwardIterator remove_if(thrust::dispatchable<System> &system,
                            ForwardIterator first,
                            ForwardIterator last,
                            Predicate pred)
{
  typedef typename thrust::iterator_traits<ForwardIterator>::value_type InputType;

  // create temporary storage for an intermediate result
  thrust::detail::temporary_array<InputType,System> temp(system, first, last);

  // remove into temp
  return thrust::remove_copy_if(system, temp.begin(), temp.end(), temp.begin(), first, pred);
} // end remove_if()


template<typename System,
         typename ForwardIterator,
         typename InputIterator,
         typename Predicate>
  ForwardIterator remove_if(thrust::dispatchable<System> &system,
                            ForwardIterator first,
                            ForwardIterator last,
                            InputIterator stencil,
                            Predicate pred)
{
  typedef typename thrust::iterator_traits<ForwardIterator>::value_type InputType;

  // create temporary storage for an intermediate result
  thrust::detail::temporary_array<InputType,System> temp(system, first, last);

  // remove into temp
  return thrust::remove_copy_if(system, temp.begin(), temp.end(), stencil, first, pred);
} // end remove_if() 


template<typename System,
         typename InputIterator,
         typename OutputIterator,
         typename Predicate>
  OutputIterator remove_copy_if(thrust::dispatchable<System> &system,
                                InputIterator first,
                                InputIterator last,
                                OutputIterator result,
                                Predicate pred)
{
  return thrust::remove_copy_if(system, first, last, first, result, pred);
} // end remove_copy_if()


template<typename System,
         typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator,
         typename Predicate>
  OutputIterator remove_copy_if(thrust::dispatchable<System> &system,
                                InputIterator1 first,
                                InputIterator1 last,
                                InputIterator2 stencil,
                                OutputIterator result,
                                Predicate pred)
{
  return thrust::copy_if(system, first, last, stencil, result, thrust::detail::not1(pred));
} // end remove_copy_if()


} // end namespace generic
} // end namespace detail
} // end namespace system
} // end namespace thrust

