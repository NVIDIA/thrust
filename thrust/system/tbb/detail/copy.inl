/*
 *  Copyright 2008-2012 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in ctbbliance with the License.
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
#include <thrust/system/tbb/detail/copy.h>
#include <thrust/system/detail/generic/copy.h>
#include <thrust/detail/type_traits/minimum_type.h>
#include <thrust/system/cpp/detail/copy.h>

namespace thrust
{
namespace system
{
namespace tbb
{
namespace detail
{
namespace dispatch
{

template<typename System,
         typename InputIterator,
         typename OutputIterator>
  OutputIterator copy(dispatchable<System> &system,
                      InputIterator first,
                      InputIterator last,
                      OutputIterator result,
                      thrust::incrementable_traversal_tag)
{
  return thrust::system::cpp::detail::copy(system, first, last, result);
} // end copy()


template<typename System,
         typename InputIterator,
         typename OutputIterator>
  OutputIterator copy(dispatchable<System> &system,
                      InputIterator first,
                      InputIterator last,
                      OutputIterator result,
                      thrust::random_access_traversal_tag)
{
  return thrust::system::detail::generic::copy(system, first, last, result);
} // end copy()


template<typename System,
         typename InputIterator,
         typename Size,
         typename OutputIterator>
  OutputIterator copy_n(dispatchable<System> &system,
                        InputIterator first,
                        Size n,
                        OutputIterator result,
                        thrust::incrementable_traversal_tag)
{
  return thrust::system::cpp::detail::copy_n(system, first, n, result);
} // end copy_n()


template<typename System,
         typename InputIterator,
         typename Size,
         typename OutputIterator>
  OutputIterator copy_n(dispatchable<System> &system,
                        InputIterator first,
                        Size n,
                        OutputIterator result,
                        thrust::random_access_traversal_tag)
{
  return thrust::system::detail::generic::copy_n(system, first, n, result);
} // end copy_n()

} // end dispatch


template<typename System,
         typename InputIterator,
         typename OutputIterator>
OutputIterator copy(dispatchable<System> &system,
                    InputIterator first,
                    InputIterator last,
                    OutputIterator result)
{
  typedef typename thrust::iterator_traversal<InputIterator>::type  traversal1;
  typedef typename thrust::iterator_traversal<OutputIterator>::type traversal2;
  
  typedef typename thrust::detail::minimum_type<traversal1,traversal2>::type traversal;

  // dispatch on minimum traversal
  return thrust::system::tbb::detail::dispatch::copy(system,first,last,result,traversal());
} // end copy()



template<typename System,
         typename InputIterator,
         typename Size,
         typename OutputIterator>
OutputIterator copy_n(dispatchable<System> &system,
                      InputIterator first,
                      Size n,
                      OutputIterator result)
{
  typedef typename thrust::iterator_traversal<InputIterator>::type  traversal1;
  typedef typename thrust::iterator_traversal<OutputIterator>::type traversal2;
  
  typedef typename thrust::detail::minimum_type<traversal1,traversal2>::type traversal;

  // dispatch on minimum traversal
  return thrust::system::tbb::detail::dispatch::copy_n(system,first,n,result,traversal());
} // end copy_n()


} // end namespace detail
} // end namespace tbb
} // end namespace system
} // end namespace thrust

