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
#include <thrust/system/omp/detail/copy.h>
#include <thrust/system/detail/generic/copy.h>
#include <thrust/detail/type_traits/minimum_type.h>
#include <thrust/system/cpp/detail/copy.h>

namespace thrust
{
namespace system
{
namespace omp
{
namespace detail
{
namespace dispatch
{

template<typename InputIterator,
         typename OutputIterator>
  OutputIterator copy(InputIterator first,
                      InputIterator last,
                      OutputIterator result,
                      thrust::incrementable_traversal_tag)
{
  return thrust::system::cpp::detail::copy(tag(), first, last, result);
} // end copy()


template<typename InputIterator,
         typename OutputIterator>
  OutputIterator copy(InputIterator first,
                      InputIterator last,
                      OutputIterator result,
                      thrust::random_access_traversal_tag)
{
  // XXX WAR problems reconciling unrelated types such as omp & tbb
  // reinterpret iterators as omp
  // this ensures that generic::copy's implementation, which eventually results in
  // zip_iterator works correctly
  thrust::detail::tagged_iterator<OutputIterator,tag> retagged_result(result);

  return thrust::system::detail::generic::copy(tag(), thrust::reinterpret_tag<tag>(first), thrust::reinterpret_tag<tag>(last), retagged_result).base();
} // end copy()


template<typename InputIterator,
         typename Size,
         typename OutputIterator>
  OutputIterator copy_n(InputIterator first,
                        Size n,
                        OutputIterator result,
                        thrust::incrementable_traversal_tag)
{
  return thrust::system::cpp::detail::copy_n(tag(), first, n, result);
} // end copy_n()


template<typename InputIterator,
         typename Size,
         typename OutputIterator>
  OutputIterator copy_n(InputIterator first,
                        Size n,
                        OutputIterator result,
                        thrust::random_access_traversal_tag)
{
  typedef typename thrust::iterator_system<OutputIterator>::type original_tag;

  // XXX WAR problems reconciling unrelated types such as omp & tbb
  // reinterpret iterators as omp
  // this ensures that generic::copy's implementation, which eventually results in
  // zip_iterator works correctly
  thrust::detail::tagged_iterator<OutputIterator,tag> retagged_result(result);

  return thrust::system::detail::generic::copy_n(tag(), thrust::reinterpret_tag<tag>(first), n, retagged_result).base();
} // end copy_n()

} // end dispatch


template<typename InputIterator,
         typename OutputIterator>
OutputIterator copy(tag,
                    InputIterator first,
                    InputIterator last,
                    OutputIterator result)
{
  typedef typename thrust::iterator_traversal<InputIterator>::type  traversal1;
  typedef typename thrust::iterator_traversal<OutputIterator>::type traversal2;
  
  typedef typename thrust::detail::minimum_type<traversal1,traversal2>::type traversal;

  // dispatch on minimum traversal
  return thrust::system::omp::detail::dispatch::copy(first,last,result,traversal());
} // end copy()



template<typename InputIterator,
         typename Size,
         typename OutputIterator>
OutputIterator copy_n(tag,
                      InputIterator first,
                      Size n,
                      OutputIterator result)
{
  typedef typename thrust::iterator_traversal<InputIterator>::type  traversal1;
  typedef typename thrust::iterator_traversal<OutputIterator>::type traversal2;
  
  typedef typename thrust::detail::minimum_type<traversal1,traversal2>::type traversal;

  // dispatch on minimum traversal
  return thrust::system::omp::detail::dispatch::copy_n(first,n,result,traversal());
} // end copy_n()


} // end namespace detail
} // end namespace omp
} // end namespace system
} // end namespace thrust

