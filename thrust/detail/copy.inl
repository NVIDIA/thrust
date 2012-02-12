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


#include <thrust/detail/config.h>
#include <thrust/detail/copy.h>
#include <thrust/system/detail/generic/select_system.h>
#include <thrust/system/detail/generic/copy.h>
#include <thrust/detail/adl_helper.h>

namespace thrust
{


template<typename InputIterator,
         typename OutputIterator>
  OutputIterator copy(InputIterator first,
                      InputIterator last,
                      OutputIterator result)
{
  using thrust::system::detail::generic::select_system;
  using thrust::system::detail::generic::copy;

  typedef typename thrust::iterator_system<InputIterator>::type  system1;
  typedef typename thrust::iterator_system<OutputIterator>::type system2;

  return copy(select_system(system1(),system2()), first, last, result);
} // end copy()


template<typename InputIterator,
         typename Size,
         typename OutputIterator>
  OutputIterator copy_n(InputIterator first,
                        Size n,
                        OutputIterator result)
{
  using thrust::system::detail::generic::select_system;
  using thrust::system::detail::generic::copy_n;

  typedef typename thrust::iterator_system<InputIterator>::type  system1;
  typedef typename thrust::iterator_system<OutputIterator>::type system2;

  return copy_n(select_system(system1(),system2()), first, n, result);
} // end copy_n()


} // end namespace thrust

