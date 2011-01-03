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

/*! \file casts.h
 *  \brief Unsafe casts for internal use.
 */

#pragma once

#include <thrust/iterator/iterator_traits.h>

namespace thrust
{

namespace detail
{

namespace dispatch
{

template<typename TrivialIterator>
  typename thrust::iterator_traits<TrivialIterator>::value_type *
    raw_pointer_cast(TrivialIterator i,
                     thrust::random_access_host_iterator_tag)
{
  typedef typename thrust::iterator_traits<TrivialIterator>::value_type * Pointer;

  // cast away constness
  return const_cast<Pointer>(&*i);
} // end raw_pointer_cast()

// this path will work for device_ptr & device_vector::iterator
template<typename TrivialIterator>
  typename thrust::iterator_traits<TrivialIterator>::value_type *
    raw_pointer_cast(TrivialIterator i,
                     thrust::random_access_device_iterator_tag)
{
  typedef typename thrust::iterator_traits<TrivialIterator>::value_type * Pointer;

  // cast away constness
  return const_cast<Pointer>((&*i).get());
} // end raw_pointer_cast()

} // end dispatch

template<typename TrivialIterator>
  typename thrust::iterator_traits<TrivialIterator>::value_type *raw_pointer_cast(TrivialIterator i)
{
  return detail::dispatch::raw_pointer_cast(i, thrust::iterator_traits<TrivialIterator>::iterator_category());
} // end raw_pointer_cast()

} // end detail

} // end thrust

