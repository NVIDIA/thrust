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


/*! \file make_device_dereferenceable.h
 *  \brief Defines a function for transforming
 *         iterators over device storage which are dereferenceable
 *         from the host into similar versions of themselves which
 *         are dereferenceable from __global__ & __device__ functions.
 */

#pragma once

#include <iterator>

namespace thrust
{

namespace detail
{

template<typename T>
  struct device_dereferenceable_iterator_traits
{
  typedef typename T::device_dereferenceable_type              device_dereferenceable_type;
}; // end device_dereferenceable_iterator_traits

// XXX this is a hack to allow host_vector to compile
//     this specialization for T* is not meant to imply that
//     we wish to transform host pointers into something device
//     dereferenceable
template<typename T>
  struct device_dereferenceable_iterator_traits<T*>
    : public std::iterator_traits<T*>
{
  typedef T* device_dereferenceable_type;
}; // end device_dereferenceable_iterator_traits

template<typename T>
  struct make_device_dereferenceable
{
  __host__ __device__
  static
  typename device_dereferenceable_iterator_traits<T>::device_dereferenceable_type
  transform(T x)
  {
    return x.device_dereferenceable();
  } // end transform()
}; // end make_device_dereferenceable_functor

} // end namespace detail

} // end namespace thrust

