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


/*! \file device_reference.inl
 *  \brief Inline file for device_reference.h.
 */

#include <thrust/device_reference.h>

namespace thrust
{

template<typename T>
  template<typename OtherT>
    device_reference<T> &
      device_reference<T>
        ::operator=(const device_reference<OtherT> &other)
{
  return super_t::operator=(other);
} // end operator=()

template<typename T>
  device_reference<T> &
    device_reference<T>
      ::operator=(const value_type &x)
{
  return super_t::operator=(x);
} // end operator=()

namespace detail
{

// XXX iterator_facade tries to instantiate the Reference
//     type when computing the answer to is_convertible<Reference,Value>
//     we can't do that at that point because cuda_reference
//     is not complete
//     WAR the problem by specializing is_convertible
template<typename T>
  struct is_convertible<thrust::device_reference<T>, T>
    : thrust::detail::true_type
{};

} // end detail

template<typename T>
__host__ __device__
void swap(device_reference<T> &a, device_reference<T> &b)
{
  a.swap(b);
} // end swap()

} // end thrust

