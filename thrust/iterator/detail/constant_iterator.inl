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

#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/iterator_traits.h>

namespace thrust
{

namespace detail
{

// specialize iterator_device_reference for counting_iterator
// transform_iterator returns the same reference on the device as on the host
template <typename Value, typename Incrementable, typename Space>
  struct iterator_device_reference<
    thrust::constant_iterator<
      Value, Incrementable, Space
    >
  >
{
  typedef typename thrust::iterator_traits< thrust::constant_iterator<Value,Incrementable,Space> >::reference type;
}; // end iterator_device_reference


namespace device
{

template<typename Value, typename Incrementable, typename Space>
  inline __device__
    typename iterator_device_reference< thrust::constant_iterator<Value,Incrementable,Space> >::type
      dereference(thrust::constant_iterator<Value,Incrementable,Space> iter)
{
  return *iter;
} // end dereference()

template<typename Value, typename Incrementable, typename Space, typename IndexType>
  inline __device__
    typename iterator_device_reference< thrust::constant_iterator<Value,Incrementable,Space> >::type
      dereference(thrust::constant_iterator<Value,Incrementable,Space> iter, IndexType n)
{
  return iter[n];
} // end dereference()

} // end namespace device

} // end namespace detail

} // end namespace thrust

