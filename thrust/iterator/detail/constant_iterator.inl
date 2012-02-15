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

#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/detail/backend/dereference.h>

namespace thrust
{

namespace detail
{

namespace backend
{


// specialize dereference_result for counting_iterator
// transform_iterator returns the same reference on the device as on the host
template <typename Value, typename Incrementable, typename Space>
  struct dereference_result<
    thrust::constant_iterator<
      Value, Incrementable, Space
    >
  >
{
  typedef typename thrust::iterator_traits< thrust::constant_iterator<Value,Incrementable,Space> >::reference type;
}; // end dereference_result


template<typename Value, typename Incrementable, typename Space>
  inline __host__ __device__
    typename dereference_result< thrust::constant_iterator<Value,Incrementable,Space> >::type
      dereference(const thrust::constant_iterator<Value,Incrementable,Space> &iter)
{
  return iter.value();
} // end dereference()


template<typename Value, typename Incrementable, typename Space, typename IndexType>
  inline __host__ __device__
    typename dereference_result< thrust::constant_iterator<Value,Incrementable,Space> >::type
      dereference(const thrust::constant_iterator<Value,Incrementable,Space> &iter, IndexType n)
{
  return iter[n];
} // end dereference()


} // end namespace backend

} // end namespace detail

} // end namespace thrust

