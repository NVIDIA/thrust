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

#pragma once

#include <thrust/iterator/zip_iterator.h>
#include <thrust/detail/tuple_transform.h>

namespace thrust
{


template <typename IteratorTuple>
  zip_iterator<IteratorTuple>
    ::zip_iterator(void)
{
} // end zip_iterator::zip_iterator()


template <typename IteratorTuple>
  zip_iterator<IteratorTuple>
    ::zip_iterator(IteratorTuple iterator_tuple)
      :m_iterator_tuple(iterator_tuple)
{
} // end zip_iterator::zip_iterator()


template <typename IteratorTuple>
const IteratorTuple &zip_iterator<IteratorTuple>
  ::get_iterator_tuple(void) const
{
  return m_iterator_tuple;
} // end zip_iterator::get_iterator_tuple()


template <typename IteratorTuple>
  typename zip_iterator<IteratorTuple>::super_t::reference
    zip_iterator<IteratorTuple>
      ::dereference(void) const
{
  using namespace detail::tuple_impl_specific;

  return thrust::detail::tuple_host_transform<detail::dereference_iterator::template apply>(get_iterator_tuple(), detail::dereference_iterator());
} // end zip_iterator::dereference()


template <typename IteratorTuple>
  template <typename OtherIteratorTuple>
    bool zip_iterator<IteratorTuple>
      ::equal(const zip_iterator<OtherIteratorTuple> &other) const
{
  return get<0>(get_iterator_tuple()) == get<0>(other.get_iterator_tuple());
} // end zip_iterator::equal()


template <typename IteratorTuple>
  void zip_iterator<IteratorTuple>
    ::advance(typename super_t::difference_type n)
{
  using namespace detail::tuple_impl_specific;

  // dispatch on space
  tuple_for_each(m_iterator_tuple,
                 detail::advance_iterator<typename super_t::difference_type>(n),
                 typename thrust::iterator_space<zip_iterator>::type());
} // end zip_iterator::advance()


template <typename IteratorTuple>
  void zip_iterator<IteratorTuple>
    ::increment(void)
{
  using namespace detail::tuple_impl_specific;

  // dispatch on space
  tuple_for_each(m_iterator_tuple, detail::increment_iterator(),
                 typename thrust::iterator_space<zip_iterator>::type());
} // end zip_iterator::increment()


template <typename IteratorTuple>
  void zip_iterator<IteratorTuple>
    ::decrement(void)
{
  using namespace detail::tuple_impl_specific;

  // dispatch on space
  tuple_for_each(m_iterator_tuple, detail::decrement_iterator(),
                 typename thrust::iterator_space<zip_iterator>::type());
} // end zip_iterator::decrement()


template <typename IteratorTuple>
  template <typename OtherIteratorTuple>
    typename zip_iterator<IteratorTuple>::super_t::difference_type
      zip_iterator<IteratorTuple>
        ::distance_to(const zip_iterator<OtherIteratorTuple> &other) const
{
  return get<0>(other.get_iterator_tuple()) - get<0>(get_iterator_tuple());
} // end zip_iterator::distance_to()


template <typename IteratorTuple>
  zip_iterator<IteratorTuple> make_zip_iterator(IteratorTuple t)
{
  return zip_iterator<IteratorTuple>(t);
} // end make_zip_iterator()


namespace detail
{

namespace device
{


template<typename DeviceIteratorTuple>
  struct dereference_result< thrust::zip_iterator<DeviceIteratorTuple> >
{
  // device_reference type is the type of the tuple obtained from the
  // iterators' device_reference types.
  typedef typename
  thrust::detail::tuple_of_dereference_result<DeviceIteratorTuple>::type type;
}; // end dereference_result


template<typename IteratorTuple>
  inline __host__ __device__
    typename dereference_result< thrust::zip_iterator<IteratorTuple> >::type
      dereference(const thrust::zip_iterator<IteratorTuple> &iter)
{
  using namespace thrust::detail::tuple_impl_specific;

  return thrust::detail::tuple_host_device_transform<thrust::detail::device_dereference_iterator::template apply>(iter.get_iterator_tuple(), thrust::detail::device_dereference_iterator());
}; // end dereference()


template<typename IteratorTuple, typename IndexType>
  inline __host__ __device__
    typename dereference_result< thrust::zip_iterator<IteratorTuple> >::type
      dereference(const thrust::zip_iterator<IteratorTuple> &iter,
                  IndexType n)
{
  using namespace thrust::detail::tuple_impl_specific;

  thrust::detail::device_dereference_iterator_with_index<IndexType> f;
  f.n = n;

  return thrust::detail::tuple_host_device_transform<thrust::detail::device_dereference_iterator_with_index<IndexType>::template apply>(iter.get_iterator_tuple(), f);
}; // end dereference()


} // end device

} // end detail

} // end thrust

