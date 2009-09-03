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

#pragma once

#include <thrust/iterator/zip_iterator.h>

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

  return tuple_transform(get_iterator_tuple(), detail::dereference_iterator());
} // end zip_iterator::dereference()


template <typename IteratorTuple>
  template <typename OtherIteratorTuple>
    bool zip_iterator<IteratorTuple>
      ::equal(const zip_iterator<OtherIteratorTuple> &other) const
{
  return get_iterator_tuple() == other.get_iterator_tuple();
} // end zip_iterator::equal()


template <typename IteratorTuple>
  void zip_iterator<IteratorTuple>
    ::advance(typename super_t::difference_type n)
{
  using namespace detail::tuple_impl_specific;

  tuple_for_each(m_iterator_tuple, detail::advance_iterator<typename super_t::difference_type>(n));
} // end zip_iterator::advance()


template <typename IteratorTuple>
  void zip_iterator<IteratorTuple>
    ::increment(void)
{
  using namespace detail::tuple_impl_specific;

  tuple_for_each(m_iterator_tuple, detail::increment_iterator());
} // end zip_iterator::increment()


template <typename IteratorTuple>
  void zip_iterator<IteratorTuple>
    ::decrement(void)
{
  using namespace detail::tuple_impl_specific;

  tuple_for_each(m_iterator_tuple, detail::decrement_iterator());
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

template<typename DeviceIteratorTuple>
  struct iterator_device_reference< thrust::zip_iterator<DeviceIteratorTuple> >
{
  // device_reference type is the type of the tuple obtained from the
  // iterators' device_reference types.
  typedef typename
  thrust::detail::tuple_of_device_references<DeviceIteratorTuple>::type type;
}; // end iterator_device_reference

namespace device
{

template<typename IteratorTuple>
  inline __device__
    typename thrust::detail::iterator_device_reference< thrust::zip_iterator<IteratorTuple> >::type
      dereference(thrust::zip_iterator<IteratorTuple> iter)
{
  using namespace thrust::detail::tuple_impl_specific;

  return tuple_transform(iter.get_iterator_tuple(), thrust::detail::device_dereference_iterator());
}; // end dereference()

template<typename IteratorTuple, typename IndexType>
  inline __device__
    typename thrust::detail::iterator_device_reference< thrust::zip_iterator<IteratorTuple> >::type
      dereference(thrust::zip_iterator<IteratorTuple> iter,
                  IndexType n)
{
  using namespace thrust::detail::tuple_impl_specific;

  thrust::detail::device_dereference_iterator_with_index<IndexType> f;
  f.n = n;

  return tuple_transform(iter.get_iterator_tuple(), f);
}; // end dereference()

} // end device

} // end detail

} // end thrust

