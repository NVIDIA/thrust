/*
 *  Copyright 2008-2010 NVIDIA Corporation
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
#include <thrust/iterator/detail/placement/place.h>
#include <thrust/iterator/detail/placement/placed_iterator_base.h>
#include <thrust/detail/device/dereference.h>

namespace thrust
{

namespace detail
{


template<typename Iterator>
  class placed_iterator
    : public placed_iterator_base<Iterator>::type
{
  private:
    typedef typename placed_iterator_base<Iterator>::type super_t;

  public:
    typedef thrust::detail::place place;

    __host__ __device__
    inline placed_iterator(void);

    __host__ __device__
    inline placed_iterator(Iterator i, place p = 0);

    template<typename OtherIterator>
    __host__ __device__
    inline placed_iterator(placed_iterator<OtherIterator> i, place p = 0);

    __host__ __device__
    void set_place(place p);

    __host__ __device__
    place get_place(void) const;

  private:
    place m_place;

    // iterator core interface follows
    friend class thrust::experimental::iterator_core_access;

    __host__ __device__
    typename super_t::reference dereference(void) const;
}; // end placed_iterator

template<typename Iterator> placed_iterator<Iterator> make_placed_iterator(Iterator i, place p);

namespace device
{

template<typename Iterator>
  struct dereference_result< placed_iterator<Iterator> >
    : dereference_result<Iterator>
{
}; // end dereference_result


template<typename Iterator>
  inline __host__ __device__
    typename dereference_result< placed_iterator<Iterator> >::type
      dereference(const placed_iterator<Iterator> &iter)
{
  return dereference(iter.base());
} // end dereference()

template<typename Iterator, typename IndexType>
  inline __host__ __device__
    typename dereference_result< placed_iterator<Iterator> >::type
      dereference(const placed_iterator<Iterator> &iter, IndexType n)
{
  return dereference(iter.base(), n);
} // end dereference()

} // end device

} // end detail

} // end thrust

#include <thrust/iterator/detail/placement/placed_iterator.inl>

