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
#include <thrust/iterator/iterator_traits.h>
#include <thrust/iterator/detail/placement/place.h>
#include <thrust/iterator/detail/placement/placed_iterator_base.h>
#include <thrust/detail/device/dereference.h>

namespace thrust
{

namespace detail
{


template<typename UnplacedIterator>
  class placed_iterator
    : public placed_iterator_base<UnplacedIterator>::type
{
  private:
    typedef typename placed_iterator_base<UnplacedIterator>::type super_t;

  public:
    typedef thrust::detail::place_detail::place<
      typename thrust::iterator_space<UnplacedIterator>::type
    > place;

    __host__ __device__
    inline placed_iterator(void);

    __host__ __device__
    inline placed_iterator(UnplacedIterator i, place p = place());

    template<typename OtherIterator>
    __host__ __device__
    inline placed_iterator(placed_iterator<OtherIterator> i, place p = place());

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

template<typename UnplacedIterator> placed_iterator<UnplacedIterator> make_placed_iterator(UnplacedIterator i, place p);
template<typename UnplacedIterator> placed_iterator<UnplacedIterator> make_placed_iterator(UnplacedIterator i, std::size_t p);

namespace device
{

// XXX consider removing device::dereference for placed_iterator if
//     we intend to always strip its place before kernel launch
template<typename UnplacedIterator>
  struct dereference_result< placed_iterator<UnplacedIterator> >
    : dereference_result<UnplacedIterator>
{
}; // end dereference_result


template<typename UnplacedIterator>
  inline __host__ __device__
    typename dereference_result< placed_iterator<UnplacedIterator> >::type
      dereference(const placed_iterator<UnplacedIterator> &iter)
{
  return dereference(iter.base());
} // end dereference()

template<typename UnplacedIterator, typename IndexType>
  inline __host__ __device__
    typename dereference_result< placed_iterator<UnplacedIterator> >::type
      dereference(const placed_iterator<UnplacedIterator> &iter, IndexType n)
{
  return dereference(iter.base(), n);
} // end dereference()

} // end device

} // end detail

} // end thrust

#include <thrust/iterator/detail/placement/placed_iterator.inl>

