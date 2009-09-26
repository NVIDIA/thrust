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


/*! \file reverse_iterator.h
 *  \brief Defines the interface to an iterator
 *         which adapts another iterator to step backwards.
 */

// thrust::reverse_iterator is derived from
// boost::reverse_iterator of the Boost Iterator
// Library, which is the work of
// David Abrahams, Jeremy Siek, & Thomas Witt.
// See http://www.boost.org for details.

#pragma once

#include <thrust/detail/config.h>
#include <thrust/iterator/iterator_adaptor.h>
#include <thrust/detail/type_traits.h>

namespace thrust
{

namespace experimental
{

template<typename BidirectionalIterator>
  class reverse_iterator
    : public thrust::experimental::iterator_adaptor<
        reverse_iterator<BidirectionalIterator>,
        BidirectionalIterator>
{
  /*! \cond
   */
  private:
    typedef thrust::experimental::iterator_adaptor<
      reverse_iterator<BidirectionalIterator>,
      BidirectionalIterator>
    > super_t;

    friend class thrust::experimental::iterator_core_access;
  /*! \endcond
   */

  public:
    /*! Default constructor does nothing.
     */
    __host__ __device__
    reverse_iterator() {}

    __host__ __device__
    explicit reverse_iterator(BidirectionalIterator x);

    template<typename OtherBidirectionalIterator>
    __host__ __device__
    reverse_iterator(reverse_iterator<OtherBidirectionalIterator> const &r
                     , typename thrust::detail::enable_if<
                         thrust::detail::is_convertible<
                           OtherBidirectionalIterator,
                           BidirectionalIterator
                         >::value
                       >::type * = 0
                     );

  private:
    typename super_t::reference dereference(void) const;

    __host__ __device__
    void increment(void);

    __host__ __device__
    void decrement(void);

    __host__ __device__
    void advance(typename super_t::difference_type n);

    template<typename OtherBidirectionalIterator>
    __host__ __device__
    distance_to(reverse_iterator<OtherBidirectionalIterator> const &y) const;
}; // end reverse_iterator

template<typename BidirectionalIterator>
__host__ __device__
reverse_iterator<BidirectionalIterator> make_reverse_iterator(BidirectionalIterator x);

} // end experimental

} // end thrust

#include <thrust/iterator/detail/reverse_iterator.inl>

