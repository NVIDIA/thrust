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


/*! \file zip_iterator.h
 *  \brief Defines the interface to an iterator
 *         whose reference is a tuple of the references
 *         of a tuple of iterators.
 */

#pragma once

#include <thrust/detail/config.h>
#include <thrust/iterator/detail/zip_iterator_base.h>

namespace thrust
{

template <typename IteratorTuple>
  class zip_iterator
    : public detail::zip_iterator_base<IteratorTuple>::type
{
  public:
    // null constructor
    __host__ __device__
    zip_iterator(void);

    // constructor from iterator tuple
    __host__ __device__
    zip_iterator(IteratorTuple iterator_tuple);

    // Get method for the iterator tuple.
    __host__ __device__
    const IteratorTuple &get_iterator_tuple() const;

  private:
    typedef typename
    detail::zip_iterator_base<IteratorTuple>::type super_t;

    friend class experimental::iterator_core_access;

    // Dereferencing returns a tuple built from the dereferenced
    // iterators in the iterator tuple.
    __host__ __device__
    typename super_t::reference dereference() const;

    // Two zip_iterators are equal if all iterators in the iterator
    // tuple are equal.
    template<typename OtherIteratorTuple>
    __host__ __device__
    bool equal(const zip_iterator<OtherIteratorTuple> &other) const;

    // Advancing a zip_iterator means to advance all iterators in the tuple
    __host__ __device__
    void advance(typename super_t::difference_type n);

    // Incrementing a zip iterator means to increment all iterators in the tuple
    __host__ __device__
    void increment();

    // Decrementing a zip iterator means to decrement all iterators in the tuple
    __host__ __device__
    void decrement();

    // Distance is calculated using the first iterator in the tuple.
    template<typename OtherIteratorTuple>
    __host__ __device__
      typename super_t::difference_type
        distance_to(const zip_iterator<OtherIteratorTuple> &other) const;

    // The iterator tuple.
    IteratorTuple m_iterator_tuple;
}; // end zip_iterator

// Make function for zip iterator
template<typename IteratorTuple>
__host__ __device__
zip_iterator<IteratorTuple> make_zip_iterator(IteratorTuple t);

} // end thrust

#include <thrust/iterator/detail/zip_iterator.inl>

