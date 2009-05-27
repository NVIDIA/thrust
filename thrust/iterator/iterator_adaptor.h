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


/*! \file iterator_adaptor.h
 *  \brief Defines a class which implements a host/device
 *         iterator from a more primitive iterator-like type.
 *         Based on Boost's iterator_adaptor class.
 */

#pragma once

#include <thrust/detail/config.h>
#include <thrust/iterator/iterator_facade.h>

namespace thrust
{

namespace experimental
{

struct use_default {};

template<typename Derived,
         typename Base,
         typename Value,
         typename CategoryOrTraversal,
         typename Reference,
         typename Pointer,
         typename Difference>
  class iterator_adaptor:
    public iterator_facade<Derived, Value, CategoryOrTraversal, Reference, Pointer, Difference>
{
    friend class iterator_core_access;
  
  public:
    __host__ __device__
    iterator_adaptor(){}

    __host__ __device__
    explicit iterator_adaptor(Base const& iter)
      : m_iterator(iter)
    {}

    typedef Base       base_type;
    typedef Reference  reference;
    typedef Difference difference_type;

    __host__ __device__
    Base const& base() const
    { return m_iterator; }

  protected:
    typedef iterator_adaptor iterator_adaptor_;

    __host__ __device__
    Base const& base_reference() const
    { return m_iterator; }

    __host__ __device__
    Base& base_reference()
    { return m_iterator; }

  private: // Core iterator interface for iterator_facade

    __host__ __device__
    typename iterator_adaptor::reference dereference() const
    { return *m_iterator; }

    template<typename OtherDerived, typename OtherIterator, typename V, typename C, typename R, typename P, typename D>
    __host__ __device__
    bool equal(iterator_adaptor<OtherDerived, OtherIterator, V, C, R, P, D> const& x) const
    { return m_iterator == x.base(); }

    __host__ __device__
    void advance(typename iterator_adaptor::difference_type n)
    {
      // XXX statically assert on random_access_traversal_tag
      m_iterator += n;
    }

    __host__ __device__
    void increment()
    { ++m_iterator; }

    __host__ __device__
    void decrement()
    {
      // XXX statically assert on bidirectional_traversal_tag
      --m_iterator;
    }

    template<typename OtherDerived, typename OtherIterator, typename V, typename C, typename R, typename P, typename D>
    __host__ __device__
    typename iterator_adaptor::difference_type distance_to(iterator_adaptor<OtherDerived, OtherIterator, V, C, R, P, D> const& y) const
    { return y.base() - m_iterator; }

  private:
    Base m_iterator; // exposition only
}; // end iterator_adaptor

} // end experimental

} // end thrust

#include <thrust/iterator/detail/iterator_adaptor.inl>

