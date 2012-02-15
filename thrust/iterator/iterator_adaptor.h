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


/*! \file iterator_adaptor.h
 *  \brief An iterator which adapts a base iterator
 */

/*
 * (C) Copyright David Abrahams 2002.
 * (C) Copyright Jeremy Siek    2002.
 * (C) Copyright Thomas Witt    2002.
 * 
 * Distributed under the Boost Software License, Version 1.0.
 * (See accompanying NOTICE file for the complete license)
 *
 * For more information, see http://www.boost.org
 */

#pragma once

#include <thrust/detail/config.h>
#include <thrust/iterator/iterator_facade.h>

// #include the details first
#include <thrust/iterator/detail/iterator_adaptor.inl>

namespace thrust
{

struct use_default {};

namespace experimental
{

template <
      typename Derived
    , typename Base
    , typename Pointer
    // XXX nvcc can't handle these defaults at the moment
    //, typename Value                = use_default
    //, typename CategoryOrSpace      = use_default
    //, typename CategoryOrTraversal  = use_default
    //, typename Reference            = use_default
    //, typename Difference           = use_default
    , typename Value
    , typename Space
    , typename Traversal
    , typename Reference
    , typename Difference = use_default
  >
  class iterator_adaptor:
    public detail::iterator_adaptor_base<
      Derived, Base, Pointer, Value, Space, Traversal, Reference, Difference
    >::type
{
    friend class iterator_core_access;

  protected:
    typedef typename detail::iterator_adaptor_base<
        Derived, Base, Pointer, Value, Space, Traversal, Reference, Difference
    >::type super_t;
  
  public:
    __host__ __device__
    iterator_adaptor(){}

    __host__ __device__
    explicit iterator_adaptor(Base const& iter)
      : m_iterator(iter)
    {}

    typedef Base       base_type;
    // XXX BUG: why do we have to declare this here?  it's supposed to be published in super_t
    typedef typename super_t::reference reference;
    // XXX BUG: why do we have to declare this here?  it's supposed to be published in super_t
    typedef typename super_t::difference_type difference_type;

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

    typename iterator_adaptor::reference dereference() const
    { return *m_iterator; }

    template<typename OtherDerived, typename OtherIterator, typename P, typename V, typename S, typename T, typename R, typename D>
    __host__ __device__
    bool equal(iterator_adaptor<OtherDerived, OtherIterator, P, V, S, T, R, D> const& x) const
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

    template<typename OtherDerived, typename OtherIterator, typename P, typename V, typename S, typename T, typename R, typename D>
    __host__ __device__
    typename iterator_adaptor::difference_type distance_to(iterator_adaptor<OtherDerived, OtherIterator, P, V, S, T, R, D> const& y) const
    { return y.base() - m_iterator; }

  private:
    Base m_iterator; // exposition only
}; // end iterator_adaptor

} // end experimental

} // end thrust

