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


/*! \file transform_iterator.h
 *  \brief Defines the interface to an iterator
 *         whose value_type is the result of
 *         a unary function applied to the value_type
 *         of another iterator.
 */

#pragma once

#include <thrust/detail/config.h>

// #include the details first
#include <thrust/iterator/detail/transform_iterator.inl>
#include <thrust/iterator/iterator_facade.h>

namespace thrust
{

namespace experimental
{
  template <class UnaryFunc, class Iterator, class Reference = use_default, class Value = use_default>
  class transform_iterator
    : public detail::transform_iterator_base<UnaryFunc, Iterator, Reference, Value>::type
  {
    public:
    typedef typename
    detail::transform_iterator_base<UnaryFunc, Iterator, Reference, Value>::type
    super_t;

    friend class iterator_core_access;

  public:
    __host__ __device__
    transform_iterator() {}

    __host__ __device__
    transform_iterator(Iterator const& x, UnaryFunc f)
      : super_t(x), m_f(f) {
    }

    __host__ __device__
    explicit transform_iterator(Iterator const& x)
      : super_t(x) { }

    // XXX figure this out
//    template<
//        class OtherUnaryFunction
//      , class OtherIterator
//      , class OtherReference
//      , class OtherValue>
//    transform_iterator(
//         transform_iterator<OtherUnaryFunction, OtherIterator, OtherReference, OtherValue> const& t
//       , typename enable_if_convertible<OtherIterator, Iterator>::type* = 0
//#if !BOOST_WORKAROUND(BOOST_MSVC, == 1310)
//       , typename enable_if_convertible<OtherUnaryFunction, UnaryFunc>::type* = 0
//#endif 
//    )
//      : super_t(t.base()), m_f(t.functor())
//   {}

    __host__ __device__
    UnaryFunc functor() const
      { return m_f; }

  private:
    __host__ __device__
    typename super_t::reference dereference() const
    { 
      return m_f(*this->base());
    }

    // tag this as mutable per Dave Abrahams in this thread:
    // http://lists.boost.org/Archives/boost/2004/05/65332.php
    mutable UnaryFunc m_f;
  }; // end transform_iterator

  template <class UnaryFunc, class Iterator>
  __host__ __device__
  transform_iterator<UnaryFunc, Iterator>
  make_transform_iterator(Iterator it, UnaryFunc fun)
  {
    return transform_iterator<UnaryFunc, Iterator>(it, fun);
  } // end make_transform_iterator
} // end experimental

} // end thrust

