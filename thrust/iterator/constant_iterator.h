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


/*! \file constant_iterator.h
 *  \brief Defines the interface to an iterator
 *         which returns a constant value when
 *         dereferenced.
 */

#pragma once

#include <thrust/detail/config.h>
#include <thrust/iterator/detail/constant_iterator_base.h>

namespace thrust
{

template<typename Value,
         typename Incrementable = thrust::experimental::use_default,
         typename Space = thrust::experimental::use_default>
  class constant_iterator
    : public detail::constant_iterator_base<Value, Incrementable, Space>::type
{
    friend class thrust::experimental::iterator_core_access;
    typedef typename detail::constant_iterator_base<Value, Incrementable, Space>::type          super_t;
    typedef typename detail::constant_iterator_base<Value, Incrementable, Space>::incrementable incrementable;
    typedef typename detail::constant_iterator_base<Value, Incrementable, Space>::base_iterator base_iterator;

  public:
    typedef typename super_t::reference  reference;
    typedef typename super_t::value_type value_type;

    __host__ __device__
    constant_iterator(void)
      : super_t(), m_value(){};

    __host__ __device__
    constant_iterator(constant_iterator const &rhs)
      : super_t(rhs.base()), m_value(rhs.m_value) {}

    __host__ __device__
    constant_iterator(value_type const& v, incrementable const &i = incrementable())
      : super_t(base_iterator(i)), m_value(v) {}

    template<typename OtherValue, typename OtherIncrementable>
    __host__ __device__
    constant_iterator(OtherValue const& v, OtherIncrementable const& i = incrementable())
      : super_t(base_iterator(i)), m_value(v) {}

    __host__ __device__
    Value const& value(void) const
    { return m_value; }

  protected:
    __host__ __device__
    Value const& value_reference(void) const
    { return m_value; }

    __host__ __device__
    Value & value_reference(void)
    { return m_value; }
  
  private: // Core iterator interface
    __host__ __device__
    reference dereference(void) const
    {
      return m_value;
    }

  private:
    const Value m_value;
}; // end constant_iterator

template<typename V, typename I>
inline __host__ __device__
constant_iterator<V,I> make_constant_iterator(V x, I i = int())
{
  return constant_iterator<V,I>(x, i);
} // end make_constant_iterator()

template<typename V>
inline __host__ __device__
constant_iterator<V> make_constant_iterator(V x)
{
  return constant_iterator<V>(x, 0);
} // end make_constant_iterator()

} // end namespace thrust

#include <thrust/iterator/detail/constant_iterator.inl>

