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
#include <thrust/iterator/iterator_adaptor.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/iterator_categories.h>

namespace thrust
{

namespace experimental
{

template<typename Value>
  class constant_iterator
    : public iterator_adaptor<constant_iterator<Value>,
                              counting_iterator<ptrdiff_t, ptrdiff_t>,
                              Value,
                              thrust::experimental::random_access_universal_iterator_tag,
                              Value const &,
                              Value *,
                              ptrdiff_t>
{
    friend class iterator_core_access;

    //typedef counting_iterator<ptrdiff_t,ptrdiff_t> Base;

    typedef iterator_adaptor<constant_iterator<Value>,
                             counting_iterator<ptrdiff_t, ptrdiff_t>,
                             Value,
                             thrust::experimental::random_access_universal_iterator_tag,
                             Value const &,
                             Value *,
                             ptrdiff_t> super_t;

  public:
    typedef Value const & reference;
    typedef typename super_t::difference_type difference_type;

    __host__ __device__
    constant_iterator(void){};

  //  // XXX nvcc can't compile this at the moment
  //  //__host__ __device__
  //  //constant_iterator(constant_iterator const &rhs):super_t(rhs.base()){}

    __host__ __device__
    constant_iterator(Value const& v, ptrdiff_t const& c = ptrdiff_t())
      : super_t(typename super_t::base_type(c)), m_value(v) {}

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
    typename constant_iterator::reference dereference(void) const
    {
      return m_value;
    }

  private:
    Value m_value;
}; // end constant_iterator

template<typename T>
__host__ __device__
constant_iterator<T> make_constant_iterator(T x, ptrdiff_t c = ptrdiff_t())
{
  return constant_iterator<T>(x, c);
} // end make_constant_iterator()

// XXX TODO consider adding this convenience
//template<typename T>
//__host__ __device__
//std::pair< constant_iterator<T>,constant_iterator<T> > make_constant_range(T x, ptrdiff_t size)
//{
//  return std::make_pair(make_constant_iterator(x, 0), make_constant_iterator(x, size));
//} // end make_constant_range()

} // end namespace experimental

namespace detail
{

template<typename T>
  struct make_device_dereferenceable< thrust::experimental::constant_iterator<T> >
{
  __host__ __device__
  static
  thrust::experimental::constant_iterator<T> &
  transform(thrust::experimental::constant_iterator<T> &x)
  {
    return x;
  } // end transform()

  __host__ __device__
  static
  const thrust::experimental::constant_iterator<T> &
  transform(const thrust::experimental::constant_iterator<T> &x)
  {
    return x;
  } // end transform()
}; // end make_device_dereferenceable

} // end detail

} // end namespace thrust

