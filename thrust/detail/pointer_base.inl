/*
 *  Copyright 2008-2011 NVIDIA Corporation
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

#include <thrust/detail/pointer_base.h>

namespace thrust
{
namespace detail
{


template<typename Element, typename Space, typename Reference, typename Derived>
  pointer_base<Element,Space,Reference,Derived>
    ::pointer_base()
      : super_t(static_cast<Element*>(0))
{}

template<typename Element, typename Space, typename Reference, typename Derived>
  template<typename OtherElement>
    pointer_base<Element,Space,Reference,Derived>
      ::pointer_base(OtherElement *other)
        : super_t(other)
{}

template<typename Element, typename Space, typename Reference, typename Derived>
  template<typename OtherPointer>
    pointer_base<Element,Space,Reference,Derived>
      ::pointer_base(const OtherPointer &other,
                     typename thrust::detail::enable_if_pointer_is_convertible<
                       OtherPointer,
                       pointer_base<Element,Space,Reference,Derived>
                      >::type *)
        : super_t(thrust::detail::pointer_traits<OtherPointer>::get(other))
{}

template<typename Element, typename Space, typename Reference, typename Derived>
  template<typename OtherPointer>
    typename thrust::detail::enable_if_pointer_is_convertible<
      OtherPointer,
      pointer_base<Element,Space,Reference,Derived>,
      typename pointer_base<Element,Space,Reference,Derived>::derived_type &
    >::type
      pointer_base<Element,Space,Reference,Derived>
        ::operator=(const OtherPointer &other)
{
  super_t::base_reference() = thrust::detail::pointer_traits<OtherPointer>::get(other);
  return static_cast<derived_type&>(*this);
}

template<typename Element, typename Space, typename Reference, typename Derived>
  typename pointer_base<Element,Space,Reference,Derived>::super_t::reference
    pointer_base<Element,Space,Reference,Derived>
      ::dereference() const
{
  return typename super_t::reference(static_cast<const derived_type&>(*this));
}

template<typename Element, typename Space, typename Reference, typename Derived>
  Element *pointer_base<Element,Space,Reference,Derived>
    ::get() const
{
  return super_t::base();
}

namespace backend
{

// forward declaration of dereference_result
template<typename T> struct dereference_result;


template<typename Element, typename Space, typename Reference, typename Derived>
  struct dereference_result< pointer_base<Element,Space,Reference,Derived> >
{
  typedef Element& type;
};

template<typename Element, typename Space, typename Reference, typename Derived>
  inline __host__ __device__
    typename dereference_result< pointer_base<Element,Space,Reference,Derived> >::type
      dereference(pointer_base<Element,Space,Reference,Derived> ptr)
{
  return *ptr.get();
} // dereference


template<typename Element, typename Space, typename Reference, typename Derived, typename IndexType>
  inline __host__ __device__
    typename dereference_result< pointer_base<Element,Space,Reference,Derived> >::type
      dereference(pointer_base<Element,Space,Reference,Derived> ptr, IndexType n)
{
  return ptr.get()[n];
} // dereference

} // end backend
      

} // end detail
} // end thrust

