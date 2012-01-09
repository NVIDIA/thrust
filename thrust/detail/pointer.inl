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

#include <thrust/detail/config.h>
#include <thrust/detail/pointer.h>


namespace thrust
{


template<typename Element, typename Tag, typename Reference, typename Derived>
  pointer<Element,Tag,Reference,Derived>
    ::pointer()
      : super_t(static_cast<Element*>(0))
{} // end pointer::pointer


template<typename Element, typename Tag, typename Reference, typename Derived>
  template<typename OtherElement>
    pointer<Element,Tag,Reference,Derived>
      ::pointer(OtherElement *other)
        : super_t(other)
{} // end pointer::pointer


template<typename Element, typename Tag, typename Reference, typename Derived>
  template<typename OtherPointer>
    pointer<Element,Tag,Reference,Derived>
      ::pointer(const OtherPointer &other,
                typename thrust::detail::enable_if_pointer_is_convertible<
                  OtherPointer,
                  pointer<Element,Tag,Reference,Derived>
                 >::type *)
        : super_t(thrust::detail::pointer_traits<OtherPointer>::get(other))
{} // end pointer::pointer


template<typename Element, typename Tag, typename Reference, typename Derived>
  template<typename OtherPointer>
    typename thrust::detail::enable_if_pointer_is_convertible<
      OtherPointer,
      pointer<Element,Tag,Reference,Derived>,
      typename pointer<Element,Tag,Reference,Derived>::derived_type &
    >::type
      pointer<Element,Tag,Reference,Derived>
        ::operator=(const OtherPointer &other)
{
  super_t::base_reference() = thrust::detail::pointer_traits<OtherPointer>::get(other);
  return static_cast<derived_type&>(*this);
} // end pointer::operator=


template<typename Element, typename Tag, typename Reference, typename Derived>
  typename pointer<Element,Tag,Reference,Derived>::super_t::reference
    pointer<Element,Tag,Reference,Derived>
      ::dereference() const
{
  return typename super_t::reference(static_cast<const derived_type&>(*this));
} // end pointer::dereference


template<typename Element, typename Tag, typename Reference, typename Derived>
  Element *pointer<Element,Tag,Reference,Derived>
    ::get() const
{
  return super_t::base();
} // end pointer::get


} // end thrust

