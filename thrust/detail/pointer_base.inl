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

#include <thrust/detail/pointer_base.h>

namespace thrust
{
namespace detail
{


template<typename Derived, typename Value, typename Reference, typename Space>
  pointer_base<Derived,Value,Reference,Space>
    ::pointer_base()
      : super_t(static_cast<Value*>(0))
{}

template<typename Derived, typename Value, typename Reference, typename Space>
  template<typename OtherValue>
    pointer_base<Derived,Value,Reference,Space>
      ::pointer_base(OtherValue *other)
        : super_t(other)
{}

template<typename Derived, typename Value, typename Reference, typename Space>
  template<typename OtherDerived, typename OtherValue, typename OtherReference>
    pointer_base<Derived,Value,Reference,Space>
      ::pointer_base(const pointer_base<OtherDerived,OtherValue,OtherReference,Space> &other)
        : super_t(other.get())
{}

template<typename Derived, typename Value, typename Reference, typename Space>
  template<typename OtherDerived, typename OtherValue, typename OtherReference>
    pointer_base<Derived,Value,Reference,Space> &
      pointer_base<Derived,Value,Reference,Space>
        ::operator=(const pointer_base<OtherDerived,OtherValue,OtherReference,Space> &other)
{
  super_t::base_reference() = other.get();
  return *this;
}

template<typename Derived, typename Value, typename Reference, typename Space>
  typename pointer_base<Derived,Value,Reference,Space>::super_t::reference
    pointer_base<Derived,Value,Reference,Space>
      ::dereference() const
{
  return typename super_t::reference(static_cast<const Derived&>(*this));
}

template<typename Derived, typename Value, typename Reference, typename Space>
  Value *pointer_base<Derived,Value,Reference,Space>
    ::get() const
{
  return super_t::base();
}
      

} // end detail
} // end thrust

