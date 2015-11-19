/*
 *  Copyright 2008-2013 NVIDIA Corporation
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
  __host__ __device__
  pointer<Element,Tag,Reference,Derived>
    ::pointer()
      : super_t(static_cast<Element*>(0))
{} // end pointer::pointer


template<typename Element, typename Tag, typename Reference, typename Derived>
  template<typename OtherElement>
    __host__ __device__
    pointer<Element,Tag,Reference,Derived>
      ::pointer(OtherElement *other)
        : super_t(other)
{} // end pointer::pointer


template<typename Element, typename Tag, typename Reference, typename Derived>
  template<typename OtherPointer>
    __host__ __device__
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
    __host__ __device__
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
  __host__ __device__
  typename pointer<Element,Tag,Reference,Derived>::super_t::reference
    pointer<Element,Tag,Reference,Derived>
      ::dereference() const
{
  return typename super_t::reference(static_cast<const derived_type&>(*this));
} // end pointer::dereference


template<typename Element, typename Tag, typename Reference, typename Derived>
  __host__ __device__
  Element *pointer<Element,Tag,Reference,Derived>
    ::get() const
{
  return super_t::base();
} // end pointer::get

template<typename Element, typename Tag, typename Reference, typename Derived,
         typename charT, typename traits>
std::basic_ostream<charT, traits> &
operator<<(std::basic_ostream<charT, traits> &os,
           const pointer<Element, Tag, Reference, Derived> &p) {
  return os << p.get();
}

namespace detail
{

#if (THRUST_HOST_COMPILER == THRUST_HOST_COMPILER_MSVC) && (_MSC_VER <= 1400)
// XXX WAR MSVC 2005 problem with correctly implementing
//     pointer_raw_pointer for pointer by specializing it here
template<typename Element, typename Tag, typename Reference, typename Derived>
  struct pointer_raw_pointer< thrust::pointer<Element,Tag,Reference,Derived> >
{
  typedef typename pointer<Element,Tag,Reference,Derived>::raw_pointer type;
}; // end pointer_raw_pointer
#endif


#if (THRUST_HOST_COMPILER == THRUST_HOST_COMPILER_GCC) && (THRUST_GCC_VERSION < 40200)
// XXX WAR g++-4.1 problem with correctly implementing
//     pointer_element for pointer by specializing it here
template<typename Element, typename Tag>
  struct pointer_element< thrust::pointer<Element,Tag> >
{
  typedef Element type;
}; // end pointer_element

template<typename Element, typename Tag, typename Reference>
  struct pointer_element< thrust::pointer<Element,Tag,Reference> >
    : pointer_element< thrust::pointer<Element,Tag> >
{}; // end pointer_element

template<typename Element, typename Tag, typename Reference, typename Derived>
  struct pointer_element< thrust::pointer<Element,Tag,Reference,Derived> >
    : pointer_element< thrust::pointer<Element,Tag,Reference> >
{}; // end pointer_element



// XXX WAR g++-4.1 problem with correctly implementing
//     rebind_pointer for pointer by specializing it here
template<typename Element, typename Tag, typename NewElement>
  struct rebind_pointer<thrust::pointer<Element,Tag>, NewElement>
{
  // XXX note we don't attempt to rebind the pointer's Reference type (or Derived)
  typedef thrust::pointer<NewElement,Tag> type;
};

template<typename Element, typename Tag, typename Reference, typename NewElement>
  struct rebind_pointer<thrust::pointer<Element,Tag,Reference>, NewElement>
    : rebind_pointer<thrust::pointer<Element,Tag>, NewElement>
{};

template<typename Element, typename Tag, typename Reference, typename Derived, typename NewElement>
  struct rebind_pointer<thrust::pointer<Element,Tag,Reference,Derived>, NewElement>
    : rebind_pointer<thrust::pointer<Element,Tag,Reference>, NewElement>
{};
#endif

} // end namespace detail


} // end thrust

