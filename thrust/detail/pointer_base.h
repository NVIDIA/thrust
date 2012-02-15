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

#pragma once

#include <thrust/detail/config.h>
#include <thrust/iterator/iterator_adaptor.h>
#include <thrust/iterator/iterator_traits.h>
#include <ostream>

namespace thrust
{
namespace detail
{

template<typename Derived, typename Value, typename Reference, typename Space> class pointer_base;

// this metafunction computes the type of iterator_adaptor pointer_base should inherit from
template<typename Derived, typename Value, typename Reference, typename Space>
  struct pointer_base_base
{
  // void pointers should have no value type
  typedef typename thrust::detail::eval_if<
    thrust::detail::is_void<typename thrust::detail::remove_const<Value>::type>::value,
    thrust::detail::identity_<void>,
    thrust::detail::identity_<Value>
  >::type value_type;

  // void pointers should have no reference type
  typedef typename thrust::detail::eval_if<
    thrust::detail::is_void<typename thrust::detail::remove_const<Value>::type>::value,
    thrust::detail::identity_<void>,
    thrust::detail::identity_<Reference>
  >::type reference;

  typedef thrust::experimental::iterator_adaptor<
    Derived,                             // pass along the type of our Derived class to iterator_adaptor
    Value *,                             // we adapt a raw pointer
    Derived,                             // our pointer type is the same as our Derived type
    value_type,                          // the value type
    Space,                               // space
    thrust::random_access_traversal_tag, // pointers have random access traversal
    reference,                           // pass along our Reference type
    std::ptrdiff_t
  > type;
}; // end pointer_base_base


// the base type for all of thrust's space-annotated pointers.
// for reasonable pointer-like semantics, derived types should reimplement the following:
// 1. no-argument constructor
// 2. constructor from OtherValue *
// 3. constructor from derived_type<OtherDerived,...>
// 4. assignment from derived_type<OtherDerived,...>
// These should just call the corresponding members of pointer_base.
template<typename Derived, typename Value, typename Reference, typename Space>
  class pointer_base
    : public pointer_base_base<Derived,Value,Reference,Space>::type
{
  private:
    typedef typename pointer_base_base<Derived,Value,Reference,Space>::type super_t;

    // friend iterator_core_access to give it access to dereference
    friend class thrust::experimental::iterator_core_access;

    __host__ __device__
    typename super_t::reference dereference() const;

    // don't provide access to this part of super_t's interface
    using super_t::base;
    using super_t::base_type;

  public:
    // constructors
    
    __host__ __device__
    pointer_base();

    // OtherValue shall be convertible to Value
    template<typename OtherValue>
    __host__ __device__
    explicit pointer_base(OtherValue *ptr);

    // OtherValue shall be convertible to Value
    template<typename OtherDerived, typename OtherValue, typename OtherReference>
    __host__ __device__
    pointer_base(const pointer_base<OtherDerived,OtherValue,OtherReference,Space> &other);

    // assignment
    
    // OtherValue shall be convertible to Value
    template<typename OtherDerived, typename OtherValue, typename OtherReference>
    __host__ __device__
    pointer_base &operator=(const pointer_base<OtherDerived,OtherValue,OtherReference,Space> &other);

    // observers

    __host__ __device__
    Value *get() const;
};

} // end detail
} // end thrust

#include <thrust/detail/pointer_base.inl>

