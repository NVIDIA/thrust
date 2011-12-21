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

#pragma once

#include <thrust/detail/config.h>
#include <thrust/tuple.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/detail/tuple_meta_transform.h>
#include <thrust/detail/reference_forward_declaration.h>

namespace thrust
{
namespace detail
{


// Metafunction to obtain the type of the tuple whose element types
// are the reference types of an iterator tuple.
//
template<typename IteratorTuple>
  struct tuple_of_references_base
    : tuple_meta_transform<
          IteratorTuple, 
          iterator_reference
        >
{
}; // end tuple_of_references_base

  
template<typename IteratorTuple>
  class tuple_of_iterator_references
    : public thrust::detail::tuple_of_references_base<IteratorTuple>::type
{
  private:
    typedef typename thrust::detail::tuple_of_references_base<IteratorTuple>::type super_t;

  public:
    // allow implicit construction from tuple<refs>
    inline __host__ __device__
    tuple_of_iterator_references(const super_t &other)
      : super_t(other)
    {}

    // allow assignment from tuples
    // XXX might be worthwhile to guard this with an enable_if is_assignable
    template<typename U1, typename U2>
    inline __host__ __device__
    tuple_of_iterator_references &operator=(const detail::cons<U1,U2> &other)
    {
      super_t::operator=(other);
      return *this;
    }

    // allow assignment from reference<tuple>
    template<
      typename T0, typename T1, typename T2,
      typename T3, typename T4, typename T5,
      typename T6, typename T7, typename T8,
      typename T9,
      typename Pointer,
      typename Derived
    >
    inline __host__ __device__
    typename thrust::detail::enable_if<
      thrust::detail::is_assignable<
        super_t,
        const thrust::tuple<T0,T1,T2,T3,T4,T5,T6,T7,T8,T9>
      >::value,
      tuple_of_iterator_references &
    >::type
    operator=(const thrust::reference<thrust::tuple<T0,T1,T2,T3,T4,T5,T6,T7,T8,T9>, Pointer, Derived> &other)
    {
      typedef thrust::tuple<T0,T1,T2,T3,T4,T5,T6,T7,T8,T9> tuple_type;

      // XXX perhaps this could be accelerated
      tuple_type other_tuple = other;
      super_t::operator=(other_tuple);
      return *this;
    }
};


} // end detail
} // end thrust

