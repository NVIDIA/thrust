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

#include <thrust/detail/allocator/destroy_range.h>
#include <thrust/detail/allocator/allocator_traits.h>
#include <thrust/detail/type_traits/pointer_traits.h>
#include <thrust/for_each.h>

namespace thrust
{
namespace detail
{
namespace allocator_traits_detail
{


// when T has a trivial destructor and Allocator has no
// destroy member function, destroying T has no effect
template<typename Allocator, typename T>
  struct has_no_effect_destroy
    : integral_constant<
        bool,
        has_trivial_destructor<T>::value && !has_member_destroy1<Allocator,T>::value
      >
{};

// we know that std::allocator::destroy's only effect is to
// call T's destructor, so we needn't use it when destroying T
template<typename U, typename T>
  struct has_no_effect_destroy<std::allocator<U>, T>
    : has_trivial_destructor<T>
{};



// destroy_range is a no op if element destruction has no effect
template<typename Allocator, typename Pointer, typename Size>
  typename enable_if<
    has_no_effect_destroy<
      Allocator,
      typename pointer_element<Pointer>::type
    >::value
  >::type
    destroy_range(Allocator &, Pointer, Size)
{
  // no op
}


template<typename Allocator>
  struct destroy_via_allocator
{
  Allocator &a;

  destroy_via_allocator(Allocator &a)
    : a(a)
  {}

  template<typename T>
  inline __host__ __device__
  void operator()(T &x)
  {
    allocator_traits<Allocator>::destroy(a, &x);
  }
};


// destroy_range is effectful if element destroy has effect
// XXX need to check whether we really need to destroy via the allocator or not
template<typename Allocator, typename Pointer, typename Size>
  typename disable_if<
    has_no_effect_destroy<
      Allocator,
      typename pointer_element<Pointer>::type
    >::value
  >::type
    destroy_range(Allocator &a, Pointer p, Size n)
{
  thrust::for_each_n(p, n, destroy_via_allocator<Allocator>(a));
}


} // end allocator_traits_detail


template<typename Allocator, typename Pointer, typename Size>
  void destroy_range(Allocator &a, Pointer p, Size n)
{
  return allocator_traits_detail::destroy_range(a,p,n);
}


} // end detail
} // end thrust

