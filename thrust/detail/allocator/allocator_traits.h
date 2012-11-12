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
#include <thrust/detail/type_traits/pointer_traits.h>
#include <thrust/detail/type_traits/has_nested_type.h>
#include <thrust/detail/type_traits.h>

namespace thrust
{
namespace detail
{
namespace allocator_traits_detail
{

__THRUST_DEFINE_HAS_NESTED_TYPE(has_pointer, pointer)
__THRUST_DEFINE_HAS_NESTED_TYPE(has_const_pointer, const_pointer)
__THRUST_DEFINE_HAS_NESTED_TYPE(has_reference, reference)
__THRUST_DEFINE_HAS_NESTED_TYPE(has_const_reference, const_reference)
__THRUST_DEFINE_HAS_NESTED_TYPE(has_void_pointer, void_pointer)
__THRUST_DEFINE_HAS_NESTED_TYPE(has_const_void_pointer, const_void_pointer)
__THRUST_DEFINE_HAS_NESTED_TYPE(has_difference_type, difference_type)
__THRUST_DEFINE_HAS_NESTED_TYPE(has_size_type, size_type)
__THRUST_DEFINE_HAS_NESTED_TYPE(has_propagate_on_container_copy_assignment, propagate_on_container_copy_assignment)
__THRUST_DEFINE_HAS_NESTED_TYPE(has_propagate_on_container_move_assignment, propagate_on_container_move_assignment)
__THRUST_DEFINE_HAS_NESTED_TYPE(has_propagate_on_container_swap, propagate_on_container_swap)
__THRUST_DEFINE_HAS_NESTED_TYPE(has_system_type, system_type)

template<typename T>
  struct nested_pointer
{
  typedef typename T::pointer type;
};

template<typename T>
  struct nested_const_pointer
{
  typedef typename T::const_pointer type;
};

template<typename T>
  struct nested_reference
{
  typedef typename T::reference type;
};

template<typename T>
  struct nested_const_reference
{
  typedef typename T::const_reference type;
};

template<typename T>
  struct nested_void_pointer
{
  typedef typename T::void_pointer type;
};

template<typename T>
  struct nested_const_void_pointer
{
  typedef typename T::const_void_pointer type;
};

template<typename T>
  struct nested_difference_type
{
  typedef typename T::difference_type type;
};

template<typename T>
  struct nested_size_type
{
  typedef typename T::size_type type;
};

template<typename T>
  struct nested_propagate_on_container_copy_assignment
{
  typedef typename T::propagate_on_container_copy_assignment type;
};

template<typename T>
  struct nested_propagate_on_container_move_assignment
{
  typedef typename T::propagate_on_container_move_assignment type;
};

template<typename T>
  struct nested_propagate_on_container_swap
{
  typedef typename T::propagate_on_container_swap type;
};

template<typename T>
  struct nested_system_type
{
  typedef typename T::system_type type;
};

} // end allocator_traits_detail


template<typename Alloc>
  struct allocator_traits
{
  typedef Alloc allocator_type;

  typedef typename allocator_type::value_type value_type;

  typedef typename eval_if<
    allocator_traits_detail::has_pointer<allocator_type>::value,
    allocator_traits_detail::nested_pointer<allocator_type>,
    identity_<value_type*>
  >::type pointer;

  private:
    template<typename T>
      struct rebind_pointer
    {
      typedef typename pointer_traits<pointer>::template rebind<T>::other type;
    };

  public:

  typedef typename eval_if<
    allocator_traits_detail::has_const_pointer<allocator_type>::value,
    allocator_traits_detail::nested_const_pointer<allocator_type>,
    rebind_pointer<const value_type>
  >::type const_pointer;

  typedef typename eval_if<
    allocator_traits_detail::has_void_pointer<allocator_type>::value,
    allocator_traits_detail::nested_void_pointer<allocator_type>,
    rebind_pointer<void>
  >::type void_pointer;

  typedef typename eval_if<
    allocator_traits_detail::has_const_void_pointer<allocator_type>::value,
    allocator_traits_detail::nested_const_void_pointer<allocator_type>,
    rebind_pointer<const void>
  >::type const_void_pointer;

  typedef typename eval_if<
    allocator_traits_detail::has_difference_type<allocator_type>::value,
    allocator_traits_detail::nested_difference_type<allocator_type>,
    pointer_difference<pointer>
  >::type difference_type;

  typedef typename eval_if<
    allocator_traits_detail::has_size_type<allocator_type>::value,
    allocator_traits_detail::nested_size_type<allocator_type>,
    make_unsigned<difference_type>
  >::type size_type;

  typedef typename eval_if<
    allocator_traits_detail::has_propagate_on_container_copy_assignment<allocator_type>::value,
    allocator_traits_detail::nested_propagate_on_container_copy_assignment<allocator_type>,
    identity_<false_type>
  >::type propagate_on_container_copy_assignment;

  typedef typename eval_if<
    allocator_traits_detail::has_propagate_on_container_move_assignment<allocator_type>::value,
    allocator_traits_detail::nested_propagate_on_container_move_assignment<allocator_type>,
    identity_<false_type>
  >::type propagate_on_container_move_assignment;

  typedef typename eval_if<
    allocator_traits_detail::has_propagate_on_container_swap<allocator_type>::value,
    allocator_traits_detail::nested_propagate_on_container_swap<allocator_type>,
    identity_<false_type>
  >::type propagate_on_container_swap;

  typedef typename eval_if<
    allocator_traits_detail::has_system_type<allocator_type>::value,
    allocator_traits_detail::nested_system_type<allocator_type>,
    thrust::iterator_system<pointer>
  >::type system_type;

  // XXX rebind and rebind_traits are alias templates
  //     and so are omitted while c++11 is unavailable

  inline static pointer allocate(allocator_type &a, size_type n);

  inline static pointer allocate(allocator_type &a, size_type n, const_void_pointer hint);

  inline static void deallocate(allocator_type &a, pointer p, size_type n);

  // XXX should probably change T* to pointer below and then relax later

  template<typename T>
  inline __host__ __device__ static void construct(allocator_type &a, T *p);
  
  template<typename T, typename Arg1>
  inline __host__ __device__ static void construct(allocator_type &a, T *p, const Arg1 &arg1);

  template<typename T>
  inline __host__ __device__ static void destroy(allocator_type &a, T *p);

  inline static size_type max_size(const allocator_type &a);
}; // end allocator_traits


// XXX consider moving this non-standard functionality inside allocator_traits
template<typename Alloc>
  struct allocator_system
{
  // the type of the allocator's system
  typedef typename eval_if<
    allocator_traits_detail::has_system_type<Alloc>::value,
    allocator_traits_detail::nested_system_type<Alloc>,
    thrust::iterator_system<
      typename allocator_traits<Alloc>::pointer
    >
  >::type type;

  inline static type &get(Alloc &a);
};


} // end detail
} // end thrust

#include <thrust/detail/allocator/allocator_traits.inl>

