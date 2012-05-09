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

#include <thrust/detail/config.h>
#include <thrust/detail/allocator/allocator_traits.h>
#include <thrust/detail/type_traits/has_member_function.h>
#include <thrust/detail/type_traits/pointer_traits.h>
#include <thrust/detail/raw_pointer_cast.h>
#include <thrust/for_each.h>
#include <new>
#include <limits>

namespace thrust
{
namespace detail
{
namespace allocator_traits_detail
{

__THRUST_DEFINE_HAS_MEMBER_FUNCTION2(has_member_allocate_with_hint_impl, allocate);

template<typename Alloc>
  struct has_member_allocate_with_hint
    : has_member_allocate_with_hint_impl<
        Alloc,
        typename allocator_traits<Alloc>::pointer, 
        typename allocator_traits<Alloc>::size_type,
        typename allocator_traits<Alloc>::const_void_pointer
      >
{};

template<typename Alloc>
  typename enable_if<
    has_member_allocate_with_hint<Alloc>::value,
    typename allocator_traits<Alloc>::pointer
  >::type
    allocate(Alloc &a, typename allocator_traits<Alloc>::size_type n, typename allocator_traits<Alloc>::const_void_pointer hint)
{
  return a.allocate(n,hint);
}

template<typename Alloc>
  typename disable_if<
    has_member_allocate_with_hint<Alloc>::value,
    typename allocator_traits<Alloc>::pointer
  >::type
    allocate(Alloc &a, typename allocator_traits<Alloc>::size_type n, typename allocator_traits<Alloc>::const_void_pointer)
{
  return a.allocate(n);
}


__THRUST_DEFINE_HAS_MEMBER_FUNCTION2(has_member_construct2_impl, construct);

template<typename Alloc, typename Arg1>
  struct has_member_construct2
    : has_member_construct2_impl<
        Alloc,
        void,
        typename allocator_traits<Alloc>::pointer,
        Arg1
      >
{};

template<typename Alloc, typename Arg1>
  inline __host__ __device__
    typename enable_if<
      has_member_construct2<Alloc,Arg1>::value
    >::type
      construct(Alloc &a, typename allocator_traits<Alloc>::pointer p, const Arg1 &arg1)
{
  a.construct(p,arg1);
}

template<typename Alloc, typename Arg1>
  inline __host__ __device__
    typename disable_if<
      has_member_construct2<Alloc,Arg1>::value
    >::type
      construct(Alloc &, typename allocator_traits<Alloc>::pointer p, const Arg1 &arg1)
{
  typedef typename allocator_traits<Alloc>::value_type T;
  ::new(static_cast<void*>(thrust::raw_pointer_cast(p))) T(arg1);
}


__THRUST_DEFINE_HAS_MEMBER_FUNCTION1(has_member_destroy1_impl, destroy);

template<typename Alloc>
  struct has_member_destroy1
    : has_member_destroy1_impl<
        Alloc,
        void,
        typename allocator_traits<Alloc>::pointer
      >
{};

template<typename Alloc>
  inline __host__ __device__
    typename enable_if<
      has_member_destroy1<Alloc>::value
    >::type
      destroy(Alloc &a, typename allocator_traits<Alloc>::pointer p)
{
  a.destroy(p);
}

template<typename Alloc>
  inline __host__ __device__
    typename disable_if<
      has_member_destroy1<Alloc>::value
    >::type
      destroy(Alloc &, typename allocator_traits<Alloc>::pointer p)
{
  typedef typename allocator_traits<Alloc>::value_type T;
  thrust::raw_pointer_cast(p)->~T();
}


__THRUST_DEFINE_HAS_MEMBER_FUNCTION0(has_member_max_size0_impl, max_size);

template<typename Alloc>
  struct has_member_max_size
    : has_member_max_size0_impl<
        Alloc,
        typename allocator_traits<Alloc>::size_type
      >
{};

template<typename Alloc>
  typename enable_if<
    has_member_max_size<Alloc>::value,
    typename allocator_traits<Alloc>::size_type
  >::type
    max_size(const Alloc &a)
{
  return a.max_size();
}

template<typename Alloc>
  typename disable_if<
    has_member_max_size<Alloc>::value,
    typename allocator_traits<Alloc>::size_type
  >::type
    max_size(const Alloc &a)
{
  typedef typename allocator_traits<Alloc>::size_type size_type;
  return std::numeric_limits<size_type>::max();
}


} // end allocator_traits_detail


template<typename Alloc>
  typename allocator_traits<Alloc>::pointer
    allocator_traits<Alloc>
      ::allocate(Alloc &a, typename allocator_traits<Alloc>::size_type n)
{
  return a.allocate(n);
}

template<typename Alloc>
  typename allocator_traits<Alloc>::pointer
    allocator_traits<Alloc>
      ::allocate(Alloc &a, typename allocator_traits<Alloc>::size_type n, typename allocator_traits<Alloc>::const_void_pointer hint)
{
  return allocator_traits_detail::allocate(a, n, hint);
}

template<typename Alloc>
  void allocator_traits<Alloc>
    ::deallocate(Alloc &a, typename allocator_traits<Alloc>::pointer p, typename allocator_traits<Alloc>::size_type n)
{
  return a.deallocate(p,n);
}

template<typename Alloc>
  template<typename Arg1>
    void allocator_traits<Alloc>
      ::construct(allocator_type &a, pointer p, const Arg1 &arg1)
{
  return allocator_traits_detail::construct(a,p,arg1);
}

template<typename Alloc>
  void allocator_traits<Alloc>
    ::destroy(allocator_type &a, pointer p)
{
  return allocator_traits_detail::destroy(a,p);
}

template<typename Alloc>
  typename allocator_traits<Alloc>::size_type
    allocator_traits<Alloc>
      ::max_size(const allocator_type &a)
{
  return allocator_traits_detail::max_size(a);
}

namespace allocator_traits_detail
{

template<typename Pointer, typename Size>
  typename enable_if<
    has_trivial_destructor<
      typename pointer_element<Pointer>::type
    >::value
  >::type
    destroy_range(Pointer, Size)
{
  // no op
}

struct destroyer
{
  template<typename T>
  inline __host__ __device__
  void operator()(T &x)
  {
    x.~T();
  }
};

template<typename Pointer, typename Size>
  typename disable_if<
    has_trivial_destructor<
      typename pointer_element<Pointer>::type
    >::value
  >::type
    destroy_range(Pointer p, Size n)
{
  thrust::for_each_n(p, n, destroyer());
}

} // end allocator_traits_detail

template<typename Alloc>
  void allocator_traits<Alloc>
    ::destroy(Alloc &, typename allocator_traits<Alloc>::pointer p, typename allocator_traits<Alloc>::size_type n)
{
  return allocator_traits_detail::destroy_range(p,n);
}


} // end detail
} // end thrust

