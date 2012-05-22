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
#include <thrust/uninitialized_fill.h>
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


__THRUST_DEFINE_HAS_MEMBER_FUNCTION1(has_member_construct1_impl, construct);

template<typename Alloc, typename T>
  struct has_member_construct1
    : has_member_construct1_impl<
        Alloc,
        void,
        T*
      >
{};

template<typename Alloc, typename T>
  inline __host__ __device__
    typename enable_if<
      has_member_construct1<Alloc,T>::value
    >::type
      construct(Alloc &a, T *p)
{
  a.construct(p);
}

template<typename Alloc, typename T>
  inline __host__ __device__
    typename disable_if<
      has_member_construct1<Alloc,T>::value
    >::type
      construct(Alloc &a, T *p)
{
  ::new(static_cast<void*>(p)) T();
}


__THRUST_DEFINE_HAS_MEMBER_FUNCTION2(has_member_construct2_impl, construct);

template<typename Alloc, typename T, typename Arg1>
  struct has_member_construct2
    : has_member_construct2_impl<
        Alloc,
        void,
        T*,
        const Arg1 &
      >
{};

template<typename Alloc, typename T, typename Arg1>
  inline __host__ __device__
    typename enable_if<
      has_member_construct2<Alloc,T,Arg1>::value
    >::type
      construct(Alloc &a, T *p, const Arg1 &arg1)
{
  a.construct(p,arg1);
}

template<typename Alloc, typename T, typename Arg1>
  inline __host__ __device__
    typename disable_if<
      has_member_construct2<Alloc,T,Arg1>::value
    >::type
      construct(Alloc &, T *p, const Arg1 &arg1)
{
  ::new(static_cast<void*>(p)) T(arg1);
}


__THRUST_DEFINE_HAS_MEMBER_FUNCTION1(has_member_destroy1_impl, destroy);

template<typename Alloc, typename T>
  struct has_member_destroy1
    : has_member_destroy1_impl<
        Alloc,
        void,
        T*
      >
{};

template<typename Alloc, typename T>
  inline __host__ __device__
    typename enable_if<
      has_member_destroy1<Alloc,T>::value
    >::type
      destroy(Alloc &a, T *p)
{
  a.destroy(p);
}

template<typename Alloc, typename T>
  inline __host__ __device__
    typename disable_if<
      has_member_destroy1<Alloc,T>::value
    >::type
      destroy(Alloc &, T *p)
{
  p->~T();
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
  template<typename T>
    void allocator_traits<Alloc>
      ::construct(allocator_type &a, T *p)
{
  return allocator_traits_detail::construct(a,p);
}

template<typename Alloc>
  template<typename T, typename Arg1>
    void allocator_traits<Alloc>
      ::construct(allocator_type &a, T *p, const Arg1 &arg1)
{
  return allocator_traits_detail::construct(a,p,arg1);
}

template<typename Alloc>
  template<typename T>
    void allocator_traits<Alloc>
      ::destroy(allocator_type &a, T *p)
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

template<typename Allocator, typename Arg1>
  struct construct2_via_allocator
{
  Allocator &a;
  Arg1 arg;

  construct2_via_allocator(Allocator &a, const Arg1 &arg)
    : a(a), arg(arg)
  {}

  template<typename T>
  inline __host__ __device__
  void operator()(T &x)
  {
    allocator_traits<Allocator>::construct(a, &x, arg);
  }
};


template<typename Allocator, typename Pointer, typename Size, typename T>
  typename enable_if<
    has_member_construct2<
      Allocator,
      typename pointer_element<Pointer>::type,
      T
    >::value
  >::type
    fill_construct_range(Allocator &a, Pointer p, Size n, const T &value)
{
  thrust::for_each_n(p, n, construct2_via_allocator<Allocator,T>(a, value));
}


template<typename Allocator, typename Pointer, typename Size, typename T>
  typename disable_if<
    has_member_construct2<
      Allocator,
      typename pointer_element<Pointer>::type,
      T
    >::value
  >::type
    fill_construct_range(Allocator &, Pointer p, Size n, const T &value)
{
  thrust::uninitialized_fill_n(p, n, value);
}


} // end allocator_traits_detail


template<typename Alloc, typename Pointer, typename Size, typename T>
  void fill_construct_range(Alloc &a, Pointer p, Size n, const T &value)
{
  return allocator_traits_detail::fill_construct_range(a,p,n,value);
}


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

namespace allocator_traits_detail
{


template<typename Allocator>
  struct construct1_via_allocator
{
  Allocator &a;

  construct1_via_allocator(Allocator &a)
    : a(a)
  {}

  template<typename T>
  inline __host__ __device__
  void operator()(T &x)
  {
    allocator_traits<Allocator>::construct(a, &x);
  }
};


template<typename Allocator, typename Pointer, typename Size>
  typename enable_if<
    has_member_construct1<
      Allocator,
      typename pointer_element<Pointer>::type
    >::value
  >::type
    default_construct_range(Allocator &a, Pointer p, Size n)
{
  thrust::for_each_n(p, n, construct1_via_allocator<Allocator>(a));
}


template<typename Allocator, typename Pointer, typename Size>
  typename disable_if<
    has_member_construct1<
      Allocator,
      typename pointer_element<Pointer>::type
    >::value
  >::type
    default_construct_range(Allocator &, Pointer p, Size n)
{
  thrust::uninitialized_fill_n(p, n, typename pointer_element<Pointer>::type());
}


} // end allocator_traits_detail

template<typename Allocator, typename Pointer, typename Size>
  void default_construct_range(Allocator &a, Pointer p, Size n)
{
  return allocator_traits_detail::default_construct_range(a,p,n);
}

} // end detail
} // end thrust

