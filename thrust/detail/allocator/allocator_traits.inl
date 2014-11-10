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
#include <thrust/detail/allocator/allocator_traits.h>
#include <thrust/detail/type_traits/is_call_possible.h>
#include <thrust/detail/integer_traits.h>
#include <new>

namespace thrust
{
namespace detail
{
namespace allocator_traits_detail
{

__THRUST_DEFINE_IS_CALL_POSSIBLE(has_member_allocate_with_hint_impl, allocate)

template<typename Alloc>
  class has_member_allocate_with_hint
{
  typedef typename allocator_traits<Alloc>::pointer            pointer;
  typedef typename allocator_traits<Alloc>::size_type          size_type;
  typedef typename allocator_traits<Alloc>::const_void_pointer const_void_pointer;

  public:
    typedef typename has_member_allocate_with_hint_impl<Alloc, pointer(size_type,const_void_pointer)>::type type;
    static const bool value = type::value;
};

template<typename Alloc>
__host__ __device__
  typename enable_if<
    has_member_allocate_with_hint<Alloc>::value,
    typename allocator_traits<Alloc>::pointer
  >::type
    allocate(Alloc &a, typename allocator_traits<Alloc>::size_type n, typename allocator_traits<Alloc>::const_void_pointer hint)
{
  return a.allocate(n,hint);
}

template<typename Alloc>
__host__ __device__
  typename disable_if<
    has_member_allocate_with_hint<Alloc>::value,
    typename allocator_traits<Alloc>::pointer
  >::type
    allocate(Alloc &a, typename allocator_traits<Alloc>::size_type n, typename allocator_traits<Alloc>::const_void_pointer)
{
  return a.allocate(n);
}


__THRUST_DEFINE_IS_CALL_POSSIBLE(has_member_construct1_impl, construct)

template<typename Alloc, typename T>
  struct has_member_construct1
    : has_member_construct1_impl<Alloc, void(T*)>
{};

__thrust_hd_warning_disable__
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


__THRUST_DEFINE_IS_CALL_POSSIBLE(has_member_construct2_impl, construct)

template<typename Alloc, typename T, typename Arg1>
  struct has_member_construct2
    : has_member_construct2_impl<Alloc, void(T*,const Arg1 &)>
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


__THRUST_DEFINE_IS_CALL_POSSIBLE(has_member_destroy_impl, destroy)

template<typename Alloc, typename T>
  struct has_member_destroy
    : has_member_destroy_impl<Alloc, void(T*)>
{};

template<typename Alloc, typename T>
  inline __host__ __device__
    typename enable_if<
      has_member_destroy<Alloc,T>::value
    >::type
      destroy(Alloc &a, T *p)
{
  a.destroy(p);
}

template<typename Alloc, typename T>
  inline __host__ __device__
    typename disable_if<
      has_member_destroy<Alloc,T>::value
    >::type
      destroy(Alloc &, T *p)
{
  p->~T();
}


__THRUST_DEFINE_IS_CALL_POSSIBLE(has_member_max_size_impl, max_size)

template<typename Alloc>
  class has_member_max_size
{
  typedef typename allocator_traits<Alloc>::size_type size_type;

  public:
    typedef typename has_member_max_size_impl<Alloc, size_type(void)>::type type;
    static const bool value = type::value;
};

template<typename Alloc>
__host__ __device__
  typename enable_if<
    has_member_max_size<Alloc>::value,
    typename allocator_traits<Alloc>::size_type
  >::type
    max_size(const Alloc &a)
{
  return a.max_size();
}

template<typename Alloc>
__host__ __device__
  typename disable_if<
    has_member_max_size<Alloc>::value,
    typename allocator_traits<Alloc>::size_type
  >::type
    max_size(const Alloc &a)
{
  typedef typename allocator_traits<Alloc>::size_type size_type;
  return thrust::detail::integer_traits<size_type>::const_max;
}

template<typename Alloc>
__host__ __device__
  typename enable_if<
    has_member_system<Alloc>::value,
    typename allocator_system<Alloc>::type &
  >::type
    system(Alloc &a)
{
  // return the allocator's system
  return a.system();
}

template<typename Alloc>
__host__ __device__
  typename disable_if<
    has_member_system<Alloc>::value,
    typename allocator_system<Alloc>::type
  >::type
    system(Alloc &a)
{
  // return a copy of a default-constructed system
  typename allocator_system<Alloc>::type result;
  return result;
}


} // end allocator_traits_detail


template<typename Alloc>
__host__ __device__
  typename allocator_traits<Alloc>::pointer
    allocator_traits<Alloc>
      ::allocate(Alloc &a, typename allocator_traits<Alloc>::size_type n)
{
  struct workaround_warnings
  {
    __thrust_hd_warning_disable__
    static __host__ __device__ 
    typename allocator_traits<Alloc>::pointer
      allocate(Alloc &a, typename allocator_traits<Alloc>::size_type n)
    {
      return a.allocate(n);
    }
  };

  return workaround_warnings::allocate(a, n);
}

template<typename Alloc>
__host__ __device__
  typename allocator_traits<Alloc>::pointer
    allocator_traits<Alloc>
      ::allocate(Alloc &a, typename allocator_traits<Alloc>::size_type n, typename allocator_traits<Alloc>::const_void_pointer hint)
{
  return allocator_traits_detail::allocate(a, n, hint);
}

template<typename Alloc>
__host__ __device__
  void allocator_traits<Alloc>
    ::deallocate(Alloc &a, typename allocator_traits<Alloc>::pointer p, typename allocator_traits<Alloc>::size_type n)
{
  struct workaround_warnings
  {
    __thrust_hd_warning_disable__
    static __host__ __device__
    void deallocate(Alloc &a, typename allocator_traits<Alloc>::pointer p, typename allocator_traits<Alloc>::size_type n)
    {
      return a.deallocate(p,n);
    }
  };

  return workaround_warnings::deallocate(a,p,n);
}

template<typename Alloc>
  template<typename T>
  __host__ __device__
    void allocator_traits<Alloc>
      ::construct(allocator_type &a, T *p)
{
  return allocator_traits_detail::construct(a,p);
}

template<typename Alloc>
  template<typename T, typename Arg1>
  __host__ __device__
    void allocator_traits<Alloc>
      ::construct(allocator_type &a, T *p, const Arg1 &arg1)
{
  return allocator_traits_detail::construct(a,p,arg1);
}

template<typename Alloc>
  template<typename T>
  __host__ __device__
    void allocator_traits<Alloc>
      ::destroy(allocator_type &a, T *p)
{
  return allocator_traits_detail::destroy(a,p);
}

template<typename Alloc>
__host__ __device__
  typename allocator_traits<Alloc>::size_type
    allocator_traits<Alloc>
      ::max_size(const allocator_type &a)
{
  return allocator_traits_detail::max_size(a);
}

template<typename Alloc>
__host__ __device__
  typename allocator_system<Alloc>::get_result_type
    allocator_system<Alloc>
      ::get(Alloc &a)
{
  return allocator_traits_detail::system(a);
}


} // end detail
} // end thrust

