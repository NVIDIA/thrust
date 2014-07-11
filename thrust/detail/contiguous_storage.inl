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

#pragma once

#include <thrust/detail/contiguous_storage.h>
#include <thrust/detail/swap.h>
#include <thrust/detail/allocator/allocator_traits.h>
#include <thrust/detail/allocator/copy_construct_range.h>
#include <thrust/detail/allocator/default_construct_range.h>
#include <thrust/detail/allocator/destroy_range.h>
#include <thrust/detail/allocator/fill_construct_range.h>
#include <utility> // for use of std::swap in the WAR below

namespace thrust
{

namespace detail
{

__thrust_hd_warning_disable__
template<typename T, typename Alloc>
__host__ __device__
  contiguous_storage<T,Alloc>
    ::contiguous_storage(const Alloc &alloc)
      :m_allocator(alloc),
       m_begin(pointer(static_cast<T*>(0))),
       m_size(0)
{
  ;
} // end contiguous_storage::contiguous_storage()

__thrust_hd_warning_disable__
template<typename T, typename Alloc>
__host__ __device__
  contiguous_storage<T,Alloc>
    ::contiguous_storage(size_type n, const Alloc &alloc)
      :m_allocator(alloc),
       m_begin(pointer(static_cast<T*>(0))),
       m_size(0)
{
  allocate(n);
} // end contiguous_storage::contiguous_storage()

__thrust_hd_warning_disable__
template<typename T, typename Alloc>
__host__ __device__
  contiguous_storage<T,Alloc>
    ::~contiguous_storage(void)
{
  deallocate();
} // end contiguous_storage::~contiguous_storage()

template<typename T, typename Alloc>
__host__ __device__
  typename contiguous_storage<T,Alloc>::size_type
    contiguous_storage<T,Alloc>
      ::size(void) const
{
  return m_size;
} // end contiguous_storage::size()

template<typename T, typename Alloc>
__host__ __device__
  typename contiguous_storage<T,Alloc>::size_type
    contiguous_storage<T,Alloc>
      ::max_size(void) const
{
  return alloc_traits::max_size(m_allocator);
} // end contiguous_storage::max_size()

template<typename T, typename Alloc>
__host__ __device__
  typename contiguous_storage<T,Alloc>::iterator
    contiguous_storage<T,Alloc>
      ::begin(void)
{
  return m_begin;
} // end contiguous_storage::begin()

template<typename T, typename Alloc>
__host__ __device__
  typename contiguous_storage<T,Alloc>::const_iterator
    contiguous_storage<T,Alloc>
      ::begin(void) const
{
  return m_begin;
} // end contiguous_storage::begin()

template<typename T, typename Alloc>
__host__ __device__
  typename contiguous_storage<T,Alloc>::iterator
    contiguous_storage<T,Alloc>
      ::end(void)
{
  return m_begin + size();
} // end contiguous_storage::end()

template<typename T, typename Alloc>
__host__ __device__
  typename contiguous_storage<T,Alloc>::const_iterator
    contiguous_storage<T,Alloc>
      ::end(void) const
{
  return m_begin + size();
} // end contiguous_storage::end()

template<typename T, typename Alloc>
__host__ __device__
  typename contiguous_storage<T,Alloc>::reference
    contiguous_storage<T,Alloc>
      ::operator[](size_type n)
{
  return m_begin[n];
} // end contiguous_storage::operator[]()

template<typename T, typename Alloc>
__host__ __device__
  typename contiguous_storage<T,Alloc>::const_reference
    contiguous_storage<T,Alloc>
      ::operator[](size_type n) const
{
  return m_begin[n];
} // end contiguous_storage::operator[]()

template<typename T, typename Alloc>
__host__ __device__
  typename contiguous_storage<T,Alloc>::allocator_type
    contiguous_storage<T,Alloc>
      ::get_allocator(void) const
{
  return m_allocator;
} // end contiguous_storage::get_allocator()

template<typename T, typename Alloc>
__host__ __device__
  void contiguous_storage<T,Alloc>
    ::allocate(size_type n)
{
  if(n > 0)
  {
    m_begin = iterator(alloc_traits::allocate(m_allocator,n));
    m_size = n;
  } // end if
  else
  {
    m_begin = iterator(pointer(static_cast<T*>(0)));
    m_size = 0;
  } // end else
} // end contiguous_storage::allocate()

template<typename T, typename Alloc>
__host__ __device__
  void contiguous_storage<T,Alloc>
    ::deallocate(void)
{
  if(size() > 0)
  {
    alloc_traits::deallocate(m_allocator,m_begin.base(), size());
    m_begin = iterator(pointer(static_cast<T*>(0)));
    m_size = 0;
  } // end if
} // end contiguous_storage::deallocate()

template<typename T, typename Alloc>
__host__ __device__
  void contiguous_storage<T,Alloc>
    ::swap(contiguous_storage &x)
{
  thrust::swap(m_begin, x.m_begin);
  thrust::swap(m_size, x.m_size);

  thrust::swap(m_allocator, x.m_allocator);
} // end contiguous_storage::swap()

template<typename T, typename Alloc>
__host__ __device__
  void contiguous_storage<T,Alloc>
    ::default_construct_n(iterator first, size_type n)
{
  default_construct_range(m_allocator, first.base(), n);
} // end contiguous_storage::default_construct_n()

template<typename T, typename Alloc>
__host__ __device__
  void contiguous_storage<T,Alloc>
    ::uninitialized_fill_n(iterator first, size_type n, const value_type &x)
{
  fill_construct_range(m_allocator, first.base(), n, x);
} // end contiguous_storage::uninitialized_fill()

template<typename T, typename Alloc>
  template<typename System, typename InputIterator>
  __host__ __device__
    typename contiguous_storage<T,Alloc>::iterator
      contiguous_storage<T,Alloc>
        ::uninitialized_copy(thrust::execution_policy<System> &from_system, InputIterator first, InputIterator last, iterator result)
{
  return iterator(copy_construct_range(from_system, m_allocator, first, last, result.base()));
} // end contiguous_storage::uninitialized_copy()

template<typename T, typename Alloc>
  template<typename InputIterator>
  __host__ __device__
    typename contiguous_storage<T,Alloc>::iterator
      contiguous_storage<T,Alloc>
        ::uninitialized_copy(InputIterator first, InputIterator last, iterator result)
{
  // XXX assumes InputIterator's associated System is default-constructible
  typename thrust::iterator_system<InputIterator>::type from_system;

  return iterator(copy_construct_range(from_system, m_allocator, first, last, result.base()));
} // end contiguous_storage::uninitialized_copy()

template<typename T, typename Alloc>
  template<typename System, typename InputIterator, typename Size>
  __host__ __device__
    typename contiguous_storage<T,Alloc>::iterator
      contiguous_storage<T,Alloc>
        ::uninitialized_copy_n(thrust::execution_policy<System> &from_system, InputIterator first, Size n, iterator result)
{
  return iterator(copy_construct_range_n(from_system, m_allocator, first, n, result.base()));
} // end contiguous_storage::uninitialized_copy_n()

template<typename T, typename Alloc>
  template<typename InputIterator, typename Size>
  __host__ __device__
    typename contiguous_storage<T,Alloc>::iterator
      contiguous_storage<T,Alloc>
        ::uninitialized_copy_n(InputIterator first, Size n, iterator result)
{
  // XXX assumes InputIterator's associated System is default-constructible
  typename thrust::iterator_system<InputIterator>::type from_system;

  return iterator(copy_construct_range_n(from_system, m_allocator, first, n, result.base()));
} // end contiguous_storage::uninitialized_copy_n()

template<typename T, typename Alloc>
__host__ __device__
  void contiguous_storage<T,Alloc>
    ::destroy(iterator first, iterator last)
{
  destroy_range(m_allocator, first.base(), last - first);
} // end contiguous_storage::destroy()

} // end detail

template<typename T, typename Alloc>
__host__ __device__
  void swap(detail::contiguous_storage<T,Alloc> &lhs, detail::contiguous_storage<T,Alloc> &rhs)
{
  lhs.swap(rhs);
} // end swap()

} // end thrust

