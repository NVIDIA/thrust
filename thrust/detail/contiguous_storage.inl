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

#include <thrust/detail/contiguous_storage.h>
#include <thrust/swap.h>
#include <utility> // for use of std::swap in the WAR below

namespace thrust
{

namespace detail
{

template<typename T, typename Alloc>
  contiguous_storage<T,Alloc>
    ::contiguous_storage(void)
      :m_allocator(),
       m_begin(pointer(static_cast<T*>(0))),
       m_size(0)
{
  ;
} // end contiguous_storage::contiguous_storage()

template<typename T, typename Alloc>
  contiguous_storage<T,Alloc>
    ::contiguous_storage(size_type n)
      :m_allocator(),
       m_begin(pointer(static_cast<T*>(0))),
       m_size(0)
{
  allocate(n);
} // end contiguous_storage::contiguous_storage()

template<typename T, typename Alloc>
  contiguous_storage<T,Alloc>
    ::~contiguous_storage(void)
{
  deallocate();
} // end contiguous_storage::~contiguous_storage()

template<typename T, typename Alloc>
  typename contiguous_storage<T,Alloc>::size_type
    contiguous_storage<T,Alloc>
      ::size(void) const
{
  return m_size;
} // end contiguous_storage::size()

template<typename T, typename Alloc>
  typename contiguous_storage<T,Alloc>::size_type
    contiguous_storage<T,Alloc>
      ::max_size(void) const
{
  return m_allocator.max_size();
} // end contiguous_storage::max_size()

template<typename T, typename Alloc>
  typename contiguous_storage<T,Alloc>::iterator
    contiguous_storage<T,Alloc>
      ::begin(void)
{
  return m_begin;
} // end contiguous_storage::begin()

template<typename T, typename Alloc>
  typename contiguous_storage<T,Alloc>::const_iterator
    contiguous_storage<T,Alloc>
      ::begin(void) const
{
  return m_begin;
} // end contiguous_storage::begin()

template<typename T, typename Alloc>
  typename contiguous_storage<T,Alloc>::iterator
    contiguous_storage<T,Alloc>
      ::end(void)
{
  return m_begin + size();
} // end contiguous_storage::end()

template<typename T, typename Alloc>
  typename contiguous_storage<T,Alloc>::const_iterator
    contiguous_storage<T,Alloc>
      ::end(void) const
{
  return m_begin + size();
} // end contiguous_storage::end()

template<typename T, typename Alloc>
  typename contiguous_storage<T,Alloc>::reference
    contiguous_storage<T,Alloc>
      ::operator[](size_type n)
{
  return m_begin[n];
} // end contiguous_storage::operator[]()

template<typename T, typename Alloc>
  typename contiguous_storage<T,Alloc>::const_reference
    contiguous_storage<T,Alloc>
      ::operator[](size_type n) const
{
  return m_begin[n];
} // end contiguous_storage::operator[]()

template<typename T, typename Alloc>
  typename contiguous_storage<T,Alloc>::allocator_type
    contiguous_storage<T,Alloc>
      ::get_allocator(void) const
{
  return m_allocator;
} // end contiguous_storage::get_allocator()

template<typename T, typename Alloc>
  void contiguous_storage<T,Alloc>
    ::allocate(size_type n)
{
  if(n > 0)
  {
    m_begin = iterator(m_allocator.allocate(n));
    m_size = n;
  } // end if
  else
  {
    m_begin = iterator(pointer(static_cast<T*>(0)));
    m_size = 0;
  } // end else
} // end contiguous_storage::allocate()

template<typename T, typename Alloc>
  void contiguous_storage<T,Alloc>
    ::deallocate(void)
{
  if(size() > 0)
  {
    m_allocator.deallocate(m_begin.base(), size());
    m_begin = iterator(pointer(static_cast<T*>(0)));
    m_size = 0;
  } // end if
} // end contiguous_storage::deallocate()

template<typename T, typename Alloc>
  void contiguous_storage<T,Alloc>
    ::swap(contiguous_storage &x)
{
  thrust::swap(m_begin, x.m_begin);
  thrust::swap(m_size, x.m_size);

  // XXX WAR nvcc 4.0's "calling a __host__ function from a __host__ __device__ function is not allowed" warning
  //thrust::swap(m_allocator, x.m_allocator);
  std::swap(m_allocator, x.m_allocator);
} // end contiguous_storage::swap()

} // end detail

template<typename T, typename Alloc>
  void swap(detail::contiguous_storage<T,Alloc> &lhs, detail::contiguous_storage<T,Alloc> &rhs)
{
  lhs.swap(rhs);
} // end swap()

} // end thrust

