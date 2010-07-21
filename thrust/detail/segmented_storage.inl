/*
 *  Copyright 2008-2010 NVIDIA Corporation
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

#include <thrust/detail/segmented_storage.h>
#include <thrust/for_each.h>
#include <thrust/reduce.h>
#include <functional>

namespace thrust
{

namespace detail
{


template<typename T, typename Allocator>
  segmented_storage<T,Allocator>
    ::segmented_storage(void)
      :m_storage(choose_number_of_segments())
{
  ;
} // end segmented_storage::segmented_storage()


template<typename T, typename Allocator>
  segmented_storage<T,Allocator>
    ::segmented_storage(size_type n)
      :m_storage(choose_number_of_segments())
{
  allocate(n);
} // end segmented_storage::segmented_storage()


template<typename T, typename Allocator>
  segmented_storage<T,Allocator>
    ::~segmented_storage(void)
{
  deallocate();
} // end segmented_storage::~segmented_storage()


template<typename T, typename Allocator>
  typename segmented_storage<T,Allocator>::size_type
    segmented_storage<T,Allocator>
      ::size(void) const
{
  // return the sum of the sizes of the individual storages
  return thrust::reduce(m_storage.begin(), m_storage.end(), std::mem_fun_ref(&storage_type::size));
} // end segmented_storage::size()

template<typename T, typename Allocator>
  typename segmented_storage<T,Allocator>::size_type
    segmented_storage<T,Allocator>
      ::max_size(void) const
{
  // return the sum of the max_sizes of the individual storages
  return thrust::reduce(m_storage.begin(), m_storage.end(), std::mem_fun_ref(&storage_type::max_size));
} // end segmented_storage::max_size()


template<typename T, typename Allocator>
  typename segmented_storage<T,Allocator>::iterator
    segmented_storage<T,Allocator>
      ::begin(void)
{
  return thrust::detail::make_segmented_iterator(m_storage.begin(), m_storage.end());
} // end segmented_storage::begin()


template<typename T, typename Allocator>
  typename segmented_storage<T,Allocator>::const_iterator
    segmented_storage<T,Allocator>
      ::begin(void) const
{
  return thrust::detail::make_segmented_iterator(m_storage.begin(), m_storage.end());
} // end segmented_storage::begin()


template<typename T, typename Allocator>
  typename segmented_storage<T,Allocator>::iterator
    segmented_storage<T,Allocator>
      ::end(void)
{
  return thrust::detail::make_segmented_iterator(m_storage.end(), m_storage.end());
} // end segmented_storage::end()


template<typename T, typename Allocator>
  typename segmented_storage<T,Allocator>::const_iterator
    segmented_storage<T,Allocator>
      ::end(void) const
{
  return thrust::detail::make_segmented_iterator(m_storage.end(), m_storage.end());
} // end segmented_storage::end()


template<typename T, typename Allocator>
  typename segmented_storage<T,Allocator>::allocator_type
    segmented_storage<T,Allocator>
      ::get_allocator(void) const
{
  // return the first storage's allocator i guess
  return m_storage[0].get_allocator();
} // end segmented_storage::get_allocator()


template<typename T, typename Allocator>
  void segmented_storage<T,Allocator>
    ::allocate(size_type n)
{
  const size_type m = m_storage.size();

  // break up n into m chunks of n/m (except possibly the last one)
  // if n is small, just give it all to the first segment
  const size_type size_per_segment = (n > m) ? (n / m) : n;

  // XXX might want to parallelize this with for_each
  size_type i = 0;
  while(n > 0)
  {
    const size_type size_to_allocate = thrust::min(size_per_segment, n);

    m_storage[i].allocate(size_to_allocate);

    n -= size_to_allocate;
    ++i;
  } // end while
} // segmented_storage::allocate()


template<typename T, typename Allocator>
  void segmented_storage<T,Allocator>
    ::deallocate(void)
{
  thrust::for_each(m_storage.begin(), m_storage.end(), std::mem_fun_ref(&storage_type::deallocate));
} // end segmented_storage::deallocate();


template<typename T, typename Allocator>
  void segmented_storage<T,Allocator>
    ::swap(segmented_storage &x)
{
  thrust::swap(m_storage, x.m_storage);
} // end segmented_storage::swap()


template<typename T, typename Allocator>
  typename segmented_storage<T,Allocator>::size_type
    segmented_storage<T,Allocator>
      ::choose_number_of_segments(void)
{
  return 20;
} // end segmented_storage::choose_number_of_segments()


} // end detail

} // end thrust

