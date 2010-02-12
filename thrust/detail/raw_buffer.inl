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

#include <thrust/detail/raw_buffer.h>
#include <thrust/distance.h>
#include <thrust/copy.h>


namespace thrust
{

namespace detail
{


template<typename T, typename Space>
  raw_buffer<T,Space>
    ::raw_buffer(size_type n)
{
  m_begin = m_allocator.allocate(n);
  m_end = m_begin + n;
} // end raw_buffer::raw_buffer()


template<typename T, typename Space>
  template<typename InputIterator>
    raw_buffer<T,Space>
      ::raw_buffer(InputIterator first, InputIterator last)
{
  size_type n = thrust::distance(first,last);
  m_begin = m_allocator.allocate(n);
  m_end = m_begin + n;
  thrust::copy(first, last, begin());
} // end raw_buffer::raw_buffer()


template<typename T, typename Space>
  raw_buffer<T,Space>
    ::~raw_buffer(void)
{
  if(size() > 0)
  {
    m_allocator.deallocate(&*m_begin, size());
    m_end = m_begin;
  } // end if
} // end raw_buffer::~raw_buffer()


template<typename T, typename Space>
  typename raw_buffer<T,Space>::size_type raw_buffer<T,Space>
    ::size(void) const
{
  return m_end - m_begin;
} // end raw_buffer::size()


template<typename T, typename Space>
  typename raw_buffer<T,Space>::iterator raw_buffer<T,Space>
    ::begin(void)
{
  return m_begin;
} // end raw_buffer::begin()


template<typename T, typename Space>
  typename raw_buffer<T,Space>::const_iterator raw_buffer<T,Space>
    ::begin(void) const
{
  return m_begin;
} // end raw_buffer::begin()


template<typename T, typename Space>
  typename raw_buffer<T,Space>::const_iterator raw_buffer<T,Space>
    ::cbegin(void) const
{
  return m_begin;
} // end raw_buffer::cbegin()


template<typename T, typename Space>
  typename raw_buffer<T,Space>::iterator raw_buffer<T,Space>
    ::end(void)
{
  return m_end;
} // end raw_buffer::end()


template<typename T, typename Space>
  typename raw_buffer<T,Space>::const_iterator raw_buffer<T,Space>
    ::end(void) const
{
  return m_end;
} // end raw_buffer::end()


template<typename T, typename Space>
  typename raw_buffer<T,Space>::const_iterator raw_buffer<T,Space>
    ::cend(void) const
{
  return m_end;
} // end raw_buffer::cend()


template<typename T, typename Space>
  typename raw_buffer<T,Space>::reference raw_buffer<T,Space>
    ::operator[](size_type n)
{
  return m_begin[n];
} // end raw_buffer::operator[]()


template<typename T, typename Space>
  typename raw_buffer<T,Space>::const_reference raw_buffer<T,Space>
    ::operator[](size_type n) const
{
  return m_begin[n];
} // end raw_buffer::operator[]()


} // end detail

} // end thrust

