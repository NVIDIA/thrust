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

#include <thrust/detail/placed_allocator.h>
#include <stdexcept>

namespace thrust
{

namespace detail
{


template<typename T, typename Allocator>
  placed_allocator<T,Allocator>
    ::placed_allocator(place p)
      : m_allocator(),
        m_place(p)
{
} // end placed_allocator::placed_allocator()


template<typename T, typename Allocator>
  placed_allocator<T,Allocator>
    ::~placed_allocator(void)
{
} // end placed_allocator::placed_allocator()


template<typename T, typename Allocator>
  place placed_allocator<T,Allocator>
    ::get_place(void) const
{
  return m_place;
} // end placed_allocator::get_place()


template<typename T, typename Allocator>
  void placed_allocator<T,Allocator>
    ::set_place(place p)
{
  m_place = p;
} // end placed_allocator::set_place()


template<typename T, typename Allocator>
  typename placed_allocator<T,Allocator>::pointer
    placed_allocator<T,Allocator>
      ::address(reference r)
{
  return m_allocator.address(r);
} // end placed_allocator::address()


template<typename T, typename Allocator>
  typename placed_allocator<T,Allocator>::const_pointer
    placed_allocator<T,Allocator>
      ::address(const_reference r)
{
  return m_allocator.address(r);
} // end placed_allocator::address()


template<typename T, typename Allocator>
  typename placed_allocator<T,Allocator>::pointer
    placed_allocator<T,Allocator>
      ::allocate(size_type cnt,
                 const_pointer x)
{
  thrust::detail::push_place(get_place());

  pointer result;

  try
  {
    result = thrust::detail::make_placed_iterator(m_allocator.allocate(cnt, x.base()), get_place());
  }
  catch(...)
  {
    // pop before rethrowing
    thrust::detail::pop_place();
    throw;
  }

  thrust::detail::pop_place();

  return result;
} // end placed_pointer::allocate()


template<typename T, typename Allocator>
  void placed_allocator<T,Allocator>
    ::deallocate(pointer p,
                 size_type cnt)
{
  // XXX should we check that p's place == m_place?
  thrust::detail::push_place(p.get_place());

  try
  {
    m_allocator.deallocate(p.base(),cnt);
  }
  catch(...)
  {
    // pop before rethrowing
    thrust::detail::pop_place();
    throw;
  }

  thrust::detail::pop_place();
} // end placed_pointer::deallocate()


template<typename T, typename Allocator>
  typename placed_allocator<T,Allocator>::size_type
    placed_allocator<T,Allocator>
      ::max_size(void) const
{
  return m_allocator.max_size();
} // end placed_allocator::max_size()


template<typename T, typename Allocator>
  bool placed_allocator<T,Allocator>
    ::operator==(placed_allocator const &x)
{
  return get_place() == x.get_place();
} // end placed_allocator::operator==()


template<typename T, typename Allocator>
  bool placed_allocator<T,Allocator>
    ::operator!=(placed_allocator const &x)
{
  return !(*this == x);
} // end placed_allocator::operator==()


} // end detail

} // end thrust

