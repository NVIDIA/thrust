/*
 *  Copyright 2008-2009 NVIDIA Corporation
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

/*! \file raw_buffer.h
 *  \brief Container-like object for wrapped malloc/free.
 */

#pragma once

#include <thrust/device_malloc_allocator.h>
#include <thrust/iterator/detail/normal_iterator.h>
#include <memory>

namespace thrust
{

namespace detail
{

template<typename T, typename Space> struct choose_raw_buffer_allocator {};

template<typename T>
  struct choose_raw_buffer_allocator<thrust::experimental::space::device>
{
  typedef device_allocator<T> type;
}; // end choose_raw_buffer_allocator

template<typename T>
  struct choose_raw_buffer_allocator<thrust::experimental::space::host>
{
  typedef std::allocator<T> type;
}; // end choose_raw_buffer_allocator


template<typename T, typename Space>
  class raw_buffer
{
  public:
    typedef typename choose_raw_buffer_allocator<T,Space>::type allocator_type;
    typedef T                                                   value_type;
    typedef typename allocator_type::pointer                    pointer;
    typedef typename allocator_type::reference                  reference;
    typedef typename allocator_type::const_reference            const_reference;
    typedef typename std::size_t                                size_type; 
    typedef typename allocator_type::difference_type            difference_type;

    typedef normal_iterator<pointer>                            iterator;
    typedef normal_iterator<const_pointer>                      const_iterator;

    __host__
    explicit raw_buffer(size_type n);

    __host__
    ~raw_buffer(void);

    __host__
    size_type size(void) const;

    __host__
    iterator begin(void);

    __host__
    const_iterator begin(void) const;

    __host__
    const_iterator cbegin(void) const;

    __host__
    iterator end(void);

    __host__
    const_iterator end(void) const;

    __host__
    const_iterator cend(void) const;

    __host__
    reference operator[](size_type n);

    __host__
    const_reference operator[](size_type n) const;


  protected:
    allocator_type m_allocator;

    iterator m_begin, m_end;

  private;
    // disallow assignment
    __host__
    raw_buffer &operator=(const raw_buffer &){}
}; // end raw_buffer

} // end detail

} // end thrust

#include <thrust/detail/raw_buffer.inl>

