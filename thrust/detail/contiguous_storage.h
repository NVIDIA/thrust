/*
 *  Copyright 2008-2011 NVIDIA Corporation
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

#include <thrust/iterator/detail/normal_iterator.h>

namespace thrust
{

namespace detail
{

template<typename T, typename Alloc>
  class contiguous_storage
{
  public:
    typedef Alloc                                          allocator_type;
    typedef T                                              value_type;
    typedef typename allocator_type::pointer               pointer;
    typedef typename allocator_type::const_pointer         const_pointer;
    typedef typename allocator_type::reference             reference;
    typedef typename allocator_type::const_reference       const_reference;
    typedef typename allocator_type::size_type             size_type;
    typedef typename allocator_type::difference_type       difference_type;

    typedef thrust::detail::normal_iterator<pointer>       iterator;
    typedef thrust::detail::normal_iterator<const_pointer> const_iterator;

    contiguous_storage(void);

    explicit contiguous_storage(size_type n);

    ~contiguous_storage(void);

    size_type size(void) const;

    size_type max_size(void) const;

    iterator begin(void);
    
    const_iterator begin(void) const;

    iterator end(void);

    const_iterator end(void) const;

    reference operator[](size_type n);

    const_reference operator[](size_type n) const;

    allocator_type get_allocator(void) const;

    // note that allocate does *not* automatically call deallocate
    void allocate(size_type n);

    void deallocate(void);

    void swap(contiguous_storage &x);

  private:
    // XXX we could inherit from this to take advantage of empty base class optimization
    allocator_type m_allocator;

    iterator m_begin;
    
    size_type m_size;

    // disallow assignment
    contiguous_storage &operator=(const contiguous_storage &x);
}; // end contiguous_storage

} // end detail

template<typename T, typename Alloc> void swap(detail::contiguous_storage<T,Alloc> &lhs, detail::contiguous_storage<T,Alloc> &rhs);

} // end thrust

#include <thrust/detail/contiguous_storage.inl>

