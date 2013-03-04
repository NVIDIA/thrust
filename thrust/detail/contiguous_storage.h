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

#include <thrust/iterator/detail/normal_iterator.h>
#include <thrust/detail/execution_policy.h>
#include <thrust/detail/allocator/allocator_traits.h>

namespace thrust
{

namespace detail
{

// XXX parameter T is redundant with parameter Alloc
template<typename T, typename Alloc>
  class contiguous_storage
{
  private:
    typedef thrust::detail::allocator_traits<Alloc> alloc_traits;

  public:
    typedef Alloc                                      allocator_type;
    typedef T                                          value_type;
    typedef typename alloc_traits::pointer             pointer;
    typedef typename alloc_traits::const_pointer       const_pointer;
    typedef typename alloc_traits::size_type           size_type;
    typedef typename alloc_traits::difference_type     difference_type;

    // XXX we should bring reference & const_reference into allocator_traits
    //     at the moment, it's unclear how -- we have nothing analogous to
    //     rebind_pointer for references
    //     we either need to add reference_traits or extend the existing
    //     pointer_traits to support wrapped references
    typedef typename Alloc::reference                  reference;
    typedef typename Alloc::const_reference            const_reference;

    typedef thrust::detail::normal_iterator<pointer>       iterator;
    typedef thrust::detail::normal_iterator<const_pointer> const_iterator;

    explicit contiguous_storage(const allocator_type &alloc = allocator_type());

    explicit contiguous_storage(size_type n, const allocator_type &alloc = allocator_type());

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

    void default_construct_n(iterator first, size_type n);

    void uninitialized_fill_n(iterator first, size_type n, const value_type &value);

    template<typename InputIterator>
    iterator uninitialized_copy(InputIterator first, InputIterator last, iterator result);

    template<typename System, typename InputIterator>
    iterator uninitialized_copy(thrust::execution_policy<System> &from_system,
                                InputIterator first,
                                InputIterator last,
                                iterator result);

    template<typename InputIterator, typename Size>
    iterator uninitialized_copy_n(InputIterator first, Size n, iterator result);

    template<typename System, typename InputIterator, typename Size>
    iterator uninitialized_copy_n(thrust::execution_policy<System> &from_system,
                                  InputIterator first,
                                  Size n,
                                  iterator result);

    void destroy(iterator first, iterator last);

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

