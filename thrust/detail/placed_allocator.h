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

#include <thrust/detail/config.h>
#include <thrust/iterator/detail/placement/place.h>
#include <thrust/iterator/detail/placement/placed_iterator.h>
#include <thrust/iterator/iterator_traits.h>

namespace thrust
{

namespace detail
{

// XXX allocator_adaptor might be nice
template<typename T, typename Allocator>
  class placed_allocator
{
  public:
    typedef typename Allocator::template rebind<T>::other              base_allocator_type;
    typedef typename base_allocator_type::value_type                   value_type;
    typedef thrust::detail::placed_iterator<
      typename base_allocator_type::pointer
    >                                                                  pointer;
    typedef thrust::detail::placed_iterator<
      typename base_allocator_type::const_pointer
    >                                                                  const_pointer;
    typedef typename pointer::reference                                reference;
    typedef typename const_pointer::reference                          const_reference;
    typedef typename base_allocator_type::size_type                    size_type;
    typedef typename thrust::iterator_traits<pointer>::difference_type difference_type;

    template<typename U>
      struct rebind
    {
      typedef placed_allocator<U,Allocator> other;
    }; // end rebind

    inline placed_allocator(place p = place());

    inline ~placed_allocator(void);

    // place
    inline place get_place(void) const;

    inline void set_place(place p);

    // address
    __host__ __device__
    inline pointer address(reference r);

    __host__ __device__
    inline const_pointer address(const_reference r);

    // memory allocation
    inline pointer allocate(size_type cnt,
                            const_pointer = const_pointer());

    inline void deallocate(pointer p, size_type cnt);

    inline size_type max_size(void) const;

    inline bool operator==(placed_allocator const &x);

    inline bool operator!=(placed_allocator const &x);

  public:
    base_allocator_type m_allocator;
    place m_place;
}; // end placed_allocator

} // end detail

} // end thrust

#include <thrust/detail/placed_allocator.inl>

