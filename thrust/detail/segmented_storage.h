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

#include <thrust/iterator/detail/segmentation/segmented_iterator.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/detail/contiguous_storage.h>
#include <thrust/host_vector.h>

namespace thrust
{

namespace detail
{

// XXX consider making the second parameter Storage instead of Allocator
//     this would allow hierarchical storage
template<typename T, typename Allocator>
  class segmented_storage
{
//  private:
  public:
    typedef contiguous_storage<T,Allocator>                          storage_type;
    typedef thrust::host_vector<storage_type>                        range_of_segments_type;

  public:
    typedef Allocator                                                allocator_type;
    typedef T                                                        value_type;
    typedef thrust::detail::segmented_iterator<
      typename range_of_segments_type::iterator
    >                                                                pointer;
    typedef thrust::detail::segmented_iterator<
      typename range_of_segments_type::const_iterator
    >                                                                const_pointer;
    typedef typename thrust::iterator_reference<pointer>::type       reference;
    typedef typename thrust::iterator_reference<const_pointer>::type const_reference;

    // XXX should these be something else?
    typedef typename allocator_type::size_type                       size_type;
    typedef typename allocator_type::difference_type                 difference_type;

    typedef pointer                                                  iterator;
    typedef const_pointer                                            const_iterator;

    segmented_storage(void);

    explicit segmented_storage(size_type n);

    ~segmented_storage(void);

    size_type size(void) const;

    size_type max_size(void) const;

    iterator begin(void);

    const_iterator begin(void) const;

    iterator end(void);

    const_iterator end(void) const;

    allocator_type get_allocator(void) const;

    // note that allocate does *not* automatically call deallocate
    void allocate(size_type n);

    void deallocate(void);

    void swap(segmented_storage &x);

  private:
    // a range of segments
    range_of_segments_type m_storage;

    // XXX we need to abstract this sort of thing into a segmentation policy
    static size_type choose_number_of_segments(void);
}; // end segmented_storage

} // end detail

} // end thrust

#include <thrust/detail/segmented_storage.inl>

