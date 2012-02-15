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

/*! \file uninitialized_array.h
 *  \brief Container-like class for wrapped malloc/free.
 */

#pragma once

#include <thrust/detail/backend/internal_allocator.h>
#include <thrust/iterator/detail/normal_iterator.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/detail/contiguous_storage.h>
#include <memory>

namespace thrust
{

// forward declaration of device_malloc_allocator
template<typename T> class device_malloc_allocator;

namespace detail
{

// forward declaration of normal_iterator
template<typename> class normal_iterator;

template<typename T, typename Space>
  struct choose_uninitialized_array_allocator
    : eval_if<
        // catch any_space_tag and output an error
        is_convertible<Space, thrust::any_space_tag>::value,
        
        void,

        eval_if<
          // XXX this check is technically incorrect: any could convert to host
          is_convertible<Space, thrust::host_space_tag>::value,

          identity_< std::allocator<T> >,

          // XXX add backend-specific allocators here?

          eval_if<
            // XXX this check is technically incorrect: any could convert to device
            is_convertible<Space, thrust::device_space_tag>::value,

            identity_< thrust::detail::backend::internal_allocator<T> >,

            void
          >
        >
      >
{};


template<typename T, typename Space>
  class uninitialized_array
    : public contiguous_storage<
               T,
               typename choose_uninitialized_array_allocator<T,Space>::type
             >
{
  private:
    typedef contiguous_storage<
      T,
      typename choose_uninitialized_array_allocator<T,Space>::type
    > super_t;

  public:
    typedef typename super_t::size_type size_type;

    explicit uninitialized_array(size_type n);

    template<typename InputIterator>
    uninitialized_array(InputIterator first, InputIterator last);
}; // end uninitialized_array


// XXX eliminate this when we do ranges for real
template<typename Iterator>
  class iterator_range
{
  public:
    iterator_range(Iterator first, Iterator last)
      : m_begin(first), m_end(last) {}

    Iterator begin(void) const { return m_begin; }
    Iterator end(void) const { return m_end; }

  private:
    Iterator m_begin, m_end;
};


// if the space of Iterator1 is convertible to Iterator2, then just make a shallow
// copy of the range.  else, use a uninitialized_array
template<typename Iterator1, typename Iterator2>
  struct move_to_space_base
    : public eval_if<
        is_convertible<
          typename thrust::iterator_space<Iterator1>::type,
          typename thrust::iterator_space<Iterator2>::type
        >::value,
        identity_<
          iterator_range<Iterator1>
        >,
        identity_<
          uninitialized_array<
            typename thrust::iterator_value<Iterator1>::type,
            typename thrust::iterator_space<Iterator2>::type
          >
        >
      >
{};


template<typename Iterator1, typename Iterator2>
  class move_to_space
    : public move_to_space_base<
        Iterator1,
        Iterator2
      >::type
{
  typedef typename move_to_space_base<Iterator1,Iterator2>::type super_t;

  public:
    move_to_space(Iterator1 first, Iterator1 last)
      : super_t(first, last) {}
};

} // end detail

} // end thrust

#include <thrust/detail/uninitialized_array.inl>

