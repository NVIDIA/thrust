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

/*! \file temporary_array.h
 *  \brief Container-like class temporary storage inside algorithms.
 */

#pragma once

#include <thrust/detail/config.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/iterator/retag.h>
#include <thrust/detail/contiguous_storage.h>
#include <thrust/detail/allocator/temporary_allocator.h>
#include <thrust/detail/allocator/no_throw_allocator.h>
#include <thrust/system/cpp/detail/tag.h>
#include <memory>

namespace thrust
{

namespace detail
{

// XXX eliminate this
template<typename T, typename System>
  struct choose_temporary_array_allocator
    : eval_if<
        // catch any_system_tag and output an error
        is_convertible<System, thrust::any_system_tag>::value,
        
        void,

        eval_if<
          // XXX this case shouldn't exist
          is_same<System, thrust::cpp::tag>::value,

          identity_< std::allocator<T> >,

          // XXX add backend-specific allocators here?
          identity_< no_throw_allocator<temporary_allocator<T,System> > >
        >
      >
{};


template<typename T, typename System>
  class temporary_array
    : public contiguous_storage<
               T,
               typename choose_temporary_array_allocator<T,System>::type
             >
{
  private:
    typedef contiguous_storage<
      T,
      typename choose_temporary_array_allocator<T,System>::type
    > super_t;

  public:
    typedef typename super_t::size_type size_type;

    explicit temporary_array(size_type n);

    template<typename InputIterator>
    temporary_array(InputIterator first, InputIterator last);
}; // end temporary_array


// XXX eliminate this when we do ranges for real
template<typename Iterator, typename Tag>
  class tagged_iterator_range
{
  public:
    typedef thrust::detail::tagged_iterator<Iterator,Tag> iterator;

    tagged_iterator_range(Iterator first, Iterator last)
      : m_begin(reinterpret_tag<Tag>(first)),
        m_end(reinterpret_tag<Tag>(last))
    {}

    iterator begin(void) const { return m_begin; }
    iterator end(void) const { return m_end; }

  private:
    iterator m_begin, m_end;
};


// if the system of Iterator is convertible to Tag, then just make a shallow
// copy of the range.  else, use a temporary_array
// note that the resulting iterator is explicitly tagged with Tag either way
template<typename Iterator, typename Tag>
  struct move_to_system_base
    : public eval_if<
        is_convertible<
          typename thrust::iterator_system<Iterator>::type,
          Tag
        >::value,
        identity_<
          tagged_iterator_range<Iterator,Tag>
        >,
        identity_<
          temporary_array<
            typename thrust::iterator_value<Iterator>::type,
            Tag
          >
        >
      >
{};


template<typename Iterator, typename Tag>
  class move_to_system
    : public move_to_system_base<
        Iterator,
        Tag
      >::type
{
  typedef typename move_to_system_base<Iterator,Tag>::type super_t;

  public:
    move_to_system(Iterator first, Iterator last)
      : super_t(first, last) {}
};

} // end detail

} // end thrust

#include <thrust/detail/temporary_array.inl>

