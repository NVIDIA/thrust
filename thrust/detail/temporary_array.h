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
#include <memory>

namespace thrust
{

namespace detail
{


template<typename T, typename System>
  class temporary_array
    : public contiguous_storage<
               T,
               no_throw_allocator<
                 temporary_allocator<T,System>
               >
             >
{
  private:
    typedef contiguous_storage<
      T,
      no_throw_allocator<
        temporary_allocator<T,System>
      >
    > super_t;

    // to help out the constructor
    typedef no_throw_allocator<temporary_allocator<T,System> > alloc_type;

  public:
    typedef typename super_t::size_type size_type;

    temporary_array(thrust::dispatchable<System> &system, size_type n);

    template<typename InputIterator>
    temporary_array(thrust::dispatchable<System> &system,
                    InputIterator first,
                    InputIterator last);

    template<typename InputSystem, typename InputIterator>
    temporary_array(thrust::dispatchable<System> &system,
                    thrust::dispatchable<InputSystem> &input_system,
                    InputIterator first,
                    InputIterator last);
}; // end temporary_array


// XXX eliminate this when we do ranges for real
template<typename Iterator, typename System>
  class tagged_iterator_range
{
  public:
    typedef thrust::detail::tagged_iterator<Iterator,System> iterator;

    template<typename Ignored1, typename Ignored2>
    tagged_iterator_range(const Ignored1 &, const Ignored2 &, Iterator first, Iterator last)
      : m_begin(reinterpret_tag<System>(first)),
        m_end(reinterpret_tag<System>(last))
    {}

    iterator begin(void) const { return m_begin; }
    iterator end(void) const { return m_end; }

  private:
    iterator m_begin, m_end;
};


// if the system of Iterator is convertible to System, then just make a shallow
// copy of the range. else, use a temporary_array
// note that the resulting iterator is explicitly tagged with System either way
template<typename Iterator, typename System>
  struct move_to_system_base
    : public eval_if<
        is_convertible<
          typename thrust::iterator_system<Iterator>::type,
          System
        >::value,
        identity_<
          tagged_iterator_range<Iterator,System>
        >,
        identity_<
          temporary_array<
            typename thrust::iterator_value<Iterator>::type,
            System
          >
        >
      >
{};


template<typename Iterator, typename System>
  class move_to_system
    : public move_to_system_base<
        Iterator,
        System
      >::type
{
  typedef typename move_to_system_base<Iterator,System>::type super_t;

  typename thrust::iterator_system<Iterator>::type input_system;

  public:
    move_to_system(thrust::dispatchable<System> &system,
                   Iterator first,
                   Iterator last)
      : super_t(system, input_system, first, last) {}
};

} // end detail

} // end thrust

#include <thrust/detail/temporary_array.inl>

