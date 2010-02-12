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

/*! \file raw_buffer.h
 *  \brief Container-like class for wrapped malloc/free.
 */

#pragma once

#include <thrust/device_malloc_allocator.h>
#include <thrust/iterator/detail/normal_iterator.h>
#include <thrust/iterator/iterator_traits.h>
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
  struct choose_raw_buffer_allocator
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

            identity_< device_malloc_allocator<T> >,

            void
          >
        >
      >
{};


template<typename T, typename Space>
  class raw_buffer
{
  public:
    typedef typename choose_raw_buffer_allocator<T,Space>::type allocator_type;
    typedef T                                                   value_type;
    typedef typename allocator_type::pointer                    pointer;
    typedef typename allocator_type::const_pointer              const_pointer;
    typedef typename allocator_type::reference                  reference;
    typedef typename allocator_type::const_reference            const_reference;
    typedef typename std::size_t                                size_type; 
    typedef typename allocator_type::difference_type            difference_type;

    typedef normal_iterator<pointer>                            iterator;
    typedef normal_iterator<const_pointer>                      const_iterator;

    explicit raw_buffer(size_type n);

    template<typename InputIterator>
    raw_buffer(InputIterator first, InputIterator last);

    ~raw_buffer(void);

    size_type size(void) const;

    iterator begin(void);

    const_iterator begin(void) const;

    const_iterator cbegin(void) const;

    iterator end(void);

    const_iterator end(void) const;

    const_iterator cend(void) const;

    reference operator[](size_type n);

    const_reference operator[](size_type n) const;


  protected:
    allocator_type m_allocator;

    iterator m_begin, m_end;

  private:
    // disallow assignment
    raw_buffer &operator=(const raw_buffer &);
}; // end raw_buffer


template<typename T>
  class raw_omp_device_buffer
    : public raw_buffer<T, thrust::detail::omp_device_space_tag >
{
  private:
    typedef raw_buffer<T, thrust::detail::omp_device_space_tag > super_t;

  public:
    explicit raw_omp_device_buffer(typename super_t::size_type n):super_t(n){}

    template<typename InputIterator>
    raw_omp_device_buffer(InputIterator first, InputIterator last):super_t(first,last){}
}; // end raw_omp_device_buffer

template<typename T>
  class raw_cuda_device_buffer
    : public raw_buffer<T, thrust::detail::cuda_device_space_tag >
{
  private:
    typedef raw_buffer<T, thrust::detail::cuda_device_space_tag > super_t;

  public:
    explicit raw_cuda_device_buffer(typename super_t::size_type n):super_t(n){}

    template<typename InputIterator>
    raw_cuda_device_buffer(InputIterator first, InputIterator last):super_t(first,last){}
}; // end raw_cuda_device_buffer

template<typename T>
  class raw_host_buffer
    : public raw_buffer<T, thrust::host_space_tag >
{
  private:
    typedef raw_buffer<T, thrust::host_space_tag > super_t;

  public:
    explicit raw_host_buffer(typename super_t::size_type n):super_t(n){}

    template<typename InputIterator>
    raw_host_buffer(InputIterator first, InputIterator last):super_t(first,last){}
}; // end raw_host_buffer

} // end detail

} // end thrust

#include <thrust/detail/raw_buffer.inl>

