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

#include <thrust/detail/config.h>
#include <thrust/device_ptr.h>
#include <thrust/device_reference.h>
#include <thrust/device_malloc.h>
#include <thrust/detail/device/no_throw_free.h>
#include <limits>
#include <stdexcept>

namespace thrust
{


// XXX WAR circular #inclusion with forward declaration
template<typename T> thrust::device_ptr<T> device_malloc(const std::size_t n);


namespace detail
{

namespace device
{

template<typename T>
  class internal_allocator
{
  public:
    typedef T                                 value_type;
    typedef device_ptr<T>                     pointer;
    typedef device_ptr<const T>               const_pointer;
    typedef device_reference<T>               reference;
    typedef device_reference<const T>         const_reference;
    typedef std::size_t                       size_type;
    typedef typename pointer::difference_type difference_type;

    // convert a internal_allocator<T> to device_malloc_allocator<U>
    template<typename U>
      struct rebind
    {
      typedef internal_allocator<U> other;
    }; // end rebind

    inline internal_allocator() {}

    inline ~internal_allocator() {}

    inline internal_allocator(internal_allocator const&) {}

    template<typename U>
    inline internal_allocator(internal_allocator<U> const&) {}

    // address
    inline pointer address(reference r) { return &r; }
    
    inline const_pointer address(const_reference r) { return &r; }

    // memory allocation
    inline pointer allocate(size_type cnt,
                            const_pointer = const_pointer(static_cast<T*>(0)))
    {
      if(cnt > this->max_size())
      {
        throw std::bad_alloc();
      } // end if

      return pointer(device_malloc<T>(cnt));
    } // end allocate()

    inline void deallocate(pointer p, size_type cnt) throw()
    {
      thrust::detail::device::no_throw_free(p);
    } // end deallocate()

    inline size_type max_size() const
    {
      return (std::numeric_limits<size_type>::max)() / sizeof(T);
    } // end max_size()

    inline bool operator==(internal_allocator const&) { return true; }

    inline bool operator!=(internal_allocator const &a) {return !operator==(a); }
}; // end internal_allocator

} // end namespace device

} // end namespace detail

} // end thrust

