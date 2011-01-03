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


/*! \file device_new_allocator.h
 *  \brief Defines the interface to a
 *         standard C++ allocator class for
 *         allocating device memory with device_new.
 */

#pragma once

#include <thrust/detail/config.h>
#include <thrust/device_ptr.h>
#include <thrust/device_reference.h>
#include <thrust/device_new.h>
#include <thrust/device_delete.h>
#include <limits>
#include <stdexcept>

namespace thrust
{

/*! \addtogroup memory_management Memory Management
 *  \addtogroup memory_management_classes Memory Management Classes
 *  \ingroup memory_management
 *  \{
 */

/*! \p device_new_allocator is a device memory allocator that employs the
 *  \p device_new function for allocation.
 *
 *  \see device_new
 *  \see device_ptr
 *  \see http://www.sgi.com/tech/stl/Allocators.html
 */
template<typename T>
  class device_new_allocator
{
  public:
    typedef T                                 value_type;
    typedef device_ptr<T>                     pointer;
    typedef device_ptr<const T>               const_pointer;
    typedef device_reference<T>               reference;
    typedef device_reference<const T>         const_reference;
    typedef std::size_t                       size_type;
    typedef typename pointer::difference_type difference_type;

    // convert a device_new_allocator<T> to device_new_allocator<U>
    template<typename U>
      struct rebind
    {
      typedef device_new_allocator<U> other;
    }; // end rebind

    __host__ __device__
    inline device_new_allocator() {}

    __host__ __device__
    inline ~device_new_allocator() {}

    __host__ __device__
    inline device_new_allocator(device_new_allocator const&) {}

    template<typename U>
    __host__ __device__
    inline device_new_allocator(device_new_allocator<U> const&) {}

    // address
    __host__ __device__
    inline pointer address(reference r) { return &r; }
    
    __host__ __device__
    inline const_pointer address(const_reference r) { return &r; }

    // memory allocation
    __host__
    inline pointer allocate(size_type cnt,
                            const_pointer = const_pointer(static_cast<T*>(0)))
    {
      if(cnt > this->max_size())
      {
        throw std::bad_alloc();
      } // end if

      // use "::operator new" rather than keyword new
      return pointer(device_new<T>(cnt));
    } // end allocate()

    __host__
    inline void deallocate(pointer p, size_type cnt)
    {
      // use "::operator delete" rather than keyword delete
      device_delete(p);
    } // end deallocate()

    __host__ __device__
    inline size_type max_size() const
    {
      return std::numeric_limits<size_type>::max THRUST_PREVENT_MACRO_SUBSTITUTION () / sizeof(T);
    } // end max_size()

    __host__ __device__
    inline bool operator==(device_new_allocator const&) { return true; }

    __host__ __device__
    inline bool operator!=(device_new_allocator const &a) {return !operator==(a); }
}; // end device_new_allocator

/*! \}
 */

} // end thrust

