/*
 *  Copyright 2008-2013 NVIDIA Corporation
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


/*! \file device_allocator.h
 *  \brief An allocator which creates new elements in device memory
 */

#pragma once

#include <thrust/detail/config.h>
#include <thrust/device_new_allocator.h>
#include <limits>
#include <stdexcept>

namespace thrust
{

/*! \addtogroup memory_management_classes Memory Management Classes
 *  \{
 */

template<typename T> class device_allocator;

/*! \p device_allocator<void> is a device memory allocator.
 *  This class is a specialization for \c void.
 *
 *  \see device_ptr
 *  \see http://www.sgi.com/tech/stl/Allocators.html
 */
template<>
  class device_allocator<void>
{
  public:
    /*! Type of element allocated, \c void. */
    typedef void                              value_type;

    /*! Pointer to allocation, \c device_ptr<void>. */
    typedef device_ptr<void>                  pointer;

    /*! \c const pointer to allocation, \c device_ptr<const void>. */
    typedef device_ptr<const void>            const_pointer;

    /*! Type of allocation size, \c std::size_t. */
    typedef std::size_t                       size_type;

    /*! Type of allocation difference, \c pointer::difference_type. */
    typedef pointer::difference_type difference_type;

    /*! The \p rebind metafunction provides the type of a \p device_allocator
     *  instantiated with another type.
     *
     *  \tparam U The other type to use for instantiation.
     */
    template<typename U>
      struct rebind
    {
      /*! The typedef \p other gives the type of the rebound \p device_allocator.
       */
      typedef device_allocator<U> other;
    }; // end rebind
}; // end device_allocator<void>

/*! \p device_allocator is a device memory allocator.
 *  This implementation inherits from \p device_new_allocator.
 *
 *  \see device_ptr
 *  \see device_new_allocator
 *  \see http://www.sgi.com/tech/stl/Allocators.html
 */
template<typename T>
  class device_allocator
    : public device_new_allocator<T>
{
  public:
    /*! The \p rebind metafunction provides the type of a \p device_allocator
     *  instantiated with another type.
     *
     *  \tparam U The other type to use for instantiation.
     */
    template<typename U>
      struct rebind
    {
      /*! The typedef \p other gives the type of the rebound \p device_allocator.
       */
      typedef device_allocator<U> other;
    }; // end rebind

    /*! No-argument constructor has no effect.
     */
    __host__ __device__
    inline device_allocator() {}

    /*! Copy constructor has no effect.
     */
    __host__ __device__
    inline device_allocator(device_allocator const&) {}

    /*! Constructor from other \p allocator has no effect.
     */
    template<typename U>
    __host__ __device__
    inline device_allocator(device_allocator<U> const&) {}
}; // end device_allocator

/*! \}
 */

} // end thrust

