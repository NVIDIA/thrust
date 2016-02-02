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

/*! \file thrust/system/cuda/experimental/managed_allocator.h
 *  \brief An allocator which creates new elements in "managed" memory with \p cudaManagedMalloc
 */

#pragma once

#include <thrust/detail/config.h>
#include <thrust/system/cuda/detail/guarded_cuda_runtime_api.h>
#include <stdexcept>
#include <limits>
#include <string>
#include <thrust/system/system_error.h>
#include <thrust/system/cuda/error.h>

namespace thrust
{

namespace system
{

namespace cuda
{

namespace experimental
{

/*! \addtogroup memory_management Memory Management
 *  \addtogroup memory_management_classes
 *  \ingroup memory_management
 *  \{
 */

/*! \p managed_allocator is a CUDA-specific host memory allocator
 *  that employs \c cudaMallocHost for allocation.
 *
 *  \see http://www.sgi.com/tech/stl/Allocators.html
 */
template<typename T> class managed_allocator;

template<>
  class managed_allocator<void>
{
  public:
    typedef void           value_type;
    typedef void       *   pointer;
    typedef const void *   const_pointer;
    typedef std::size_t    size_type;
    typedef std::ptrdiff_t difference_type;

    // convert a managed_allocator<void> to managed_allocator<U>
    template<typename U>
      struct rebind
    {
      typedef managed_allocator<U> other;
    }; // end rebind
}; // end managed_allocator


template<typename T>
  class managed_allocator
{
  public:
    typedef T              value_type;
    typedef T*             pointer;
    typedef const T*       const_pointer;
    typedef T&             reference;
    typedef const T&       const_reference;
    typedef std::size_t    size_type;
    typedef std::ptrdiff_t difference_type;

    // convert a managed_allocator<T> to managed_allocator<U>
    template<typename U>
      struct rebind
    {
      typedef managed_allocator<U> other;
    }; // end rebind

    /*! \p managed_allocator's null constructor does nothing.
     */
    __host__ __device__
    inline managed_allocator() {}

    /*! \p managed_allocator's null destructor does nothing.
     */
    __host__ __device__
    inline ~managed_allocator() {}

    /*! \p managed_allocator's copy constructor does nothing.
     */
    __host__ __device__
    inline managed_allocator(managed_allocator const &) {}

    /*! This version of \p managed_allocator's copy constructor
     *  is templated on the \c value_type of the \p managed_allocator
     *  to copy from.  It is provided merely for convenience; it
     *  does nothing.
     */
    template<typename U>
    __host__ __device__
    inline managed_allocator(managed_allocator<U> const &) {}

    /*! This method returns the address of a \c reference of
     *  interest.
     *
     *  \p r The \c reference of interest.
     *  \return \c r's address.
     */
    __host__ __device__
    inline pointer address(reference r) { return &r; }

    /*! This method returns the address of a \c const_reference
     *  of interest.
     *
     *  \p r The \c const_reference of interest.
     *  \return \c r's address.
     */
    __host__ __device__
    inline const_pointer address(const_reference r) { return &r; }

    /*! This method allocates storage for objects in managed device
     *  memory.
     *
     *  \p cnt The number of objects to allocate.
     *  \return a \c pointer to the newly allocated objects.
     *  \note This method does not invoke \p value_type's constructor.
     *        It is the responsibility of the caller to initialize the
     *        objects at the returned \c pointer. 
     */
    __host__
    inline pointer allocate(size_type cnt,
                            const_pointer = 0)
    {
      if(cnt > this->max_size())
      {
        throw std::bad_alloc();
      } // end if

      pointer result(0);
      cudaError_t error = cudaMallocManaged(reinterpret_cast<void**>(&result), cnt * sizeof(value_type), cudaMemAttachGlobal);

      if(error)
      {
        throw std::bad_alloc();
      } // end if

      return result;
    } // end allocate()

    /*! This method deallocates managed device memory previously allocated
     *  with this \c managed_allocator.
     *
     *  \p p A \c pointer to the previously allocated memory.
     *  \p cnt The number of objects previously allocated at
     *         \p p.
     *  \note This method does not invoke \p value_type's destructor.
     *        It is the responsibility of the caller to destroy
     *        the objects stored at \p p.
     */
    __host__
    inline void deallocate(pointer p, size_type cnt)
    {
      cudaError_t error = cudaFree(p);

      if(error)
      {
        throw thrust::system_error(error, thrust::cuda_category());
      } // end if
    } // end deallocate()

    /*! This method returns the maximum size of the \c cnt parameter
     *  accepted by the \p allocate() method.
     *
     *  \return The maximum number of objects that may be allocated
     *          by a single call to \p allocate().
     */
    inline size_type max_size() const
    {
      return (std::numeric_limits<size_type>::max)() / sizeof(T);
    } // end max_size()

    /*! This method tests this \p managed_allocator for equality to
     *  another.
     *
     *  \param x The other \p managed_allocator of interest.
     *  \return This method always returns \c true.
     */
    __host__ __device__
    inline bool operator==(managed_allocator const& x) { return true; }

    /*! This method tests this \p managed_allocator for inequality
     *  to another.
     *
     *  \param x The other \p managed_allocator of interest.
     *  \return This method always returns \c false.
     */
    __host__ __device__
    inline bool operator!=(managed_allocator const &x) { return !operator==(x); }
}; // end managed_allocator

/*! \}
 */

} // end experimental

} // end cuda

} // end system

// alias cuda's members at top-level
namespace cuda
{

namespace experimental
{

using thrust::system::cuda::experimental::managed_allocator;

} // end experimental

} // end cuda

} // end thrust

