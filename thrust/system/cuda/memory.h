/*
 *  Copyright 2008-2011 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in ccudaliance with the License.
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

/*! \file cuda/memory.h
 *  \brief Classes for managing CUDA-typed memory.
 */

#pragma once

#include <thrust/detail/config.h>
#include <thrust/system/cuda/detail/tag.h>
#include <thrust/detail/pointer_base.h>
#include <thrust/detail/reference_base.h>
#include <thrust/detail/type_traits.h>
#include <ostream>

namespace thrust
{
namespace system
{
namespace cuda
{

// forward declaration of reference for pointer
template<typename Element> class reference;

template<typename T>
  class pointer
    : public thrust::detail::pointer_base<
               T,
               thrust::system::cuda::pointer<T>,
               thrust::system::cuda::reference<T>,
               thrust::system::cuda::tag
             >
{
  private:
    typedef thrust::detail::pointer_base<
      T,
      thrust::system::cuda::pointer<T>,
      thrust::system::cuda::reference<T>,
      thrust::system::cuda::tag
    > super_t;

  public:
    // XXX doxygenate these

    __host__ __device__
    pointer() : super_t() {}

    template<typename OtherT>
    __host__ __device__
    explicit pointer(OtherT *ptr) : super_t(ptr) {}

    template<typename OtherPointer>
    __host__ __device__
    pointer(const OtherPointer &other,
            typename thrust::detail::enable_if_pointer_is_convertible<
              OtherPointer,
              pointer
            >::type * = 0) : super_t(other) {}

    template<typename OtherPointer>
    __host__ __device__
    typename thrust::detail::enable_if_pointer_is_convertible<
      OtherPointer,
      pointer,
      pointer &
    >::type
    operator=(const OtherPointer &other)
    {
      return super_t::operator=(other);
    }
}; // end pointer


template<typename T>
  class reference
    : public thrust::detail::reference_base<
               thrust::system::cuda::reference<T>,
               T,
               thrust::system::cuda::pointer<T>
             >
{
  private:
    typedef thrust::detail::reference_base<
      thrust::system::cuda::reference<T>,
      T,
      thrust::system::cuda::pointer<T>
    > super_t;

  public:
    typedef typename super_t::value_type value_type;
    typedef typename super_t::pointer    pointer;

    template<typename OtherT>
    __host__ __device__
    reference(const reference<OtherT> &other,
              typename thrust::detail::enable_if_convertible<
                typename reference<OtherT>::pointer,
                pointer
              >::type * = 0)
      : super_t(other)
    {}

    __host__ __device__
    explicit reference(const pointer &ptr)
      : super_t(ptr)
    {}

    template<typename OtherT>
    reference &operator=(const reference<OtherT> &other);

    reference &operator=(const value_type &x);
}; // end reference

template<typename T>
__host__ __device__
void swap(reference<T> &x, reference<T> &y);

inline pointer<void> malloc(std::size_t n);

inline void free(pointer<void> ptr);

// XXX upon c++11
// template<typename T> using allocator = thrust::detail::tagged_allocator<T,tag,pointer<T> >;

template<typename T>
  struct allocator
    : thrust::detail::tagged_allocator<
        T,
        tag,
        pointer<T>
      >
{
  template<typename U>
    struct rebind
  {
    typedef allocator<U> other;
  };

  __host__ __device__
  inline allocator() {}

  __host__ __device__
  inline allocator(const allocator &) {}

  template<typename U>
  __host__ __device__
  inline allocator(const allocator<U> &) {}

  __host__ __device__
  inline ~allocator() {}
}; // end allocator

} // end cuda
} // end system

// alias cuda's members at top-level
namespace cuda
{

using thrust::system::cuda::pointer;
using thrust::system::cuda::reference;
using thrust::system::cuda::malloc;
using thrust::system::cuda::free;
using thrust::system::cuda::allocator;

} // end cuda

} // end thrust

#include <thrust/system/cuda/detail/memory.inl>

