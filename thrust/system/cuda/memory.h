/*
 *  Copyright 2008-2013 NVIDIA Corporation
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

/*! \file thrust/system/cuda/memory.h
 *  \brief Managing memory associated with Thrust's CUDA system.
 */

#pragma once

#include <thrust/detail/config.h>
#include <thrust/system/cuda/execution_policy.h>
#include <thrust/memory.h>
#include <thrust/detail/type_traits.h>
#include <thrust/detail/allocator/malloc_allocator.h>
#include <ostream>

BEGIN_NS_THRUST
namespace cuda_cub {

template <typename>
class pointer;

}    // end cuda_
END_NS_THRUST


// specialize thrust::iterator_traits to avoid problems with the name of
// pointer's constructor shadowing its nested pointer type
// do this before pointer is defined so the specialization is correctly
// used inside the definition
BEGIN_NS_THRUST

template <typename Element>
struct iterator_traits<thrust::cuda_cub::pointer<Element> >
{
private:
  typedef thrust::cuda_cub::pointer<Element> ptr;

public:
  typedef typename ptr::iterator_category iterator_category;
  typedef typename ptr::value_type        value_type;
  typedef typename ptr::difference_type   difference_type;
  typedef ptr                             pointer;
  typedef typename ptr::reference         reference;
};    // end iterator_traits

namespace cuda_cub {

// forward declaration of reference for pointer
template <typename Element>
class reference;

// XXX nvcc + msvc have trouble instantiating reference below
//     this is a workaround
template <typename Element>
struct reference_msvc_workaround
{
  typedef thrust::cuda_cub::reference<Element> type;
};    // end reference_msvc_workaround


template <typename T>
class pointer
    : public thrust::pointer<
          T,
          thrust::cuda_cub::tag,
          thrust::cuda_cub::reference<T>,
          thrust::cuda_cub::pointer<T> >
{

private:
  typedef thrust::pointer<
      T,
      thrust::cuda_cub::tag,
      typename reference_msvc_workaround<T>::type,
      thrust::cuda_cub::pointer<T> >
      super_t;

public:
  __host__ __device__
  pointer() : super_t() {}

  template <typename OtherT>
  __host__ __device__ explicit pointer(OtherT *ptr) : super_t(ptr)
  {
  }

  template <typename OtherPointer>
  __host__ __device__
  pointer(const OtherPointer &other,
          typename thrust::detail::enable_if_pointer_is_convertible<
              OtherPointer,
              pointer>::type * = 0) : super_t(other)
  {
  }

  template <typename OtherPointer>
  __host__ __device__
      typename thrust::detail::enable_if_pointer_is_convertible<
          OtherPointer,
          pointer,
          pointer &>::type
      operator=(const OtherPointer &other)
  {
    return super_t::operator=(other);
  }
};    // struct pointer


template <typename T>
class reference
    : public thrust::reference<
          T,
          thrust::cuda_cub::pointer<T>,
          thrust::cuda_cub::reference<T> >
{

private:
  typedef thrust::reference<
      T,
      thrust::cuda_cub::pointer<T>,
      thrust::cuda_cub::reference<T> >
      super_t;

public:
  typedef typename super_t::value_type value_type;
  typedef typename super_t::pointer    pointer;

  __host__ __device__ explicit reference(const pointer &ptr)
      : super_t(ptr)
  {
  }

  template <typename OtherT>
  __host__ __device__
  reference(const reference<OtherT> &other,
            typename thrust::detail::enable_if_convertible<
                typename reference<OtherT>::pointer,
                pointer>::type * = 0)
      : super_t(other)
  {
  }
  template <typename OtherT>
  __host__ __device__
      reference &
      operator=(const reference<OtherT> &other);

  __host__ __device__
      reference &
      operator=(const value_type &x);
};    // struct reference

template <typename T>
__host__ __device__ void swap(reference<T> x, reference<T> y);

inline __host__ __device__
    pointer<void>
    malloc(std::size_t n);

template <typename T>
inline __host__ __device__
    pointer<T>
    malloc(std::size_t n);

inline __host__ __device__ void free(pointer<void> ptr);

// XXX upon c++11
// template<typename T> using allocator =
// thrust::detail::malloc_allocator<T,tag,pointer<T> >;
//
template <typename T>
struct allocator
    : thrust::detail::malloc_allocator<
          T,
          tag,
          pointer<T> >
{
  template <typename U>
  struct rebind
  {
    typedef allocator<U> other;
  };

  __host__ __device__ inline allocator() {}

  __host__ __device__ inline allocator(const allocator &) {}

  template <typename U>
  __host__ __device__ inline allocator(const allocator<U> &)
  {
  }

  __host__ __device__ inline ~allocator() {}
};    // struct allocator

}    // namespace cuda_cub

namespace system {
namespace cuda {
using thrust::cuda_cub::pointer;
using thrust::cuda_cub::reference;
using thrust::cuda_cub::swap;
using thrust::cuda_cub::malloc;
using thrust::cuda_cub::free;
using thrust::cuda_cub::allocator;
} // namespace cuda
} /// namespace system

namespace cuda {
using thrust::cuda_cub::pointer;
using thrust::cuda_cub::reference;
using thrust::cuda_cub::malloc;
using thrust::cuda_cub::free;
using thrust::cuda_cub::allocator;
}    // end cuda

END_NS_THRUST

#include <thrust/system/cuda/detail/memory.inl>

