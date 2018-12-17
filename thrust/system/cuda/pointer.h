/*
 *  Copyright 2008-2018 NVIDIA Corporation
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

#pragma once

#include <thrust/detail/config.h>
#include <thrust/system/cuda/detail/execution_policy.h>
#include <thrust/detail/type_traits.h>
#include <thrust/detail/pointer.h>

namespace thrust
{
namespace cuda_cub
{

template <typename>
class pointer;

} // end cuda_cub
} // end thrust


// specialize thrust::iterator_traits to avoid problems with the name of
// pointer's constructor shadowing its nested pointer type
// do this before pointer is defined so the specialization is correctly
// used inside the definition
namespace thrust
{

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

  #if THRUST_CPP_DIALECT >= 2011
  // NOTE: This is needed so that Thrust smart pointers can be used in
  // `std::unique_ptr`.
  __host__ __device__
  pointer(decltype(nullptr)) : super_t(nullptr) {}
  #endif

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
  explicit
  pointer(const OtherPointer &other,
          typename thrust::detail::enable_if_void_pointer_is_system_convertible<
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

  #if THRUST_CPP_DIALECT >= 2011
  // NOTE: This is needed so that Thrust smart pointers can be used in
  // `std::unique_ptr`.
  __host__ __device__
  pointer& operator=(decltype(nullptr))
  {
    super_t::operator=(nullptr);
    return *this;
  }
  #endif
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

} // end cuda_cub

namespace system {
namespace cuda {
using thrust::cuda_cub::pointer;
using thrust::cuda_cub::reference;
} // end cuda
} // end system

namespace cuda {
using thrust::cuda_cub::pointer;
using thrust::cuda_cub::reference;
} // end cuda

} // end thrust

#include <thrust/system/cuda/detail/pointer.inl>
