/*
 *  Copyright 2020 NVIDIA Corporation
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

#include <thrust/detail/pointer.h>

#include <thrust/detail/type_traits.h>
#include <thrust/system/cuda/detail/execution_policy.h>

namespace thrust
{
namespace system
{
namespace cuda
{
namespace detail
{

// forward decl for iterator traits:
template <typename T>
class managed_memory_pointer;

} // end namespace detail
} // end namespace cuda
} // end namespace system

// Specialize iterator traits to define `pointer` to something meaningful.
template <typename Element, typename Tag, typename Reference>
struct iterator_traits<thrust::pointer<
  Element,
  Tag,
  Reference,
  thrust::system::cuda::detail::managed_memory_pointer<Element> > > {
private:
  typedef thrust::pointer<
    Element,
    Tag,
    Reference,
    thrust::system::cuda::detail::managed_memory_pointer<Element> >
    ptr;

public:
  typedef typename ptr::iterator_category iterator_category;
  typedef typename ptr::value_type value_type;
  typedef typename ptr::difference_type difference_type;
  typedef Element* pointer;
  typedef typename ptr::reference reference;
}; // end iterator_traits

namespace system
{
namespace cuda
{
namespace detail
{

/*! A version of thrust::cuda_cub::pointer that uses c++ references instead
 * of thrust::cuda::reference. This is to allow managed memory pointers to
 * be used with host-side code in standard libraries that are not compatible
 * with proxy references.
 */
template <typename T>
class managed_memory_pointer
    : public thrust::pointer<
        T,
        thrust::cuda_cub::tag,
        typename thrust::detail::add_reference<T>::type,
        thrust::system::cuda::detail::managed_memory_pointer<T> >
{
private:
  typedef thrust::pointer<
    T,
    thrust::cuda_cub::tag,
    typename thrust::detail::add_reference<T>::type,
    thrust::system::cuda::detail::managed_memory_pointer<T> >
    super_t;

public:
  typedef typename super_t::raw_pointer pointer;

  /*! \p managed_memory_pointer's no-argument constructor initializes its
   * encapsulated pointer to \c 0.
   */
  __host__ __device__ managed_memory_pointer()
      : super_t()
  {}

#if THRUST_CPP_DIALECT >= 2011
  // NOTE: This is needed so that Thrust smart pointers can be used in
  // `std::unique_ptr`.
  __host__ __device__ managed_memory_pointer(decltype(nullptr))
      : super_t(nullptr)
  {}
#endif

  /*! This constructor allows construction of a <tt><const T></tt> from a
   * <tt>T*</tt>.
   *
   *  \param ptr A raw pointer to copy from, presumed to point to a location
   * in memory accessible by the \p cuda system. \tparam OtherT \p OtherT
   * shall be convertible to \p T.
   */
  template <typename OtherT>
  __host__ __device__ explicit managed_memory_pointer(OtherT* ptr)
      : super_t(ptr)
  {}

  /*! This constructor allows construction from another pointer-like object
   * with related type.
   *
   *  \param other The \p OtherPointer to copy.
   *  \tparam OtherPointer The system tag associated with \p OtherPointer
   * shall be convertible to \p thrust::system::cuda::tag and its element
   * type shall be convertible to \p T.
   */
  template <typename OtherPointer>
  __host__ __device__ managed_memory_pointer(
    const OtherPointer& other,
    typename thrust::detail::enable_if_pointer_is_convertible<
      OtherPointer,
      managed_memory_pointer>::type* = 0)
      : super_t(other)
  {}

  /*! This constructor allows construction from another pointer-like object
   * with \p void type.
   *
   *  \param other The \p OtherPointer to copy.
   *  \tparam OtherPointer The system tag associated with \p OtherPointer
   * shall be convertible to \p thrust::system::cuda::tag and its element
   * type shall be \p void.
   */
  template <typename OtherPointer>
  __host__ __device__ explicit managed_memory_pointer(
    const OtherPointer& other,
    typename thrust::detail::enable_if_void_pointer_is_system_convertible<
      OtherPointer,
      managed_memory_pointer>::type* = 0)
      : super_t(other)
  {}

  /*! Assignment operator allows assigning from another pointer-like object
   * with related type.
   *
   *  \param other The other pointer-like object to assign from.
   *  \tparam OtherPointer The system tag associated with \p OtherPointer
   * shall be convertible to \p thrust::system::cuda::tag and its element
   * type shall be convertible to \p T.
   */
  template <typename OtherPointer>
  __host__ __device__ typename thrust::detail::enable_if_pointer_is_convertible<
    OtherPointer,
    managed_memory_pointer,
    managed_memory_pointer&>::type
  operator=(const OtherPointer& other)
  {
    return super_t::operator=(other);
  }

#if THRUST_CPP_DIALECT >= 2011
  // NOTE: This is needed so that Thrust smart pointers can be used in
  // `std::unique_ptr`.
  __host__ __device__ managed_memory_pointer& operator=(decltype(nullptr))
  {
    super_t::operator=(nullptr);
    return *this;
  }
#endif

  __host__ __device__
  pointer operator->() const
  {
    return this->get();
  }

}; // class managed_memory_pointer

} // namespace detail
} // namespace cuda
} // namespace system
} // namespace thrust
