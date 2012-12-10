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

/*! \file thrust/system/omp/memory.h
 *  \brief Managing memory associated with Thrust's OpenMP system.
 */

#pragma once

#include <thrust/detail/config.h>
#include <thrust/system/omp/detail/tag.h>
#include <thrust/memory.h>
#include <thrust/detail/type_traits.h>
#include <thrust/detail/allocator/malloc_allocator.h>
#include <ostream>

namespace thrust
{
namespace system
{
namespace omp
{

template<typename> class pointer;

} // end omp
} // end system
} // end thrust


/*! \cond
 */

// specialize std::iterator_traits to avoid problems with the name of
// pointer's constructor shadowing its nested pointer type
// do this before pointer is defined so the specialization is correctly
// used inside the definition
namespace std
{

template<typename Element>
  struct iterator_traits<thrust::system::omp::pointer<Element> >
{
  private:
    typedef thrust::system::omp::pointer<Element> ptr;

  public:
    typedef typename ptr::iterator_category       iterator_category;
    typedef typename ptr::value_type              value_type;
    typedef typename ptr::difference_type         difference_type;
    typedef ptr                                   pointer;
    typedef typename ptr::reference               reference;
}; // end iterator_traits

} // end std

/*! \endcond
 */


namespace thrust
{
namespace system
{

/*! \addtogroup system_backends Systems
 *  \ingroup system
 *  \{
 */

/*! \namespace thrust::system::omp
 *  \brief \p thrust::system::omp is the namespace containing functionality for allocating, manipulating,
 *         and deallocating memory available to Thrust's OpenMP backend system.
 *         The identifiers are provided in a separate namespace underneath <tt>thrust::system</tt>
 *         for import convenience but are also aliased in the top-level <tt>thrust::omp</tt>
 *         namespace for easy access.
 *
 */
namespace omp
{

// forward declaration of reference for pointer
template<typename Element> class reference;

/*! \cond
 */

// XXX nvcc + msvc have trouble instantiating reference below
//     this is a workaround
namespace detail
{

template<typename Element>
  struct reference_msvc_workaround
{
  typedef thrust::system::omp::reference<Element> type;
}; // end reference_msvc_workaround

} // end detail

/*! \endcond
 */

#if 0
/*! \p omp::tag is type representing Thrust's OpenMP backend system in C++'s type system.
 *  Iterators "tagged" with a type which is convertible to \p omp::tag assert that they may be
 *  "dispatched" to algorithm implementations in the \p omp system.
 *
 *  \p omp::tag is convertible to \p cpp::tag.
 */
struct tag { unspecified };
#endif

/*! \p pointer stores a pointer to an object allocated in memory available to the omp system.
 *  This type provides type safety when dispatching standard algorithms on ranges resident
 *  in omp memory.
 *
 *  \p pointer has pointer semantics: it may be dereferenced and manipulated with pointer arithmetic.
 *
 *  \p pointer can be created with the function \p omp::malloc, or by explicitly calling its constructor
 *  with a raw pointer.
 *
 *  The raw pointer encapsulated by a \p pointer may be obtained by eiter its <tt>get</tt> member function
 *  or the \p raw_pointer_cast function.
 *
 *  \note \p pointer is not a "smart" pointer; it is the programmer's responsibility to deallocate memory
 *  pointed to by \p pointer.
 *
 *  \tparam T specifies the type of the pointee.
 *
 *  \see omp::malloc
 *  \see omp::free
 *  \see raw_pointer_cast
 */
template<typename T>
  class pointer
    : public thrust::pointer<
               T,
               thrust::system::omp::tag,
               thrust::system::omp::reference<T>,
               thrust::system::omp::pointer<T>
             >
{
  /*! \cond
   */

  private:
    typedef thrust::pointer<
      T,
      thrust::system::omp::tag,
      //thrust::system::omp::reference<T>,
      typename detail::reference_msvc_workaround<T>::type,
      thrust::system::omp::pointer<T>
    > super_t;

  /*! \endcond
   */

  public:
    // note that omp::pointer's member functions need __host__ __device__
    // to interoperate with nvcc + iterators' dereference member function

    /*! \p pointer's no-argument constructor initializes its encapsulated pointer to \c 0.
     */
    __host__ __device__
    pointer() : super_t() {}

    /*! This constructor allows construction of a <tt>pointer<const T></tt> from a <tt>T*</tt>.
     *
     *  \param ptr A raw pointer to copy from, presumed to point to a location in memory
     *         accessible by the \p omp system.
     *  \tparam OtherT \p OtherT shall be convertible to \p T.
     */
    template<typename OtherT>
    __host__ __device__
    explicit pointer(OtherT *ptr) : super_t(ptr) {}

    /*! This constructor allows construction from another pointer-like object with related type.
     *
     *  \param other The \p OtherPointer to copy.
     *  \tparam OtherPointer The system tag associated with \p OtherPointer shall be convertible
     *          to \p thrust::system::omp::tag and its element type shall be convertible to \p T.
     */
    template<typename OtherPointer>
    __host__ __device__
    pointer(const OtherPointer &other,
            typename thrust::detail::enable_if_pointer_is_convertible<
              OtherPointer,
              pointer
            >::type * = 0) : super_t(other) {}

    /*! Assignment operator allows assigning from another pointer-like object with related type.
     *
     *  \param other The other pointer-like object to assign from.
     *  \tparam OtherPointer The system tag associated with \p OtherPointer shall be convertible
     *          to \p thrust::system::omp::tag and its element type shall be convertible to \p T.
     */
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


/*! \p reference is a wrapped reference to an object stored in memory available to the \p omp system.
 *  \p reference is the type of the result of dereferencing a \p omp::pointer.
 *
 *  \tparam T Specifies the type of the referenced object.
 */
template<typename T>
  class reference
    : public thrust::reference<
               T,
               thrust::system::omp::pointer<T>,
               thrust::system::omp::reference<T>
             >
{
  /*! \cond
   */

  private:
    typedef thrust::reference<
      T,
      thrust::system::omp::pointer<T>,
      thrust::system::omp::reference<T>
    > super_t;

  /*! \endcond
   */

  public:
    /*! \cond
     */

    typedef typename super_t::value_type value_type;
    typedef typename super_t::pointer    pointer;

    /*! \endcond
     */

    /*! This constructor initializes this \p reference to refer to an object
     *  pointed to by the given \p pointer. After this \p reference is constructed,
     *  it shall refer to the object pointed to by \p ptr.
     *
     *  \param ptr A \p pointer to copy from.
     */
    __host__ __device__
    explicit reference(const pointer &ptr)
      : super_t(ptr)
    {}

    /*! This constructor accepts a const reference to another \p reference of related type.
     *  After this \p reference is constructed, it shall refer to the same object as \p other.
     *
     *  \param other A \p reference to copy from.
     *  \tparam OtherT The element type of the other \p reference.
     *
     *  \note This constructor is templated primarily to allow initialization of <tt>reference<const T></tt>
     *        from <tt>reference<T></tt>.
     */
    template<typename OtherT>
    __host__ __device__
    reference(const reference<OtherT> &other,
              typename thrust::detail::enable_if_convertible<
                typename reference<OtherT>::pointer,
                pointer
              >::type * = 0)
      : super_t(other)
    {}

    /*! Copy assignment operator copy assigns from another \p reference of related type.
     *
     *  \param other The other \p reference to assign from.
     *  \return <tt>*this</tt>
     *  \tparam OtherT The element type of the other \p reference.
     */
    template<typename OtherT>
    reference &operator=(const reference<OtherT> &other);

    /*! Assignment operator assigns from a \p value_type.
     *
     *  \param x The \p value_type to assign from.
     *  \return <tt>*this</tt>
     */
    reference &operator=(const value_type &x);
}; // end reference

/*! Exchanges the values of two objects referred to by \p reference.
 *  \p x The first \p reference of interest.
 *  \p y The second \p reference ot interest.
 */
template<typename T>
__host__ __device__
void swap(reference<T> x, reference<T> y);

/*! Allocates an area of memory available to Thrust's <tt>omp</tt> system.
 *  \param n Number of bytes to allocate.
 *  \return A <tt>omp::pointer<void></tt> pointing to the beginning of the newly
 *          allocated memory. A null <tt>omp::pointer<void></tt> is returned if
 *          an error occurs.
 *  \note The <tt>omp::pointer<void></tt> returned by this function must be
 *        deallocated with \p omp::free.
 *  \see omp::free
 *  \see std::malloc
 */
inline pointer<void> malloc(std::size_t n);

/*! Allocates a typed area of memory available to Thrust's <tt>omp</tt> system.
 *  \param n Number of elements to allocate.
 *  \return A <tt>omp::pointer<T></tt> pointing to the beginning of the newly
 *          allocated memory. A null <tt>omp::pointer<T></tt> is returned if
 *          an error occurs.
 *  \note The <tt>omp::pointer<T></tt> returned by this function must be
 *        deallocated with \p omp::free.
 *  \see omp::free
 *  \see std::malloc
 */
template<typename T>
inline pointer<T> malloc(std::size_t n);

/*! Deallocates an area of memory previously allocated by <tt>omp::malloc</tt>.
 *  \param ptr A <tt>omp::pointer<void></tt> pointing to the beginning of an area
 *         of memory previously allocated with <tt>omp::malloc</tt>.
 *  \see omp::malloc
 *  \see std::free
 */
inline void free(pointer<void> ptr);

// XXX upon c++11
// template<typename T> using allocator = thrust::detail::malloc_allocator<T,tag,pointer<T> >;

/*! \p omp::allocator is the default allocator used by the \p omp system's containers such as
 *  <tt>omp::vector</tt> if no user-specified allocator is provided. \p omp::allocator allocates
 *  (deallocates) storage with \p omp::malloc (\p omp::free).
 */
template<typename T>
  struct allocator
    : thrust::detail::malloc_allocator<
        T,
        tag,
        pointer<T>
      >
{
  /*! The \p rebind metafunction provides the type of an \p allocator
   *  instantiated with another type.
   *
   *  \tparam U The other type to use for instantiation.
   */
  template<typename U>
    struct rebind
  {
    /*! The typedef \p other gives the type of the rebound \p allocator.
     */
    typedef allocator<U> other;
  };

  /*! No-argument constructor has no effect.
   */
  __host__ __device__
  inline allocator() {}

  /*! Copy constructor has no effect.
   */
  __host__ __device__
  inline allocator(const allocator &) {}

  /*! Constructor from other \p allocator has no effect.
   */
  template<typename U>
  __host__ __device__
  inline allocator(const allocator<U> &) {}

  /*! Destructor has no effect.
   */
  __host__ __device__
  inline ~allocator() {}
}; // end allocator

} // end omp

/*! \}
 */

} // end system

/*! \namespace thrust::omp
 *  \brief \p thrust::omp is a top-level alias for thrust::system::omp.
 */
namespace omp
{

using thrust::system::omp::pointer;
using thrust::system::omp::reference;
using thrust::system::omp::malloc;
using thrust::system::omp::free;
using thrust::system::omp::allocator;

} // end omp

} // end thrust

#include <thrust/system/omp/detail/memory.inl>

// make omp::tag's ADL targets available
#include <thrust/system/cpp/detail/adl_targets.h>

