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


/*! \file device_ptr.h
 *  \brief Defines the interface to a pointer to
 *         a variable which resides on a parallel device.
 */

#pragma once

#include <thrust/detail/config.h>
#include <thrust/iterator/iterator_categories.h>
#include <thrust/detail/type_traits.h>
#include <thrust/detail/device_ptr_traits.h>
#include <ostream>
#include <cstddef>

namespace thrust
{

/*! \addtogroup memory_management Memory Management
 *  \addtogroup memory_management_classes Memory Management Classes
 *  \ingroup memory_management
 *  \{
 */

// forward declarations
template<typename T> class device_reference;

/*! \p device_ptr stores a pointer to an object allocated in device memory. This type
 *  provides type safety when dispatching standard algorithms on ranges resident in
 *  device memory.
 *
 *  \p device_ptr can be created with the functions device_malloc, device_new, or
 *  device_pointer_cast, or by explicitly calling its constructor with a raw pointer.
 *
 *  The raw pointer encapsulated by a \p device_ptr may be obtained by either its <tt>get</tt>
 *  method or the device_pointer_cast free function.
 *
 *  \note \p device_ptr is not a smart pointer; it is the programmer's responsibility to
 *  deallocate memory pointed to by \p device_ptr.
 *
 *  \see device_malloc
 *  \see device_new
 *  \see device_pointer_cast
 *  \see raw_pointer_cast
 */
template<typename T> class device_ptr
{
  public:
    // define iterator_traits types
    typedef typename thrust::detail::device_ptr_traits<T>::iterator_category iterator_category;
    typedef typename thrust::detail::device_ptr_traits<T>::value_type        value_type;
    typedef typename thrust::detail::device_ptr_traits<T>::difference_type   difference_type;
    typedef typename thrust::detail::device_ptr_traits<T>::pointer           pointer;
    typedef typename thrust::detail::device_ptr_traits<T>::reference         reference;

    /*! \p device_ptr's null constructor initializes its raw pointer to \c 0.
     */
    __host__ __device__
    device_ptr(void) : mPtr(0) {}

    /*! \p device_ptr's copy constructor is templated to allow copying to a
     *  const T * from a T *.
     *  
     *  \param ptr A raw pointer to copy from, presumed to point to a location in
     *         device memory.
     */
    template<class Y>
    __host__ __device__
    explicit device_ptr(Y * ptr) : mPtr(ptr) {}

    /*! \p device_ptr's copy constructor allows copying from another device_ptr with related type.
     *  \param ptr The \c device_ptr to copy from.
     */
    __host__ __device__
    device_ptr(const device_ptr<value_type> &ptr) : mPtr(ptr.get()) {}

    /*! \p device_ptr's conversion operator allows conversion to <tt>device_ptr<U></tt> with
     *  \p U related to \p T. For example, <tt>device_ptr<int></tt> may be converted to
     *  <tt>device_ptr<void></tt>.
     *
     *  \return A copy of this \p device_ptr, converted to <tt>device_ptr<U></tt>.
     */
    template<typename U>
    __host__ __device__
    operator device_ptr<U> (void) const
      {return device_ptr<U>(static_cast<U*>(get()));}

    /*! Returns a \p device_ptr whose raw pointer is equal to this \p device_ptr's raw pointer
     *  plus the given sum.
     *
     *  \param rhs The sum to add to this \p device_ptr's raw pointer.
     *  \return <tt>device_ptr(mPtr + rhs)</tt>.
     */
    __host__ __device__
    device_ptr operator+(const difference_type &rhs) const {return device_ptr(mPtr + rhs);}

    /*! Returns a \p device_ptr whose raw pointer is equal to this \p device_ptr's raw pointer
     *  minus the given difference.
     *
     *  \param rhs The difference to subtract to this \p device_ptr's raw pointer.
     *  \return <tt>device_ptr(mPtr - rhs)</tt>.
     */
    __host__ __device__
    device_ptr operator-(const difference_type &rhs) const {return device_ptr(mPtr - rhs);}

    /*! The pre-increment operator increments this \p device_ptr's raw pointer and then returns
     *  a reference to this \p device_ptr
     *  \return <tt>*this</tt>
     */
    __host__ __device__
    device_ptr &operator++(void) {++mPtr; return *this;}

    /*! The post-increment operator copies this \p device_ptr, increments the copy, and then
     *  returns the copy.
     *  \return A copy of this \p device_ptr after being incremented.
     */
    __host__ __device__
    device_ptr operator++(int)
    {
      device_ptr copy(*this);
      ++(*this);
      return copy;
    } // end operator++()

    /*! The pre-decrement operator decrements this \p device_ptr's raw pointer and then returns
     *  a reference to this \p device_ptr
     *  \return <tt>*this</tt>
     */
    __host__ __device__
    device_ptr &operator--(void) {--mPtr; return *this;}

    /*! The post-decrement operator copies this \p device_ptr, decrements the copy, and then
     *  returns the copy.
     *  \return A copy of this \p device_ptr after being decremented.
     */
    __host__ __device__
    device_ptr operator--(int)
    {
      device_ptr copy(*this);
      --(*this);
      return copy;
    } // end operator--()

    /*! The addition assignment operator adds the given sum to this \p device_ptr's raw
     *  pointer and returns this \p device_ptr by reference.
     *  
     *  \param rhs The sum to add to this \p device_ptr's raw pointer.
     *  \return <tt>*this</tt>
     */
    __host__ __device__
    device_ptr &operator+=(difference_type rhs) {mPtr += rhs; return *this;}

    /*! The subtraction assignment operator subtracts the given difference from this
     *  \p device_ptr's raw pointer and returns this \p device_ptr by reference.
     *  
     *  \param rhs The difference to subtract from this \p device_ptr's raw pointer.
     *  \return <tt>*this</tt>
     */
    __host__ __device__
    device_ptr &operator-=(difference_type rhs) {mPtr -= rhs; return *this;}

    /*! The difference operator returns the difference between this \p device_ptr's
     *  raw pointer and that of another.
     *  
     *  \param rhs The \p device_ptr to subtract from this \p device_ptr.
     *  \return The difference between this \p device_ptr's raw pointer and
     *          \p rhs's raw pointer.
     */
    __host__ __device__
    difference_type operator-(const device_ptr &rhs) const {return mPtr - rhs.mPtr;}

    /*! The array subscript operator dereferences this \p device_ptr by the given index.
     *  \param i The index to add to this \p device_ptr's raw pointer before deference.
     *  \return A device_reference referring to the object pointed to by (this \p device_ptr
     *          plus \c i).
     */
    __host__ __device__
    reference operator[](const difference_type &i) const;

    /*! This method dereferences this \p device_ptr.
     *  \return a device_reference referring to the object pointed to by this \p device_ptr.
     */
    __host__ __device__
    reference operator*(void) const;

    /*! This method returns this \p device_ptr's raw pointer.
     *  \return This \p device_ptr's raw pointer.
     */
    __host__ __device__
    T *get(void) const {return mPtr;}

  private:
    T *mPtr;
}; // end device_ptr

/*! Equality comparison operator compares two \p device_ptrs with related types for
 *  equality.
 *  
 *  \param lhs The first device_ptr to compare.
 *  \param rhs The second device_ptr to compare.
 *  \return \c true if and only if \p lhs's raw pointer equals \p rhs's raw pointer;
 *          \c false, otherwise.
 */
template<typename T1, typename T2>
__host__ __device__
inline bool operator==(const device_ptr<T1> &lhs, const device_ptr<T2> &rhs);

/*! Inequality comparison operator compares two \p device_ptrs with related types for
 *  inequality.
 *  
 *  \param lhs The first device_ptr to compare.
 *  \param rhs The second device_ptr to compare.
 *  \return \c true if and only if \p lhs's raw pointer does not equal \p rhs's raw
 *          pointer; \c false, otherwise.
 */
template<typename T1, typename T2>
__host__ __device__
inline bool operator!=(const device_ptr<T1> &lhs, const device_ptr<T2> &rhs);

/*! Less than comparison operator compares two \p device_ptrs with related types.
 *  
 *  \param lhs The first device_ptr to compare.
 *  \param rhs The second device_ptr to compare.
 *  \return \c true if and only if \p lhs's raw pointer is less than \p rhs's raw
 *          pointer; \c false, otherwise.
 */
template<typename T1, typename T2>
__host__ __device__
inline bool operator<(const device_ptr<T1> &lhs, const device_ptr<T2> &rhs);

/*! Less than or equal comparison operator compares two \p device_ptrs with related types.
 *  
 *  \param lhs The first device_ptr to compare.
 *  \param rhs The second device_ptr to compare.
 *  \return \c true if and only if \p lhs's raw pointer is less than or equal to \p rhs's raw
 *          pointer; \c false, otherwise.
 */
template<typename T1, typename T2>
__host__ __device__
inline bool operator<=(const device_ptr<T1> &lhs, const device_ptr<T2> &rhs);

/*! Greater than comparison operator compares two \p device_ptrs with related types.
 *  
 *  \param lhs The first device_ptr to compare.
 *  \param rhs The second device_ptr to compare.
 *  \return \c true if and only if \p lhs's raw pointer is greater than \p rhs's raw
 *          pointer; \c false, otherwise.
 */
template<typename T1, typename T2>
__host__ __device__
inline bool operator>(const device_ptr<T1> &lhs, const device_ptr<T2> &rhs);

/*! Greater than or equal comparison operator compares two \p device_ptrs with related types.
 *  
 *  \param lhs The first device_ptr to compare.
 *  \param rhs The second device_ptr to compare.
 *  \return \c true if and only if \p lhs's raw pointer is greater than or equal to \p rhs's
 *          raw pointer; \c false, otherwise.
 */
template<typename T1, typename T2>
__host__ __device__
inline bool operator>=(const device_ptr<T1> &lhs, const device_ptr<T2> &rhs);

/*! This operator outputs the value of a \p device_ptr's raw pointer to a \p std::basic_ostream.
 *
 *  \param os The std::basic_ostream of interest.
 *  \param p The device_ptr of interest.
 *  \return os.
 */
template<class E, class T, class Y>
inline std::basic_ostream<E, T> &operator<<(std::basic_ostream<E, T> &os, const device_ptr<Y> &p);

/*! \}
 */


/*!
 *  \addtogroup memory_management_functions Memory Management Functions
 *  \ingroup memory_management
 *  \{
 */

/*! \p device_pointer_cast creates a device_ptr from a raw pointer which is presumed to point
 *  to a location in device memory.
 *
 *  \param ptr A raw pointer, presumed to point to a location in device memory.
 *  \return A device_ptr wrapping ptr.
 */
template<typename T>
__host__ __device__
inline device_ptr<T> device_pointer_cast(T *ptr);

/*! \p device_pointer_cast creates a copy of a device_ptr from another device_ptr.
 *  This version is included merely for convenience.
 *
 *  \param ptr A device_ptr.
 *  \return A copy of \p ptr.
 */
template<typename T>
__host__ __device__
inline device_ptr<T> device_pointer_cast(const device_ptr<T> &ptr);

/*! \p raw_pointer_cast creates a "raw" pointer <tt>T*</tt> from a device_ptr, simply
 *  returning the pointer wrapped by the device_ptr.
 *
 *  \param ptr The device_ptr of interest.
 *  \return <tt>ptr.get()</tt>.
 */
template<typename T>
__host__ __device__
inline T *raw_pointer_cast(const device_ptr<T> &ptr);

/*! \p raw_pointer_cast creates a copy of a "raw" pointer <tt>T*</tt>.
 *  This version is included merely for convenience.
 *
 *  \param ptr The pointer of interest.
 *  \return \p ptr
 */
template<typename T>
__host__ __device__
inline T *raw_pointer_cast(T *ptr);

/*! \}
 */

} // end thrust

#include <thrust/detail/device_ptr.inl>

