/*
 *  Copyright 2008-2009 NVIDIA Corporation
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


/*! \file device_reference.h
 *  \brief Defines the interface to a reference to
 *         a variable which resides on a CUDA device.
 */

#pragma once

#include <thrust/detail/config.h>
#include <thrust/device_ptr.h>

namespace thrust
{

/*! \addtogroup memory_management_classes Memory Management Classes
 *  \ingroup memory_management
 *  \{
 */

/*! \p device_reference acts as a reference to an object stored in device memory.
 *  \p device_reference is not intended to be used directly; rather, this type
 *  is the result of deferencing a \p device_ptr. Similarly, taking the address of
 *  a \p device_reference yields a \p device_ptr.
 *
 *  \see device_ptr
 */
template<typename T>
  struct device_reference
{
  typedef device_ptr<T> pointer;

  /*! This copy constructor accepts a const reference to another
   *  \p device_reference. After this \p device_reference is constructed,
   *  it shall refer to the same object as \p ref.
   *  
   *  \param ref A \p device_reference to copy from.
   *
   *  The following code snippet demonstrates the semantics of this
   *  copy constructor.
   *
   *  \code
   *  #include <thrust/device_vector.h>
   *  #include <assert.h>
   *  ...
   *  thrust::device_vector<int> v(1,0);
   *  thrust::device_reference<int> ref = v[0];
   *
   *  // ref equals the object at v[0]
   *  assert(ref1 == v[0]);
   *
   *  // the address of ref equals the address of v[0]
   *  assert(&ref == &v[0]);
   *
   *  // modifying v[0] modifies ref
   *  v[0] = 13;
   *  assert(ref == 13);
   *  \endcode
   */
  device_reference(const device_reference &ref);

  /*! This copy constructor initializes this \p device_reference
   *  to refer to an object pointed to by the given \p device_ptr. After
   *  this \p device_reference is constructed, it shall refer to the
   *  object pointed to by \p ptr.
   *
   *  \param ptr A \p device_ptr to copy from.
   *
   *  The following code snippet demonstrates the semantic of this
   *  copy constructor.
   *
   *  \code
   *  #include <thrust/device_vector.h>
   *  #include <assert.h>
   *  ...
   *  thrust::device_vector<int> v(1,0);
   *  thrust::device_ptr<int> ptr = &v[0];
   *  thrust::device_reference<int> ref(ptr);
   *
   *  // ref equals the object pointed to by ptr
   *  assert(ref == *ptr);
   *
   *  // the address of ref equals ptr
   *  assert(&ref == ptr);
   *
   *  // modifying *ptr modifies ref
   *  *ptr = 13;
   *  assert(ref == 13);
   *  \endcode
   */
  explicit device_reference(const pointer &ptr);

  /*! Address-of operator returns a \p device_ptr pointing to the object
   *  referenced by this \p device_reference. It does not return the
   *  address of this \p device_reference.
   *
   *  \return A \p device_ptr pointing to the object this
   *  \p device_reference references.
   */
  pointer operator&(void) const;

  /*! Assignment operator copies the value of the given object to the
   *  object referenced by this \p device_reference.
   *  
   *  \param v The value to copy from.
   *  \return This \p device_reference.
   */
  device_reference &operator=(const T &v);

  /*! Assignment operator copies the value of the object referenced by
   *  the given \p device_reference to the object referenced by this
   *  \p device_reference.
   *
   *  \param ref The \p device_reference to copy from.
   *  \return This \p device_reference.
   *
   *  \bug This needs to be templated on the type of the \p device_reference
   *       to copy from.
   */
  device_reference &operator=(const device_reference &ref);

  /*! Prefix increment operator increments the object referenced by this
   *  \p device_reference.
   *
   *  \return <tt>*this</tt>
   *  
   *  The following code snippet demonstrates the semantics of
   *  \p device_reference's prefix increment operator.
   *
   *  \code
   *  #include <thrust/device_vector.h>
   *  #include <assert.h>
   *  ...
   *  thrust::device_vector<int> v(1,0);
   *  thrust::device_ptr<int> ptr = &v[0];
   *  thrust::device_reference<int> ref(ptr);
   *
   *  // ref equals 0
   *  assert(ref == 0);
   *
   *  // the object pointed to by ptr equals 1
   *  assert(*ptr == 1);
   *
   *  // v[0] equals 1
   *  assert(v[0] == 1);
   *
   *  // increment ref
   *  ++ref;
   *
   *  // ref equals 1
   *  assert(ref == 1);
   *
   *  // the object pointed to by ptr equals 1
   *  assert(*ptr == 1);
   *
   *  // v[0] equals 1
   *  assert(v[0] == 1);
   *  \endcode
   *
   *  \note The increment executes as if it were executed on the host.
   *  This may change in a later version.
   */
  device_reference &operator++(void);

  /*! Postfix increment operator copies the object referenced by this
   *  \p device_reference, increments the object referenced by this
   *  \p device_reference, and returns the copy.
   *
   *  \return A copy of the object referenced by this \p device_reference
   *          before being incremented.
   *
   *  The following code snippet demonstrates the semantics of
   *  \p device_reference's postfix increment operator.
   *
   *  \code
   *  #include <thrust/device_vector.h>
   *  #include <assert.h>
   *  ...
   *  thrust::device_vector<int> v(1,0);
   *  thrust::device_ptr<int> ptr = &v[0];
   *  thrust::device_reference<int> ref(ptr);
   *
   *  // ref equals 0
   *  assert(ref == 0);
   *
   *  // the object pointed to by ptr equals 0
   *  assert(*ptr == 0);
   *
   *  // v[0] equals 0
   *  assert(v[0] == 0);
   *
   *  // increment ref
   *  int x = ref++;
   *
   *  // x equals 0
   *  assert(x == 0)
   *
   *  // ref equals 1
   *  assert(ref == 1);
   *
   *  // the object pointed to by ptr equals 1
   *  assert(*ptr == 1);
   *
   *  // v[0] equals 1
   *  assert(v[0] == 1);
   *  \endcode
   *
   *  \note The increment executes as if it were executed on the host.
   *  This may change in a later version.
   */
  T operator++(int);

  /*! Addition assignment operator add-assigns the object referenced by this
   *  \p device_reference and returns this \p device_reference.
   *
   *  \param rhs The right hand side of the add-assignment.
   *  \return <tt>*this</tt>.
   *
   *  The following code snippet demonstrates the semantics of
   *  \p device_reference's addition assignment operator.
   *
   *  \code
   *  #include <thrust/device_vector.h>
   *  #include <assert.h>
   *  ...
   *  thrust::device_vector<int> v(1,0);
   *  thrust::device_ptr<int> ptr = &v[0];
   *  thrust::device_reference<int> ref(ptr);
   *
   *  // ref equals 0
   *  assert(ref == 0);
   *
   *  // the object pointed to by ptr equals 0
   *  assert(*ptr == 0);
   *
   *  // v[0] equals 0
   *  assert(v[0] == 0);
   *
   *  // add-assign ref
   *  ref += 5;
   *
   *  // ref equals 5
   *  assert(ref == 5);
   *
   *  // the object pointed to by ptr equals 5
   *  assert(*ptr == 5);
   *
   *  // v[0] equals 5
   *  assert(v[0] == 5);
   *  \endcode
   *
   *  \note The add-assignment executes as as if it were executed on the host.
   *  This may change in a later version.
   */
  device_reference &operator+=(const T &rhs);

  /*! Prefix decrement operator decrements the object referenced by this
   *  \p device_reference.
   *
   *  \return <tt>*this</tt>
   *  
   *  The following code snippet demonstrates the semantics of
   *  \p device_reference's prefix decrement operator.
   *
   *  \code
   *  #include <thrust/device_vector.h>
   *  #include <assert.h>
   *  ...
   *  thrust::device_vector<int> v(1,0);
   *  thrust::device_ptr<int> ptr = &v[0];
   *  thrust::device_reference<int> ref(ptr);
   *
   *  // ref equals 0
   *  assert(ref == 0);
   *
   *  // the object pointed to by ptr equals 0
   *  assert(*ptr == 0);
   *
   *  // v[0] equals 0
   *  assert(v[0] == 0);
   *
   *  // decrement ref
   *  --ref;
   *
   *  // ref equals -1
   *  assert(ref == -1);
   *
   *  // the object pointed to by ptr equals -1
   *  assert(*ptr == -1);
   *
   *  // v[0] equals -1
   *  assert(v[0] == -1);
   *  \endcode
   *
   *  \note The decrement executes as if it were executed on the host.
   *  This may change in a later version.
   */
  device_reference &operator--(void);

  /*! Postfix decrement operator copies the object referenced by this
   *  \p device_reference, decrements the object referenced by this
   *  \p device_reference, and returns the copy.
   *
   *  \return A copy of the object referenced by this \p device_reference
   *          before being decremented.
   *
   *  The following code snippet demonstrates the semantics of
   *  \p device_reference's postfix decrement operator.
   *
   *  \code
   *  #include <thrust/device_vector.h>
   *  #include <assert.h>
   *  ...
   *  thrust::device_vector<int> v(1,0);
   *  thrust::device_ptr<int> ptr = &v[0];
   *  thrust::device_reference<int> ref(ptr);
   *
   *  // ref equals 0
   *  assert(ref == 0);
   *
   *  // the object pointed to by ptr equals 0
   *  assert(*ptr == 0);
   *
   *  // v[0] equals 0
   *  assert(v[0] == 0);
   *
   *  // decrement ref
   *  int x = ref--;
   *
   *  // x equals 0
   *  assert(x == 0)
   *
   *  // ref equals -1
   *  assert(ref == -1);
   *
   *  // the object pointed to by ptr equals -1
   *  assert(*ptr == -1);
   *
   *  // v[0] equals -1
   *  assert(v[0] == -1);
   *  \endcode
   *
   *  \note The decrement executes as if it were executed on the host.
   *  This may change in a later version.
   */
  T operator--(int);

  /*! Subtraction assignment operator subtract-assigns the object referenced by this
   *  \p device_reference and returns this \p device_reference.
   *
   *  \param rhs The right hand side of the subtraction-assignment.
   *  \return <tt>*this</tt>.
   *
   *  The following code snippet demonstrates the semantics of
   *  \p device_reference's addition assignment operator.
   *
   *  \code
   *  #include <thrust/device_vector.h>
   *  #include <assert.h>
   *  ...
   *  thrust::device_vector<int> v(1,0);
   *  thrust::device_ptr<int> ptr = &v[0];
   *  thrust::device_reference<int> ref(ptr);
   *
   *  // ref equals 0
   *  assert(ref == 0);
   *
   *  // the object pointed to by ptr equals 0
   *  assert(*ptr == 0);
   *
   *  // v[0] equals 0
   *  assert(v[0] == 0);
   *
   *  // subtract-assign ref
   *  ref -= 5;
   *
   *  // ref equals -5
   *  assert(ref == -5);
   *
   *  // the object pointed to by ptr equals -5
   *  assert(*ptr == -5);
   *
   *  // v[0] equals -5
   *  assert(v[0] == -5);
   *  \endcode
   *
   *  \note The subtract-assignment executes as as if it were executed on the host.
   *  This may change in a later version.
   */
  device_reference &operator-=(const T &rhs);

  /*! Multiplication assignment operator multiply-assigns the object referenced by this
   *  \p device_reference and returns this \p device_reference.
   *
   *  \param rhs The right hand side of the multiply-assignment.
   *  \return <tt>*this</tt>.
   *
   *  The following code snippet demonstrates the semantics of
   *  \p device_reference's multiply assignment operator.
   *
   *  \code
   *  #include <thrust/device_vector.h>
   *  #include <assert.h>
   *  ...
   *  thrust::device_vector<int> v(1,1);
   *  thrust::device_ptr<int> ptr = &v[0];
   *  thrust::device_reference<int> ref(ptr);
   *
   *  // ref equals 1
   *  assert(ref == 1);
   *
   *  // the object pointed to by ptr equals 1
   *  assert(*ptr == 1);
   *
   *  // v[0] equals 1
   *  assert(v[0] == 1);
   *
   *  // multiply-assign ref
   *  ref *= 5;
   *
   *  // ref equals 5
   *  assert(ref == 5);
   *
   *  // the object pointed to by ptr equals 5
   *  assert(*ptr == 5);
   *
   *  // v[0] equals 5
   *  assert(v[0] == 5);
   *  \endcode
   *
   *  \note The multiply-assignment executes as as if it were executed on the host.
   *  This may change in a later version.
   */
  device_reference &operator*=(const T &rhs);

  /*! Division assignment operator divide-assigns the object referenced by this
   *  \p device_reference and returns this \p device_reference.
   *
   *  \param rhs The right hand side of the divide-assignment.
   *  \return <tt>*this</tt>.
   *
   *  The following code snippet demonstrates the semantics of
   *  \p device_reference's divide assignment operator.
   *
   *  \code
   *  #include <thrust/device_vector.h>
   *  #include <assert.h>
   *  ...
   *  thrust::device_vector<int> v(1,5);
   *  thrust::device_ptr<int> ptr = &v[0];
   *  thrust::device_reference<int> ref(ptr);
   *
   *  // ref equals 5
   *  assert(ref == 5);
   *
   *  // the object pointed to by ptr equals 5
   *  assert(*ptr == 5);
   *
   *  // v[0] equals 5
   *  assert(v[0] == 5);
   *
   *  // divide-assign ref
   *  ref /= 5;
   *
   *  // ref equals 1
   *  assert(ref == 1);
   *
   *  // the object pointed to by ptr equals 1
   *  assert(*ptr == 1);
   *
   *  // v[0] equals 1
   *  assert(v[0] == 1);
   *  \endcode
   *
   *  \note The divide-assignment executes as as if it were executed on the host.
   *  This may change in a later version.
   */
  device_reference &operator/=(const T &rhs);

  /*! Modulation assignment operator modulus-assigns the object referenced by this
   *  \p device_reference and returns this \p device_reference.
   *
   *  \param rhs The right hand side of the divide-assignment.
   *  \return <tt>*this</tt>.
   *
   *  The following code snippet demonstrates the semantics of
   *  \p device_reference's divide assignment operator.
   *
   *  \code
   *  #include <thrust/device_vector.h>
   *  #include <assert.h>
   *  ...
   *  thrust::device_vector<int> v(1,5);
   *  thrust::device_ptr<int> ptr = &v[0];
   *  thrust::device_reference<int> ref(ptr);
   *
   *  // ref equals 5
   *  assert(ref == 5);
   *
   *  // the object pointed to by ptr equals 5
   *  assert(*ptr == 5);
   *
   *  // v[0] equals 5
   *  assert(v[0] == 5);
   *
   *  // modulus-assign ref
   *  ref %= 5;
   *
   *  // ref equals 0
   *  assert(ref == 0);
   *
   *  // the object pointed to by ptr equals 0
   *  assert(*ptr == 0);
   *
   *  // v[0] equals 0
   *  assert(v[0] == 0);
   *  \endcode
   *
   *  \note The modulus-assignment executes as as if it were executed on the host.
   *  This may change in a later version.
   */
  device_reference &operator%=(const T &rhs);

  /*! Bitwise left shift assignment operator left shift-assigns the object referenced by this
   *  \p device_reference and returns this \p device_reference.
   *
   *  \param rhs The right hand side of the left shift-assignment.
   *  \return <tt>*this</tt>.
   *
   *  The following code snippet demonstrates the semantics of
   *  \p device_reference's left shift assignment operator.
   *
   *  \code
   *  #include <thrust/device_vector.h>
   *  #include <assert.h>
   *  ...
   *  thrust::device_vector<int> v(1,1);
   *  thrust::device_ptr<int> ptr = &v[0];
   *  thrust::device_reference<int> ref(ptr);
   *
   *  // ref equals 1
   *  assert(ref == 1);
   *
   *  // the object pointed to by ptr equals 1
   *  assert(*ptr == 1);
   *
   *  // v[0] equals 1
   *  assert(v[0] == 1);
   *
   *  // left shift-assign ref
   *  ref <<= 1;
   *
   *  // ref equals 2
   *  assert(ref == 2);
   *
   *  // the object pointed to by ptr equals 2
   *  assert(*ptr == 2);
   *
   *  // v[0] equals 2
   *  assert(v[0] == 2);
   *  \endcode
   *
   *  \note The left shift-assignment executes as as if it were executed on the host.
   *  This may change in a later version.
   */
  device_reference &operator<<=(const T &rhs);

  /*! Bitwise right shift assignment operator right shift-assigns the object referenced by this
   *  \p device_reference and returns this \p device_reference.
   *
   *  \param rhs The right hand side of the right shift-assignment.
   *  \return <tt>*this</tt>.
   *
   *  The following code snippet demonstrates the semantics of
   *  \p device_reference's right shift assignment operator.
   *
   *  \code
   *  #include <thrust/device_vector.h>
   *  #include <assert.h>
   *  ...
   *  thrust::device_vector<int> v(1,2);
   *  thrust::device_ptr<int> ptr = &v[0];
   *  thrust::device_reference<int> ref(ptr);
   *
   *  // ref equals 2
   *  assert(ref == 2);
   *
   *  // the object pointed to by ptr equals 2
   *  assert(*ptr == 2);
   *
   *  // v[0] equals 2
   *  assert(v[0] == 2);
   *
   *  // right shift-assign ref
   *  ref >>= 1;
   *
   *  // ref equals 1
   *  assert(ref == 1);
   *
   *  // the object pointed to by ptr equals 1
   *  assert(*ptr == 1);
   *
   *  // v[0] equals 1
   *  assert(v[0] == 1);
   *  \endcode
   *
   *  \note The right shift-assignment executes as as if it were executed on the host.
   *  This may change in a later version.
   */
  device_reference &operator>>=(const T &rhs);

  /*! Bitwise AND assignment operator AND-assigns the object referenced by this
   *  \p device_reference and returns this \p device_reference.
   *
   *  \param rhs The right hand side of the AND-assignment.
   *  \return <tt>*this</tt>.
   *
   *  The following code snippet demonstrates the semantics of
   *  \p device_reference's AND assignment operator.
   *
   *  \code
   *  #include <thrust/device_vector.h>
   *  #include <assert.h>
   *  ...
   *  thrust::device_vector<int> v(1,1);
   *  thrust::device_ptr<int> ptr = &v[0];
   *  thrust::device_reference<int> ref(ptr);
   *
   *  // ref equals 1
   *  assert(ref == 1);
   *
   *  // the object pointed to by ptr equals 1
   *  assert(*ptr == 1);
   *
   *  // v[0] equals 1
   *  assert(v[0] == 1);
   *
   *  // right AND-assign ref
   *  ref &= 0;
   *
   *  // ref equals 0
   *  assert(ref == 0);
   *
   *  // the object pointed to by ptr equals 0
   *  assert(*ptr == 0);
   *
   *  // v[0] equals 0
   *  assert(v[0] == 0);
   *  \endcode
   *
   *  \note The AND-assignment executes as as if it were executed on the host.
   *  This may change in a later version.
   */
  device_reference &operator&=(const T &rhs);

  /*! Bitwise OR assignment operator OR-assigns the object referenced by this
   *  \p device_reference and returns this \p device_reference.
   *
   *  \param rhs The right hand side of the OR-assignment.
   *  \return <tt>*this</tt>.
   *
   *  The following code snippet demonstrates the semantics of
   *  \p device_reference's OR assignment operator.
   *
   *  \code
   *  #include <thrust/device_vector.h>
   *  #include <assert.h>
   *  ...
   *  thrust::device_vector<int> v(1,0);
   *  thrust::device_ptr<int> ptr = &v[0];
   *  thrust::device_reference<int> ref(ptr);
   *
   *  // ref equals 0
   *  assert(ref == 0);
   *
   *  // the object pointed to by ptr equals 0
   *  assert(*ptr == 0);
   *
   *  // v[0] equals 0
   *  assert(v[0] == 0);
   *
   *  // right OR-assign ref
   *  ref |= 1;
   *
   *  // ref equals 1
   *  assert(ref == 1);
   *
   *  // the object pointed to by ptr equals 1
   *  assert(*ptr == 1);
   *
   *  // v[0] equals 1
   *  assert(v[0] == 1);
   *  \endcode
   *
   *  \note The OR-assignment executes as as if it were executed on the host.
   *  This may change in a later version.
   */
  device_reference &operator|=(const T &rhs);

  /*! Bitwise XOR assignment operator XOR-assigns the object referenced by this
   *  \p device_reference and returns this \p device_reference.
   *
   *  \param rhs The right hand side of the XOR-assignment.
   *  \return <tt>*this</tt>.
   *
   *  The following code snippet demonstrates the semantics of
   *  \p device_reference's XOR assignment operator.
   *
   *  \code
   *  #include <thrust/device_vector.h>
   *  #include <assert.h>
   *  ...
   *  thrust::device_vector<int> v(1,1);
   *  thrust::device_ptr<int> ptr = &v[0];
   *  thrust::device_reference<int> ref(ptr);
   *
   *  // ref equals 1
   *  assert(ref == 1);
   *
   *  // the object pointed to by ptr equals 1
   *  assert(*ptr == 1);
   *
   *  // v[0] equals 1
   *  assert(v[0] == 1);
   *
   *  // right XOR-assign ref
   *  ref ^= 1;
   *
   *  // ref equals 0
   *  assert(ref == 0);
   *
   *  // the object pointed to by ptr equals 0
   *  assert(*ptr == 0);
   *
   *  // v[0] equals 0
   *  assert(v[0] == 0);
   *  \endcode
   *
   *  \note The XOR-assignment executes as as if it were executed on the host.
   *  This may change in a later version.
   */
  device_reference &operator^=(const T &rhs);

  /*! Conversion operator converts this \p device_reference to T
   *  by returning a copy of the object referenced by this
   *  \p device_reference.
   *
   *  \return A copy of the object referenced by this \p device_reference.
   */
  operator T (void) const;

  private:
    pointer mPtr;
}; // end device_reference

/*! \}
 */

} // end thrust

#include <thrust/detail/device_reference.inl>

