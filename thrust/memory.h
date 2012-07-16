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

/*! \file thrust/memory.h
 *  \brief Abstractions for Thrust's memory model.
 */

#include <thrust/detail/config.h>

#include <thrust/detail/type_traits/pointer_traits.h>
#include <thrust/detail/pointer.h>
#include <thrust/detail/reference.h>
#include <thrust/detail/raw_pointer_cast.h>
#include <thrust/detail/raw_reference_cast.h>
#include <thrust/detail/malloc_and_free.h>
#include <thrust/detail/temporary_buffer.h>

namespace thrust
{

/*! \addtogroup memory_management Memory Management
 *  \addtogroup memory_management_classes Memory Management Classes
 *  \ingroup memory_management
 *  \{
 */

/*! \p pointer stores a pointer to an object allocated in memory. Like \p device_ptr, this
 *  type ensures type safety when dispatching standard algorithms on ranges resident in memory.
 *
 *  \p pointer generalizes \p device_ptr by relaxing the backend system associated with the \p pointer.
 *  Instead of the backend system specified by \p THRUST_DEFAULT_DEVICE_BACKEND, \p pointer's
 *  system is given by its second template parameter, \p Tag. For the purpose of Thrust dispatch,
 *  <tt>device_ptr<Element></tt> and <tt>pointer<Element,device_system_tag></tt> are considered equivalent.
 *
 *  The raw pointer encapsulated by a \p pointer may be obtained through its <tt>get</tt> member function
 *  or the \p raw_pointer_cast free function.
 *
 *  \tparam Element specifies the type of the pointed-to object.
 *
 *  \tparam Tag specifies the system with which this \p pointer is associated. This may be any Thrust
 *          backend system, or a user-defined tag.
 *
 *  \tparam Reference allows the client to specify the reference type returned upon derereference.
 *          By default, this type is <tt>reference<Element,pointer></tt>.
 *
 *  \tparam Derived allows the client to specify the name of the derived type when \p pointer is used as
 *          a base class. This is useful to ensure that arithmetic on values of the derived type return
 *          values of the derived type as a result. By default, this type is <tt>pointer<Element,Tag,Reference></tt>.
 *
 *  \note \p pointer is not a smart pointer; it is the client's responsibility to deallocate memory
 *        pointer to by \p pointer.
 *
 *  \see device_ptr
 *  \see reference
 *  \see raw_pointer_cast
 */
// define pointer for the purpose of Doxygenating it
// it is actually defined elsewhere
#if 0
template<typename Element, typename Tag, typename Reference = thrust::use_default, typename Derived = thrust::use_default>
  class pointer
{
  public:
    /*! The type of the raw pointer
     */
    typedef typename super_t::base_type raw_pointer;
    
    /*! \p pointer's default constructor initializes its encapsulated pointer to \c 0
     */
    __host__ __device__
    pointer();

    /*! This constructor allows construction of a <tt>pointer<const T, ...></tt> from a <tt>T*</tt>.
     *
     *  \param ptr A raw pointer to copy from, presumed to point to a location in \p Tag's memory.
     *  \tparam OtherElement \p OtherElement shall be convertible to \p Element.
     */
    template<typename OtherElement>
    __host__ __device__
    explicit pointer(OtherElement *ptr);

    /*! This contructor allows initialization from another pointer-like object.
     *
     *  \param other The \p OtherPointer to copy.
     *
     *  \tparam OtherPointer The tag associated with \p OtherPointer shall be convertible to \p Tag,
     *                       and its element type shall be convertible to \p Element.
     */
    template<typename OtherPointer>
    __host__ __device__
    pointer(const OtherPointer &other,
            typename thrust::detail::enable_if_pointer_is_convertible<
              OtherPointer,
              pointer<Element,Tag,Reference,Derived>
            >::type * = 0);

    /*! Assignment operator allows assigning from another pointer-like object with related type.
     *
     *  \param other The other pointer-like object to assign from.
     *  \return <tt>*this</tt>
     *
     *  \tparam OtherPointer The tag associated with \p OtherPointer shall be convertible to \p Tag,
     *                       and its element type shall be convertible to \p Element.
     */
    template<typename OtherPointer>
    __host__ __device__
    typename thrust::detail::enable_if_pointer_is_convertible<
      OtherPointer,
      pointer,
      derived_type &
    >::type
    operator=(const OtherPointer &other);

    /*! \p get returns this \p pointer's encapsulated raw pointer.
     *  \return This \p pointer's raw pointer.
     */
    __host__ __device__
    Element *get() const;
};
#endif

/*! \p reference is a wrapped reference to an object stored in memory. \p reference generalizes
 *  \p device_reference by relaxing the type of pointer associated with the object. \p reference
 *  is the type of the result of dereferencing a tagged pointer-like object such as \p pointer, and
 *  intermediates operations on objects existing in a remote memory.
 *
 *  \tparam Element specifies the type of the referent object.
 *  \tparam Pointer specifies the type of the result of taking the address of \p reference.
 *  \tparam Derived allows the client to specify the name of the derived type when \p reference is used as
 *          a base class. This is useful to ensure that assignment to objects of the derived type return
 *          values of the derived type as a result. By default, this type is <tt>reference<Element,Pointer></tt>.
 */
// define pointer for the purpose of Doxygenating it
// it is actually defined elsewhere
#if 0
template<typename Element, typename Pointer, typename Derived = thrust::use_default>
  class reference
{
  public:
    /*! The type of this \p reference's wrapped pointers.
     */
    typedef Pointer                                              pointer;

    /*! The \p value_type of this \p reference.
     */
    typedef typename thrust::detail::remove_const<Element>::type value_type;

    /*! This copy constructor initializes this \p reference
     *  to refer to an object pointed to by the given \p pointer. After
     *  this \p reference is constructed, it shall refer to the
     *  object pointed to by \p ptr.
     *
     *  \param ptr A \p pointer to copy from.
     */
    __host__ __device__
    explicit reference(const pointer &ptr);

    /*! This copy constructor accepts a const reference to another
     *  \p reference of related type. After this \p reference is constructed,
     *  it shall refer to the same object as \p other.
     *  
     *  \param other A \p reference to copy from.
     *  \tparam OtherElement the element type of the other \p reference.
     *  \tparam OtherPointer the pointer type of the other \p reference.
     *  \tparam OtherDerived the derived type of the other \p reference.
     *
     *  \note This constructor is templated primarily to allow initialization of 
     *  <tt>reference<const T,...></tt> from <tt>reference<T,...></tt>.
     */
    template<typename OtherElement, typename OtherPointer, typename OtherDerived>
    __host__ __device__
    reference(const reference<OtherElement,OtherPointer,OtherDerived> &other,
              typename thrust::detail::enable_if_convertible<
                typename reference<OtherElement,OtherPointer,OtherDerived>::pointer,
                pointer
              >::type * = 0);

    /*! Copy assignment operator copy assigns from another \p reference.
     *
     *  \param other The other \p reference to assign from.
     *  \return <tt>static_cast<derived_type&>(*this)</tt>
     */
    __host__ __device__
    derived_type &operator=(const reference &other);

    /*! Assignment operator copy assigns from another \p reference of related type.
     *
     *  \param other The other \p reference to assign from.
     *  \return <tt>static_cast<derived_type&>(*this)</tt>
     *
     *  \tparam OtherElement the element type of the other \p reference.
     *  \tparam OtherPointer the pointer type of the other \p reference.
     *  \tparam OtherDerived the derived type of the other \p reference.
     */
    template<typename OtherElement, typename OtherPointer, typename OtherDerived>
    __host__ __device__
    derived_type &operator=(const reference<OtherElement,OtherPointer,OtherDerived> &other);

    /*! Assignment operator assigns from a \p value_type.
     *
     *  \param x The \p value_type to assign from.
     *  \return <tt>static_cast<derived_type&>(*this)</tt>.
     */
    __host__ __device__
    derived_type &operator=(const value_type &x);

    /*! Address-of operator returns a \p pointer pointing to the object
     *  referenced by this \p reference. It does not return the address of this
     *  \p reference.
     *
     *  \return A \p pointer pointing to the referenct object.
     */
    __host__ __device__
    pointer operator&() const;

    /*! Conversion operator converts this \p reference to \p value_type by
     *  returning a copy of the referent object.
     *  
     *  \return A copy of the referent object.
     */
    __host__ __device__
    operator value_type () const;

    /*! Swaps the value of the referent object with another.
     *
     *  \param other The other \p reference with which to swap.
     *  \note The argument is of type \p derived_type rather than \p reference.
     */
    __host__ __device__
    void swap(derived_type &other);

    /*! Prefix increment operator increments the referent object.
     *
     *  \return <tt>static_Cast<derived_type&>(*this)</tt>.
     *
     *  \note Documentation for other arithmetic operators omitted for brevity.
     */
    derived_type &operator++();
};
#endif

/*! \}
 */

/*!
 *  \addtogroup memory_management_functions Memory Management Functions
 *  \ingroup memory_management
 *  \{
 */

/*! \p raw_pointer_cast creates a "raw" pointer from a pointer-like type,
 *  simply returning the wrapped pointer, should it exist.
 *
 *  \param ptr The pointer of interest.
 *  \return <tt>ptr.get()</tt>, if the expression is well formed; <tt>ptr</tt>, otherwise.
 *  \see raw_reference_cast
 */
template<typename Pointer>
__host__ __device__
inline typename thrust::detail::pointer_traits<Pointer>::raw_pointer
  raw_pointer_cast(const Pointer &ptr);

/*! \p raw_reference_cast creates a "raw" reference from a wrapped reference type,
 *  simply returning the underlying reference, should it exist.
 *
 *  If the argument is not a reference wrapper, the result is a reference to the argument.
 *
 *  \param ref The reference of interest.
 *  \return <tt>*thrust::raw_pointer_cast(&ref)</tt>.
 *  \note There are two versions of \p raw_reference_cast. One for <tt>const</tt> references,
 *        and one for non-<tt>const</tt>.
 *  \see raw_pointer_cast
 */
template<typename T>
__host__ __device__
inline typename detail::raw_reference<T>::type
  raw_reference_cast(T &ref);

/*! \p raw_reference_cast creates a "raw" reference from a wrapped reference type,
 *  simply returning the underlying reference, should it exist.
 *
 *  If the argument is not a reference wrapper, the result is a reference to the argument.
 *
 *  \param ref The reference of interest.
 *  \return <tt>*thrust::raw_pointer_cast(&ref)</tt>.
 *  \note There are two versions of \p raw_reference_cast. One for <tt>const</tt> references,
 *        and one for non-<tt>const</tt>.
 *  \see raw_pointer_cast
 */
template<typename T>
__host__ __device__
inline typename detail::raw_reference<const T>::type
  raw_reference_cast(const T &ref);

/*! \}
 */

} // end thrust

