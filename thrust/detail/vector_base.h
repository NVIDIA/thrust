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


/*! \file vector_base.h
 *  \brief Defines the interface to a base class for
 *         host_vector & device_vector.
 */

#pragma once

#include <thrust/iterator/iterator_adaptor.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/detail/make_device_dereferenceable.h>
#include <thrust/detail/type_traits.h>
#include <thrust/utility.h>
#include <vector>

namespace thrust
{

namespace detail
{

template<typename T, typename Alloc>
  class vector_base
{
  public:
    // typedefs
    typedef T                               value_type;
    typedef typename Alloc::pointer         pointer;
    typedef typename Alloc::const_pointer   const_pointer;
    typedef typename Alloc::reference       reference;
    typedef typename Alloc::const_reference const_reference;
    typedef std::size_t                     size_type;
    typedef typename Alloc::difference_type difference_type;
    typedef Alloc                           allocator_type;

    class iterator
      : public experimental::iterator_adaptor<iterator,
                                              pointer,
                                              value_type,
                                              typename thrust::iterator_traits<pointer>::iterator_category,
                                              reference,
                                              pointer,
                                              difference_type>
    {
      public:
        __host__ __device__
        iterator() {}

        __host__ __device__
        iterator(pointer p)
          : iterator::iterator_adaptor_(p) {}


        // XXX from here down needs to be private, but nvcc can't compile it

        // device_dereferenceable_iterator_traits requires the following types to be defined
        typedef typename device_dereferenceable_iterator_traits<pointer>::device_dereferenceable_type device_dereferenceable_pointer;

        // forward declaration
        class device_dereferenceable_type;

        // shorthand: name the base class of device_dereferenceable_type
        typedef experimental::iterator_adaptor<device_dereferenceable_type,
                                               device_dereferenceable_pointer,
                                               typename thrust::iterator_traits<device_dereferenceable_pointer>::value_type,
                                               typename thrust::iterator_traits<device_dereferenceable_pointer>::iterator_category,
                                               typename thrust::iterator_traits<device_dereferenceable_pointer>::reference,
                                               typename thrust::iterator_traits<device_dereferenceable_pointer>::pointer,
                                               typename thrust::iterator_traits<device_dereferenceable_pointer>::difference_type> device_dereferenceable_type_base;


        class device_dereferenceable_type
          : public device_dereferenceable_type_base
        {
          public:
            __host__ __device__
            device_dereferenceable_type() {}

            __host__ __device__
            device_dereferenceable_type(device_dereferenceable_pointer p)
              : device_dereferenceable_type_base(p) {}
        }; // end device_dereferenceable_type

      private:

        // befriend device_dereferenceable_iterator_traits so he can get at device_dereferenceable_type
        friend struct thrust::detail::device_dereferenceable_iterator_traits<iterator>;

        device_dereferenceable_type device_dereferenceable(void)
        {
          return device_dereferenceable_type(make_device_dereferenceable<pointer>::transform(this->base()));
        } // end device_dereferenceable()

        // befriend make_device_dereferenceable so he can get at device_dereferenceable()
        friend struct thrust::detail::make_device_dereferenceable<iterator>;
    }; // end iterator

    class const_iterator
      : public experimental::iterator_adaptor<const_iterator,
                                              const_pointer,
                                              value_type,
                                              typename thrust::iterator_traits<const_pointer>::iterator_category,
                                              const_reference,
                                              const_pointer,
                                              difference_type>
    {
      public:
        __host__
        const_iterator() {}

        __host__
        const_iterator(const_pointer p)
          : const_iterator::iterator_adaptor_(p) {}

        __host__ __device__
        const_iterator(iterator const & other)
          : const_iterator::iterator_adaptor_(other.base()) {}

        // XXX from here down needs to be private, but nvcc can't compile it

        // device_dereferenceable_iterator_traits requires the following types to be defined
        typedef typename device_dereferenceable_iterator_traits<const_pointer>::device_dereferenceable_type device_dereferenceable_pointer;

        // forward declaration
        class device_dereferenceable_type;

        // shorthand: name the base class of device_dereferenceable_type
        typedef experimental::iterator_adaptor<device_dereferenceable_type,
                                               device_dereferenceable_pointer,
                                               typename thrust::iterator_traits<device_dereferenceable_pointer>::value_type,
                                               typename thrust::iterator_traits<device_dereferenceable_pointer>::iterator_category,
                                               typename thrust::iterator_traits<device_dereferenceable_pointer>::reference,
                                               typename thrust::iterator_traits<device_dereferenceable_pointer>::pointer,
                                               typename thrust::iterator_traits<device_dereferenceable_pointer>::difference_type> device_dereferenceable_type_base;

        class device_dereferenceable_type
          : public device_dereferenceable_type_base
        {
          public:
            __host__ __device__
            device_dereferenceable_type() {}

            __host__ __device__
            device_dereferenceable_type(device_dereferenceable_pointer p)
              : device_dereferenceable_type_base(p) {}
        }; // end device_dereferenceable_type

      private:
        // befriend device_dereferenceable_iterator_traits so he can get at device_dereferenceable_type
        friend struct thrust::detail::device_dereferenceable_iterator_traits<const_iterator>;

        device_dereferenceable_type device_dereferenceable(void)
        {
          return device_dereferenceable_type(make_device_dereferenceable<const_pointer>::transform(this->base()));
        } // end device_dereferenceable()

        // befriend make_device_dereferenceable so he can get at device_dereferenceable()
        friend struct thrust::detail::make_device_dereferenceable<const_iterator>;
    }; // end iterator

    /*! This constructor creates an empty vector_base.
     */
    __host__
    vector_base(void);

    /*! This constructor creates a vector_base with copies
     *  of an exemplar element.
     *  \param n The number of elements to initially create.
     *  \param value An element to copy.
     */
    __host__
    explicit vector_base(size_type n, const value_type &value = value_type());

    /*! Copy constructor copies from an exemplar vector_base.
     *  \param v The vector_base to copy.
     */
    __host__
    vector_base(const vector_base &v);

    /*! assign operator makes a copy of an exemplar vector_base.
     *  \param v The vector_base to copy.
     */
    __host__
    vector_base &operator=(const vector_base &v);

    /*! Copy constructor copies from an exemplar vector_base with different
     *  type.
     *  \param v The vector_base to copy.
     */
    template<typename OtherT, typename OtherAlloc>
    __host__
    vector_base(const vector_base<OtherT, OtherAlloc> &v);

    /*! assign operator makes a copy of an exemplar vector_base with different
     *  type.
     *  \param v The vector_base to copy.
     */
    template<typename OtherT, typename OtherAlloc>
    __host__
    vector_base &operator=(const vector_base<OtherT,OtherAlloc> &v);

    /*! Copy constructor copies from an exemplar std::vector.
     *  \param v The std::vector to copy.
     *  XXX TODO: Make this method redundant with a properly templatized constructor.
     *            We would like to copy from a vector whose element type is anything
     *            assignable to value_type.
     */
    template<typename OtherT, typename OtherAlloc>
    __host__
    vector_base(const std::vector<OtherT, OtherAlloc> &v);

    /*! assign operator makes a copy of an exemplar std::vector.
     *  \param v The vector to copy.
     *  XXX TODO: Templatize this assign on the type of the vector to copy from.
     *            We would like to copy from a vector whose element type is anything
     *            assignable to value_type.
     */
    template<typename OtherT, typename OtherAlloc>
    __host__
    vector_base &operator=(const std::vector<OtherT,OtherAlloc> &v);

    /*! This constructor builds a vector_base from a range.
     *  \param begin The beginning of the range.
     *  \param end   The end of the range.
     */
    template<typename InputIterator>
    __host__
    vector_base(InputIterator begin, InputIterator end);

    /*! The destructor erases the elements.
     */
    __host__
    ~vector_base(void);

    /*! \brief Resizes this vector_base to the specified number of elements.
     *  \param new_size Number of elements this vector_base should contain.
     *  \param x Data with which new elements should be populated.
     *  \throw std::length_error If n exceeds max_size().
     *
     *  This method will resize this vector_base to the specified number of
     *  elements.  If the number is smaller than this vector_base's current
     *  size this vector_base is truncated, otherwise this vector_base is
     *  extended and new elements are populated with given data.
     */
    __host__
    void resize(size_type new_size, value_type x = value_type());

    /*! Returns the number of elements in this vector_base.
     */
    __host__ __device__
    size_type size(void) const;

    /*! Returns the size() of the largest possible vector_base.
     *  \return The largest possible return value of size().
     */
    __host__ __device__
    size_type max_size(void) const;

    /*! \brief If n is less than or equal to capacity(), this call has no effect.
     *         Otherwise, this method is a request for allocation of additional memory. If
     *         the request is successful, then capacity() is greater than or equal to
     *         n; otherwise, capacity() is unchanged. In either case, size() is unchanged.
     *  \throw std::length_error If n exceeds max_size().
     */
    __host__
    void reserve(size_type n);

    /*! Returns the number of elements which have been reserved in this
     *  vector_base.
     */
    __host__ __device__
    size_type capacity(void) const;

    /*! \brief Subscript access to the data contained in this vector_dev.
     *  \param n The index of the element for which data should be accessed.
     *  \return Read/write reference to data.
     *
     *  This operator allows for easy, array-style, data access.
     *  Note that data access with this operator is unchecked and
     *  out_of_range lookups are not defined.
     */
    __host__ __device__
    reference operator[](size_type n);

    /*! \brief Subscript read access to the data contained in this vector_dev.
     *  \param n The index of the element for which data should be accessed.
     *  \return Read reference to data.
     *
     *  This operator allows for easy, array-style, data access.
     *  Note that data access with this operator is unchecked and
     *  out_of_range lookups are not defined.
     */
    __host__ __device__
    const_reference operator[](size_type n) const;

    /*! This method returns an iterator pointing to the beginning of
     *  this vector_base.
     *  \return mStart
     */
    __host__ __device__
    iterator begin(void);

    /*! This method returns a const_iterator pointing to the beginning
     *  of this vector_base.
     *  \return mStart
     */
    __host__ __device__
    const_iterator begin(void) const;

    /*! This method returns a const_iterator pointing to the beginning
     *  of this vector_base.
     *  \return mStart
     */
    __host__ __device__
    const_iterator cbegin(void) const;

    /*! This method returns an iterator pointing to one element past the
     *  last of this vector_base.
     *  \return begin() + size().
     */
    __host__ __device__
    iterator end(void);

    /*! This method returns a const_iterator pointing to one element past the
     *  last of this vector_base.
     *  \return begin() + size().
     */
    __host__ __device__
    const_iterator end(void) const;

    /*! This method returns a const_iterator pointing to one element past the
     *  last of this vector_base.
     *  \return begin() + size().
     */
    __host__ __device__
    const_iterator cend(void) const;

    /*! This method returns a const_reference referring to the first element of this
     *  vector_base.
     *  \return The first element of this vector_base.
     */
    __host__ __device__
    const_reference front(void) const;

    /*! This method returns a reference pointing to the first element of this
     *  vector_base.
     *  \return The first element of this vector_base.
     */
    __host__ __device__
    reference front(void);

    /*! This method returns a const reference pointing to the last element of
     *  this vector_base.
     *  \return The last element of this vector_base.
     */
    __host__ __device__
    const_reference back(void) const;

    /*! This method returns a reference referring to the last element of
     *  this vector_dev.
     *  \return The last element of this vector_base.
     */
    __host__ __device__
    reference back(void);

    /*! This method resizes this vector_base to 0.
     */
    __host__
    void clear(void);

    /*! This method returns true iff size() == 0.
     *  \return true if size() == 0; false, otherwise.
     */
    __host__ __device__
    bool empty(void) const;

    /*! This method appends the given element to the end of this vector_base.
     *  \param x The element to append.
     */
    __host__
    void push_back(const value_type &x);

    /*! This method swaps the contents of this vector_base with another vector_base.
     *  \param v The vector_base with which to swap.
     */
    __host__ __device__
    void swap(vector_base &v);

    /*! This method removes the element at position pos.
     *  \param pos The position of the element of interest.
     *  \return An iterator pointing to the new location of the element that followed the element
     *          at position pos.
     */
    __host__
    iterator erase(iterator pos);

    /*! This method removes the range of elements [begin,end) from this vector_base.
     *  \param begin The beginning of the range of elements to remove.
     *  \param end The end of the range of elements to remove.
     *  \return An iterator pointing to the new location of the element that followed the last
     *          element in the sequence [begin,end).
     */
    __host__
    iterator erase(iterator begin, iterator end);

  protected:
    // An iterator pointing to the first element of this vector_base.
    iterator mBegin;

    // The size of this vector_base, in number of elements.
    size_type mSize;

    // The capacity of this vector_base, in number of elements.
    size_type mCapacity;

    // Our allocator
    allocator_type mAllocator;

  private:
    // these methods resolve the ambiguity of the constructor template of form (Iterator, Iterator)
    template<typename IteratorOrIntegralType>
      void init_dispatch(IteratorOrIntegralType begin, IteratorOrIntegralType end, false_type); 

    template<typename IteratorOrIntegralType>
      void init_dispatch(IteratorOrIntegralType n, IteratorOrIntegralType value, true_type); 
}; // end vector_base

} // end detail

/*! This function assigns the contents of vector a to vector b and the
 *  contents of vector b to vector a.
 *
 *  \param a The first vector of interest. After completion, the contents
 *           of b will be returned here.
 *  \param b The second vector of interest. After completion, the contents
 *           of a will be returned here.
 */
template<typename T, typename Alloc>
  void swap(detail::vector_base<T,Alloc> &a,
            detail::vector_base<T,Alloc> &b);

} // end thrust

#include <thrust/detail/vector_base.inl>

