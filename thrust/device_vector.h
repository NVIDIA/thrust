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


/*! \file device_vector.h
 *  \brief Defines the interface to a std::vector-like
 *         class for device memory management.
 */

#pragma once

#include <thrust/detail/config.h>
#include <thrust/device_allocator.h>
#include <thrust/detail/vector_base.h>
#include <vector>

namespace thrust
{

// forward declaration of host_vector
template<typename T, typename Alloc> class host_vector;

/*! \addtogroup container_classes Container Classes
 *  \addtogroup device_containers Device Containers
 *  \ingroup container_classes
 *  \{
 */

/*! A \p device_vector is a Device Sequence that supports random access to elements,
 *  constant time removal of elements at the end, and linear time insertion
 *  and removal of elements at the beginning or in the middle. The number of
 *  elements in a \p device_vector may vary dynamically; memory management is
 *  automatic. The memory associated with a \p device_vector resides in the memory
 *  space of a parallel device.
 *
 *  Please refer to the <a href="http://www.sgi.com/tech/stl/Vector.html">C++ STL</a>
 *  for the documentation of \p device_vector's API.
 *
 *  \see http://www.sgi.com/tech/stl/Vector.html
 *  \see host_vector
 *
 *  \bug The following members do not exist yet:
 *       \p reverse_iterator, \p const_reverse_iterator, \p rbegin, \p rend,
 *       \p pop_back, \p insert, \p operator<.
 *
 */
template<typename T, typename Alloc = thrust::device_allocator<T> >
  class device_vector
    : public detail::vector_base<T,Alloc>
{
  private:
    typedef detail::vector_base<T,Alloc> Parent;

  public:
    // typedefs
    typedef typename Parent::size_type  size_type;
    typedef typename Parent::value_type value_type;

    /*! This constructor creates an empty \p device_vector.
     */
    __host__
    device_vector(void)
      :Parent() {}

    /*! This constructor creates a \p device_vector with copies
     *  of an exemplar element.
     *  \param n The number of elements to initially create.
     *  \param value An element to copy.
     */
    __host__
    explicit device_vector(size_type n, const value_type &value = value_type())
      :Parent(n,value) {}

    /*! Copy constructor copies from an exemplar \p device_vector.
     *  \param v The \p device_vector to copy.
     */
    __host__
    device_vector(const device_vector &v)
      :Parent(v) {}

    /*! Copy constructor copies from an exemplar \p device_vector with different type.
     *  \param v The \p device_vector to copy.
     */
    template<typename OtherT, typename OtherAlloc>
    __device__
    device_vector(const device_vector<OtherT,OtherAlloc> &v)
      :Parent(v) {}

    /*! Assign operator copies from an exemplar \p device_vector with different type.
     *  \param v The \p device_vector to copy.
     */
    template<typename OtherT, typename OtherAlloc>
    __device__
    device_vector &operator=(const device_vector<OtherT,OtherAlloc> &v)
    { Parent::operator=(v); return *this; }

    /*! Copy constructor copies from an exemplar \c std::vector.
     *  \param v The <tt>std::vector</tt> to copy.
     */
    template<typename OtherT, typename OtherAlloc>
    __host__
    device_vector(const std::vector<OtherT,OtherAlloc> &v)
      :Parent(v) {}

    /*! Assign operator copies from an exemplar <tt>std::vector</tt>.
     *  \param v The <tt>std::vector</tt> to copy.
     */
    template<typename OtherT, typename OtherAlloc>
    __host__
    device_vector &operator=(const std::vector<OtherT,OtherAlloc> &v)
    { Parent::operator=(v); return *this;}

    /*! Copy constructor copies from an exemplar \p host_vector with possibly different type.
     *  \param v The \p host_vector to copy.
     */
    template<typename OtherT, typename OtherAlloc>
    __host__
    device_vector(const host_vector<OtherT,OtherAlloc> &v);

    /*! This constructor builds a \p device_vector from a range.
     *  \param begin The beginning of the range.
     *  \param end   The end of the range.
     */
    template<typename InputIterator>
    __host__
    device_vector(InputIterator begin, InputIterator end)
      :Parent(begin,end) {}
}; // end device_vector

/*! This operator allows comparison between two \p device_vectors.
 *  \param lhs The first \p device_vector to compare.
 *  \param rhs The second \p device_vector to compare.
 *  \return \c true if and only if each corresponding element in either
 *          \p device_vector equals the other; \c false, otherwise.
 */
template<typename T1, typename Alloc1,
         typename T2, typename Alloc2>
  bool operator==(const device_vector<T1,Alloc1> &lhs,
                  const device_vector<T2,Alloc2> &rhs);

/*! This operator allows comparison between a \p host_vector & a \p device_vector.
 *  \param lhs The \p host_vector to compare.
 *  \param rhs The \p device_vector to compare.
 *  \return \c true if and only if each corresponding element in either
 *          vector equals the other; \c false, otherwise.
 */
template<typename T1, typename Alloc1,
         typename T2, typename Alloc2>
  bool operator==(const host_vector<T1,Alloc1> &lhs,
                  const device_vector<T2,Alloc2> &rhs);

/*! This operator allows comparison between a \p device_vector & a \p host_vector.
 *  \param lhs The \p device_vector to compare.
 *  \param rhs The \p host_vector to compare.
 *  \return \c true if and only if each corresponding element in either
 *          vector equals the other; \c false, otherwise.
 */
template<typename T1, typename Alloc1,
         typename T2, typename Alloc2>
  bool operator==(const device_vector<T1,Alloc1> &lhs,
                  const host_vector<T2,Alloc2> &rhs);

/*! This operator allows comparison between a \c std::vector & a \p device_vector.
 *  \param lhs The \c std::vector to compare.
 *  \param rhs The \p device_vector to compare.
 *  \return \c true if and only if each corresponding element in either
 *          vector equals the other; \c false, otherwise.
 */
template<typename T1, typename Alloc1,
         typename T2, typename Alloc2>
  bool operator==(const std::vector<T1,Alloc1> &lhs,
                  const device_vector<T2,Alloc2> &rhs);

/*! This operator allows comparison between a \c device_vector & a \p std::vector.
 *  \param lhs The \p device_vector to compare.
 *  \param rhs The \c std::vector to compare.
 *  \return \c true if and only if each corresponding element in either
 *          vector equals the other; \c false, otherwise.
 */
template<typename T1, typename Alloc1,
         typename T2, typename Alloc2>
  bool operator==(const device_vector<T1,Alloc1> &lhs,
                  const std::vector<T2,Alloc2> &rhs);

/*! \}
 */

} // end thrust

#include <thrust/detail/device_vector.inl>


