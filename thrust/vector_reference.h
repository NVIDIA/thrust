/*
 *  Copyright 2008-2013 NVIDIA Corporation
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

/*! \file vector_reference.h
 *  \brief A reference to memory hold by a vector-lice container. 
 *         It allows to access the data from C++ and cuda code.
 */

#pragma once

#include <thrust/detail/config.h>
#include <thrust/swap.h>
#include <thrust/host_vector.h>
#include <stdexcept>
#include <vector>

namespace thrust
{

/*! A \p device_reference allows access to data in a vector e.g. 
 * \p device_vector, by holding a non owning pointer as well as the 
 * size. It implements all data access functions of a vector typically 
 * has, including iterators. It does not allow access to any of the 
 * vectors functions that might change the capacity, as it only references 
 * the internal data. It can be used in hist as well as device code.
 *
 *  \see make_vector_reference
 *  \see make_device_vector_reference
 */
template <typename T, typename pointerT = T*, typename refT = T&>
  class vector_reference
{
  public:
    // typedefs
    typedef T             value_type;
    typedef std::size_t   size_type;
    typedef pointerT      pointer;
    typedef refT          reference;
    typedef pointerT      iterator;

    // construction

    /*! Construct an empty \p vector_reference
     */
    __host__ __device__
    vector_reference()
      : m_data(nullptr), m_size(0) {}

    /*! Construct \p vector_reference that references the data specified by pointer to first element and length.
     *  \param data pointer to the first element of the array to be referenced
     *  \param size length of the array to be referenced
     */
    __host__ __device__
    vector_reference(pointer data, size_type size)
      : m_data(data), m_size(size) {}

    /*!
     * @brief implicitly convert to a reference to const
     */
    template <typename U=T, class = std::enable_if_t< !std::is_const<U>::value> >
    operator vector_reference<const T>() 
    {
      return vector_reference<const T>(m_data,m_size); 
    }

    // access data

    /*! \brief Read access with to the data contained in the vector, with bounds check.
     *  \param n The index of the element for which data should be accessed.
     *  \return Read reference to data.
     * 
     * In case of a bounds violation a message is printed using printf (on host as well as device).
     * If possible, an exception is thrown. If the violation happens inside a CUDA kernel, 
     * it will be halted and an "unspecified launch error" is raised. 
     */
    __host__ __device__
    reference at(size_type n)
    {
      if(n >= m_size) {
        printf("Tried to access index %lu, but is out of bounds. Size: %lu.", 
                (unsigned long)(n), (unsigned long)(m_size));
        #if defined(__CUDA_ARCH__)
          asm("trap;");
        #else
            throw std::out_of_range("vector_reference access out of range!");
        #endif
      }
      return m_data[n];
    }

    /*! \brief Subscript access to the data contained in this vector_dev.
     *  \param n The index of the element for which data should be accessed.
     *  \return Read/write reference to data.
     *
     *  This operator allows for easy, array-style, data access.
     *  Note that data access with this operator is unchecked and
     *  out_of_range lookups are not defined.
     */
    __host__ __device__
    reference operator[](size_type n) 
    {return m_data[n];}
    
    __host__ __device__
    reference front() 
    {return m_data[0];}
    
    /*! Access to the last element.
     *  \return Reference to the last element.
     */
    __host__ __device__
    reference back() 
    {return m_data[m_size-1];}
    
    /*! Allows direct access to internal data.
     *  \return Pointer to the internal data
     */
    __host__ __device__
    pointer data() { return m_data;} //!< direct data access
  
    // iteration

    /*! This method returns an iterator pointing to the beginning
     *  of this vector_base.
     *  \return Iterator to the first element of the vector.
     */
    __host__ __device__ 
    iterator begin() { return m_data;} //!< get an iterator to the beginning
    
    /*! This method returns an iterator pointing to one element past the last.
     *  \return begin() + size().
     */ 
    __host__ __device__ 
    iterator end() 
    { return m_data+size();}

    // check size

    /*! Returns the number of elements in the vector.
     * \return  the number of elements of stored in the vector
     */
    __host__ __device__
    size_type size() const 
    {return m_size;}
    
    /*! Checks if the vector is empty.
     *  \return true if the vector has no elements, false otherwise.
     */
    __host__ __device__
    bool empty() const 
    {return m_size<=0;}

    // other

    /*! This method swaps the contents of this vector_base with another vector_base.
     *  \param v The vector_base with which to swap.
     */
    __host__ __device__
    void swap(vector_reference &v)
    { 
      thrust::swap( this->m_data,v.m_data);
      thrust::swap( this->m_size,v.m_size);
    }

  private:
    pointer m_data;
    size_type m_size;
};

/*! Exchanges the values of two vectors_references.
 *  \p x The first \p vector_reference of interest.
 *  \p y The second \p vector_reference of interest.
 */
template<typename T>
  void swap(vector_reference<T> &a, vector_reference<T> &b)
{
  a.swap(b);
}

/*!
 * @brief Creates a vector reference for usage in host code from any type of vector.
 * 
 * @tparam T The type of vector to create a reference to.
 * @param v The vector to create a reference to
 * @return auto a \p vector_reference, that references v and can be used in host code
 */
template <typename T>
auto make_vector_reference(T& v)
{
  return vector_reference<typename T::value_type, typename T::pointer, typename T::reference>(v.data(),v.size());
}

/*!
 * @brief Creates a vector reference to const for usage in host code from any type of const vector.
 * 
 * @tparam T The type of const vector to create a reference to.
 * @param v The vector to create a reference to
 * @return auto a \p vector_reference, that references v and can be used in host code
 */
template <typename T>
auto make_vector_reference(const T& v)
{
  return vector_reference<const typename T::value_type, typename T::const_pointer, const typename T::const_reference>(v.data(),v.size());
}

/*!
 * @brief Creates a vector reference for usage in device code from vectors that store device data, like \p device_vector.
 * 
 * @tparam T The type of vector to create a reference to.
 * @param v The vector to create a reference to
 * @return auto a \p vector_reference, that references v and can be used in host code
 */
template <typename T>
auto make_device_vector_reference(T& v)
{
  return vector_reference<typename T::value_type>(thrust::raw_pointer_cast(v.data()),v.size());
}

/*!
 * @brief Creates a vector reference to const for usage in device code from vectors that store device data, like \p device_vector.
 * 
 * @tparam T The type of const vector to create a reference to.
 * @param v The vector to create a reference to
 * @return auto a \p vector_reference, that references v and can be used in host code
 */
template <typename T>
auto make_device_vector_reference(const T& v)
{
  return vector_reference<const typename T::value_type>(thrust::raw_pointer_cast(v.data()),v.size());
}

} // namespace thrust