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


/*! \file vector_base.inl
 *  \brief Inline file for vector_base.h.
 */

#include <thrust/detail/vector_base.h>
#include <thrust/copy.h>
#include <thrust/uninitialized_fill.h>
#include <thrust/uninitialized_copy.h>
#include <thrust/distance.h>
#include <thrust/detail/destroy.h>
#include <stdexcept>

namespace thrust
{

namespace detail
{

// define our own min() function rather than #include <thrust/extrema.h>
template<typename T>
  T vector_base_min(const T &lhs, const T &rhs)
{
  return lhs < rhs ? lhs : rhs;
} // end vector_base_min()

template<typename T, typename Alloc>
  vector_base<T,Alloc>
    ::vector_base(void)
      :mBegin(pointer(static_cast<T*>(0))),
       mSize(0),
       mCapacity(0)
{
  ;
} // end vector_base::vector_base()

template<typename T, typename Alloc>
  vector_base<T,Alloc>
    ::vector_base(size_type n, const value_type &value)
      :mBegin(pointer(static_cast<T*>(0))),
       mSize(0),
       mCapacity(0)
{
  resize(n,value);
} // end vector_base::vector_base()

template<typename T, typename Alloc>
  vector_base<T,Alloc>
    ::vector_base(const vector_base &v)
      :mBegin(pointer(static_cast<T*>(0))),
       mSize(0),
       mCapacity(0),
       mAllocator(v.mAllocator)
{
  reserve(v.size());
  thrust::uninitialized_copy(v.begin(), v.end(), begin());
  mSize = v.size();
} // end vector_base::vector_base()

template<typename T, typename Alloc>
  vector_base<T,Alloc> &
    vector_base<T,Alloc>
      ::operator=(const vector_base &v)
{
  // copy elements already allocated
  thrust::copy(v.begin(),
                v.begin() + vector_base_min(v.size(), size()),
                begin());

  // XXX this will do a redundant copy of elements we don't
  //     wish to keep
  reserve(v.size());

  // copy construct uninitialized elements
  difference_type num_remaining = v.size() - size();

  if(num_remaining > 0)
  {
    thrust::uninitialized_copy(v.begin() + size(),
                                v.end(),
                                begin() + size());
  } // end if
  else
  {
    // destroy extra elements at the end of our range
    thrust::detail::destroy(begin() + v.size(), end());
  } // end if

  mSize = v.size();

  return *this;
} // end vector_base::operator=()

template<typename T, typename Alloc>
  template<typename OtherT, typename OtherAlloc>
    vector_base<T,Alloc>
      ::vector_base(const vector_base<OtherT,OtherAlloc> &v)
        :mBegin(pointer(static_cast<T*>(0))),
         mSize(0),
         mCapacity(0)
{
  reserve(v.size());
  thrust::uninitialized_copy(v.begin(), v.end(), begin());
  mSize = v.size();
} // end vector_base::vector_base()

template<typename T, typename Alloc>
  template<typename OtherT, typename OtherAlloc>
    vector_base<T,Alloc> &
      vector_base<T,Alloc>
        ::operator=(const vector_base<OtherT,OtherAlloc> &v)
{
  // copy elements already allocated
  thrust::copy(v.begin(),
                v.begin() + vector_base_min(v.size(), size()),
                begin());

  // XXX this will do a redundant copy of elements we don't
  //     wish to keep
  reserve(v.size());

  // copy construct uninitialized elements
  difference_type num_remaining = v.size() - size();

  if(num_remaining > 0)
  {
    thrust::uninitialized_copy(v.begin() + size(),
                                v.end(),
                                begin() + size());
  } // end if
  else
  {
    // destroy extra elements at the end of our range
    thrust::detail::destroy(begin() + v.size(), end());
  } // end if

  mSize = v.size();

  return *this;
} // end vector_base::operator=()

template<typename T, typename Alloc>
  template<typename OtherT, typename OtherAlloc>
    vector_base<T,Alloc>
      ::vector_base(const std::vector<OtherT,OtherAlloc> &v)
        :mBegin(pointer(static_cast<T*>(0))),
         mSize(0),
         mCapacity(0)
{
  reserve(v.size());
  thrust::uninitialized_copy(v.begin(), v.end(), begin());
  mSize = v.size();
} // end vector_base::vector_base()

template<typename T, typename Alloc>
  template<typename OtherT, typename OtherAlloc>
    vector_base<T,Alloc> &
      vector_base<T,Alloc>
        ::operator=(const std::vector<OtherT,OtherAlloc> &v)
{
  // copy elements already allocated
  thrust::copy(v.begin(),
                v.begin() + vector_base_min(v.size(), size()),
                begin());

  // XXX this will do a redundant copy of elements we don't
  //     wish to keep
  reserve(v.size());

  // copy construct uninitialized elements
  difference_type num_remaining = v.size() - size();

  if(num_remaining > 0)
  {
    thrust::uninitialized_copy(v.begin() + size(),
                                v.end(),
                                begin() + size());
  } // end if
  else
  {
    // destroy extra elements at the end of our range
    thrust::detail::destroy(begin() + v.size(), end());
  } // end if

  mSize = v.size();

  return *this;
} // end vector_base::operator=()

template<typename T, typename Alloc>
  template<typename IteratorOrIntegralType>
    void vector_base<T,Alloc>
      ::init_dispatch(IteratorOrIntegralType n,
                      IteratorOrIntegralType value,
                      true_type)
{
  resize(n,value);
} // end vector_base::init_dispatch()

template<typename T, typename Alloc>
  template<typename IteratorOrIntegralType>
    void vector_base<T,Alloc>
      ::init_dispatch(IteratorOrIntegralType begin,
                      IteratorOrIntegralType end,
                      false_type)
{
  resize(thrust::distance(begin, end));

  thrust::copy(begin, end, this->begin());
} // end vector_base::init_dispatch()

template<typename T, typename Alloc>
  template<typename InputIterator>
    vector_base<T,Alloc>
      ::vector_base(InputIterator begin,
                                InputIterator end)
        :mBegin(pointer(static_cast<T*>(0))),
         mSize(0),
         mCapacity(0)
{
  // check the type of InputIterator: if its an integral type,
  // we need to interpret this call as (size_type, value_type)
  typedef thrust::detail::is_integral<InputIterator> Integer;

  init_dispatch(begin, end, Integer());
} // end vector_basee::vector_base()

template<typename T, typename Alloc>
  void vector_base<T,Alloc>
    ::resize(size_type new_size, value_type x)
{
  // reserve storage
  reserve(new_size);
  
  if(capacity() >= new_size)
  {
    if(new_size > size())
    {
      // fill new elements at the end of the new array with the exemplar
      thrust::uninitialized_fill(begin() + size(), begin() + new_size, x);
    } // end if
    else if(new_size < size())
    {
      // destroy elements at the end of the new range
      thrust::detail::destroy(begin() + new_size, end());
    } // end else if
    
    mSize = new_size;
  } // end if
} // end vector_base::resize()

template<typename T, typename Alloc>
  typename vector_base<T,Alloc>::size_type
    vector_base<T,Alloc>
      ::size(void) const
{
  return mSize;
} // end vector_base::size()

template<typename T, typename Alloc>
  typename vector_base<T,Alloc>::size_type
    vector_base<T,Alloc>
      ::max_size(void) const
{
  return mAllocator.max_size();
} // end vector_base::max_size()

template<typename T, typename Alloc>
  void vector_base<T,Alloc>
    ::reserve(size_type n)
{
  if(n > capacity())
  {
    if(n > max_size())
    {
      throw std::length_error("reserve(): n exceeds max_size().");
    } // end if

    // allocate exponentially larger new storage
    size_t new_capacity = std::max<size_type>(n, 2 * capacity());

    // do not exceed maximum storage
    new_capacity = std::min<size_type>(new_capacity, max_size());

    iterator new_array(pointer(static_cast<T*>(0)));
    
    // allocate the new storage
    new_array = mAllocator.allocate(new_capacity);

    if(new_array != iterator(pointer(static_cast<T*>(0))))
    {
      // deal with the old storage
      if(size() > 0)
      {
        // copy original elements to the front of the new array
        thrust::copy(begin(), end(), new_array);

        // free the old storage
        mAllocator.deallocate(&*begin(), capacity());
      } // end if

      mBegin = new_array;
      mCapacity = new_capacity;
    } // end if
  } // end if
} // end vector_base::reserve()

template<typename T, typename Alloc>
  typename vector_base<T,Alloc>::size_type
    vector_base<T,Alloc>
      ::capacity(void) const
{
  return mCapacity;
} // end vector_base::capacity()

template<typename T, typename Alloc>
  typename vector_base<T,Alloc>::reference
    vector_base<T,Alloc>
      ::operator[](const size_type n)
{
  return *(begin() + n);
} // end vector_base::operator[]

template<typename T, typename Alloc>
  typename vector_base<T,Alloc>::const_reference 
    vector_base<T,Alloc>
      ::operator[](const size_type n) const
{
  return *(begin() + n);
} // end vector_base::operator[]

template<typename T, typename Alloc>
  typename vector_base<T,Alloc>::iterator
    vector_base<T,Alloc>
      ::begin(void)
{
  return mBegin;
} // end vector_base::begin()

template<typename T, typename Alloc>
  typename vector_base<T,Alloc>::const_iterator
    vector_base<T,Alloc>
      ::begin(void) const
{
  return mBegin;
} // end vector_base::begin()

template<typename T, typename Alloc>
  typename vector_base<T,Alloc>::const_iterator
    vector_base<T,Alloc>
      ::cbegin(void) const
{
  return begin();
} // end vector_base::cbegin()

template<typename T, typename Alloc>
  typename vector_base<T,Alloc>::iterator
    vector_base<T,Alloc>
      ::end(void)
{
  return begin() + size();
} // end vector_base::end()

template<typename T, typename Alloc>
  typename vector_base<T,Alloc>::const_iterator
    vector_base<T,Alloc>
      ::end(void) const
{
  return begin() + size();
} // end vector_base::end()

template<typename T, typename Alloc>
  typename vector_base<T,Alloc>::const_iterator
    vector_base<T,Alloc>
      ::cend(void) const
{
  return end();
} // end vector_base::cend()

template<typename T, typename Alloc>
  typename vector_base<T,Alloc>::const_reference
    vector_base<T,Alloc>
      ::front(void) const
{
  return *begin();
} // end vector_base::front()

template<typename T, typename Alloc>
  typename vector_base<T,Alloc>::reference
    vector_base<T,Alloc>
      ::front(void)
{
  return *begin();
} // end vector_base::front()

template<typename T, typename Alloc>
  typename vector_base<T,Alloc>::const_reference
    vector_base<T,Alloc>
      ::back(void) const
{
  return *(begin() + static_cast<difference_type>(size() - 1));
} // end vector_base::vector_base

template<typename T, typename Alloc>
  typename vector_base<T,Alloc>::reference
    vector_base<T,Alloc>
      ::back(void)
{
  return *(begin() + static_cast<difference_type>(size() - 1));
} // end vector_base::vector_base

template<typename T, typename Alloc>
  vector_base<T,Alloc>
    ::~vector_base(void)
{
  clear();
  mAllocator.deallocate(&*begin(), capacity());
  mCapacity = 0;
  mBegin = iterator(pointer(static_cast<T*>(0)));
} // end vector_base::~vector_base()

template<typename T, typename Alloc>
  void vector_base<T,Alloc>
    ::clear(void)
{
  resize(0);
} // end vector_base::~vector_dev()

template<typename T, typename Alloc>
  bool vector_base<T,Alloc>
    ::empty(void) const
{
  return size() == 0;
} // end vector_base::empty();

template<typename T, typename Alloc>
  void vector_base<T,Alloc>
    ::push_back(const value_type &x)
{
  resize(size() + 1, x);
} // end vector_base::push_back()

template<typename T, typename Alloc>
  typename vector_base<T,Alloc>::iterator vector_base<T,Alloc>
    ::erase(iterator pos)
{
  return erase(pos,pos+1);
} // end vector_base::erase()

template<typename T, typename Alloc>
  typename vector_base<T,Alloc>::iterator vector_base<T,Alloc>
    ::erase(iterator begin, iterator end)
{
  // count the number of elements to erase
  size_type num_erased = end - begin;

  // copy the range [end,end()) to begin
  thrust::copy(end, this->end(),begin);

  // find the new index of the position following the erased range
  size_type new_index_following_erase = begin - this->begin();

  // resize ourself: destructors get called inside resize()
  resize(size() - num_erased);

  // return an iterator pointing to the position of the first element
  // following the erased range
  return this->begin() + new_index_following_erase;
} // end vector_base::erase()

template<typename T, typename Alloc>
  void vector_base<T,Alloc>
    ::swap(vector_base &v)
{
  thrust::swap(mBegin,     v.mBegin);
  thrust::swap(mSize,      v.mSize);
  thrust::swap(mCapacity,  v.mCapacity);
  thrust::swap(mAllocator, v.mAllocator);
} // end vector_base::swap()

} // end detail

template<typename T, typename Alloc>
  void swap(detail::vector_base<T,Alloc> &a,
            detail::vector_base<T,Alloc> &b)
{
  a.swap(b);
} // end swap()

} // end thrust

