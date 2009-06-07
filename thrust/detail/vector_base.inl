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

// define our own max() function rather than #include <thrust/extrema.h>
template<typename T>
  T vector_base_max(const T &lhs, const T &rhs)
{
  return lhs > rhs ? lhs : rhs;
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

template<typename T, typename Alloc>
  void vector_base<T,Alloc>
    ::insert(iterator position, size_type n, const T &x)
{
  fill_insert(position, n, x);
} // end vector_base::insert()

template<typename T, typename Alloc>
  template<typename InputIterator>
    void vector_base<T,Alloc>
      ::insert(iterator position, InputIterator first, InputIterator last)
{
  // we could have received insert(position, n, x), so disambiguate on the
  // type of InputIterator
  typedef typename thrust::detail::is_integral<InputIterator> integral;

  insert_dispatch(position, first, last, integral());
} // end vector_base::insert()

template<typename T, typename Alloc>
  template<typename InputIterator>
    void vector_base<T,Alloc>
      ::insert_dispatch(iterator position, InputIterator first, InputIterator last, false_type)
{
  range_insert(position, first, last);
} // end vector_base::insert_dispatch()

template<typename T, typename Alloc>
  template<typename Integral>
    void vector_base<T,Alloc>
      ::insert_dispatch(iterator position, Integral n, Integral x, true_type)
{
  fill_insert(position, n, x);
} // end vector_base::insert_dispatch()

// XXX this needs to be moved out into thrust:: and dispatched properly
template<typename InputIterator, typename Distance>
  void advance(InputIterator &i, Distance n)
{
  i += n;
} // end advance()

template<typename T, typename Alloc>
  template<typename ForwardIterator>
    void vector_base<T,Alloc>
      ::range_insert(iterator position,
                     ForwardIterator first,
                     ForwardIterator last)
{
  if(first != last)
  {
    // how many new elements will we create?
    const size_type num_new_elements = thrust::distance(first, last);
    if(capacity() - size() >= num_new_elements)
    {
      // we've got room for all of them
      // how many existing elements will we displace?
      const size_type num_displaced_elements = end() - position;
      iterator old_end = end();

      if(num_displaced_elements > num_new_elements)
      {
        // construct copy n displaced elements to new elements
        // following the insertion
        thrust::uninitialized_copy(end() - num_new_elements, end(), end());

        // extend the size
        mSize += num_new_elements;

        // copy num_displaced_elements - num_new_elements elements to existing elements

        // XXX SGI's version calls copy_backward here for some reason
        //     maybe it's just more readable
        // copy_backward(position, old_end - num_new_elements, old_end);
        const size_type copy_length = (old_end - num_new_elements) - position;
        thrust::copy(position, old_end - num_new_elements, old_end - copy_length);

        // finally, copy the range to the insertion point
        thrust::copy(first, last, position);
      } // end if
      else
      {
        ForwardIterator mid = first;
        thrust::detail::advance(mid, num_displaced_elements);

        // construct copy new elements at the end of the vector
        thrust::uninitialized_copy(mid, last, end());

        // extend the size
        mSize += num_new_elements - num_displaced_elements;

        // construct copy the displaced elements
        thrust::uninitialized_copy(position, old_end, end());

        // extend the size
        mSize += num_displaced_elements;

        // copy to elements which already existed
        thrust::copy(first, mid, position);
      } // end else
    } // end if
    else
    {
      const size_type old_size = size();

      // compute the new capacity after the allocation
      size_type new_capacity = old_size + vector_base_max(old_size, num_new_elements);

      // allocate exponentially larger new storage
      new_capacity = std::max<size_type>(new_capacity, 2 * capacity());

      // do not exceed maximum storage
      new_capacity = std::min<size_type>(new_capacity, max_size());

      if(new_capacity > max_size())
      {
        throw std::length_error("insert(): insertion exceeds max_size().");
      } // end if

      iterator new_begin = mAllocator.allocate(new_capacity);
      iterator new_end = new_begin;

      try
      {
        // construct copy elements before the insertion to the beginning of the newly
        // allocated storage
        new_end = thrust::uninitialized_copy(begin(), position, new_begin);

        // construct copy elements to insert
        new_end = thrust::uninitialized_copy(first, last, new_end);

        // construct copy displaced elements from the old storage to the new storage
        // remember [position, end()) refers to the old storage
        new_end = thrust::uninitialized_copy(position, end(), new_end);
      } // end try
      catch(...)
      {
        // something went wrong, so destroy & deallocate the new storage 
        thrust::detail::destroy(new_begin, new_end);
        mAllocator.deallocate(&*new_begin, new_capacity);

        // rethrow
        throw;
      } // end catch

      // call destructors on the elements in the old storage
      thrust::detail::destroy(begin(), end());

      // deallocate the old storage
      mAllocator.deallocate(&*begin(), capacity());
  
      // record the vector's new parameters
      mBegin    = new_begin;
      mSize     = old_size + num_new_elements;
      mCapacity = new_capacity;
    } // end else
  } // end if
} // end vector_base::range_insert()

template<typename T, typename Alloc>
  void vector_base<T,Alloc>
    ::fill_insert(iterator position, size_type n, const T &x)
{
  if(n != 0)
  {
    if(capacity() - size() >= n)
    {
      // we've got room for all of them
      // how many existing elements will we displace?
      const size_type num_displaced_elements = end() - position;
      iterator old_end = end();

      if(num_displaced_elements > n)
      {
        // construct copy n displaced elements to new elements
        // following the insertion
        thrust::uninitialized_copy(end() - n, end(), end());

        // extend the size
        mSize += n;

        // copy num_displaced_elements - n elements to existing elements

        // XXX SGI's version calls copy_backward here for some reason
        //     maybe it's just more readable
        // copy_backward(position, old_end - n, old_end);
        const size_type copy_length = (old_end - n) - position;
        thrust::copy(position, old_end - n, old_end - copy_length);

        // finally, fill the range to the insertion point
        thrust::fill(position, position + n, x);
      } // end if
      else
      {
        // construct new elements at the end of the vector
        // XXX SGI's version uses uninitialized_fill_n here
        thrust::uninitialized_fill(end(),
                                   end() + (n - num_displaced_elements),
                                   x);

        // extend the size
        mSize += n - num_displaced_elements;

        // construct copy the displaced elements
        thrust::uninitialized_copy(position, old_end, end());

        // extend the size
        mSize += num_displaced_elements;

        // fill to elements which already existed
        thrust::fill(position, old_end, x);
      } // end else
    } // end if
    else
    {
      const size_type old_size = size();

      // compute the new capacity after the allocation
      size_type new_capacity = old_size + vector_base_max(old_size, n);

      // allocate exponentially larger new storage
      new_capacity = std::max<size_type>(new_capacity, 2 * capacity());

      // do not exceed maximum storage
      new_capacity = std::min<size_type>(new_capacity, max_size());

      if(new_capacity > max_size())
      {
        throw std::length_error("insert(): insertion exceeds max_size().");
      } // end if

      iterator new_begin = mAllocator.allocate(new_capacity);
      iterator new_end = new_begin;

      try
      {
        // construct copy elements before the insertion to the beginning of the newly
        // allocated storage
        new_end = thrust::uninitialized_copy(begin(), position, new_begin);

        // construct new elements to insert
        thrust::uninitialized_fill(new_end, new_end + n, x);
        new_end += n;

        // construct copy displaced elements from the old storage to the new storage
        // remember [position, end()) refers to the old storage
        new_end = thrust::uninitialized_copy(position, end(), new_end);
      } // end try
      catch(...)
      {
        // something went wrong, so destroy & deallocate the new storage 
        thrust::detail::destroy(new_begin, new_end);
        mAllocator.deallocate(&*new_begin, new_capacity);

        // rethrow
        throw;
      } // end catch

      // call destructors on the elements in the old storage
      thrust::detail::destroy(begin(), end());

      // deallocate the old storage
      mAllocator.deallocate(&*begin(), capacity());
  
      // record the vector's new parameters
      mBegin    = new_begin;
      mSize     = old_size + n;
      mCapacity = new_capacity;
    } // end else
  } // end if
} // end vector_base::fill_insert()

} // end detail

template<typename T, typename Alloc>
  void swap(detail::vector_base<T,Alloc> &a,
            detail::vector_base<T,Alloc> &b)
{
  a.swap(b);
} // end swap()

} // end thrust

