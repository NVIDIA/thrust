/*
 *  Copyright 2008-2010 NVIDIA Corporation
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
#include <thrust/detail/move.h>
#include <thrust/equal.h>
#include <thrust/uninitialized_fill.h>
#include <thrust/uninitialized_copy.h>
#include <thrust/distance.h>
#include <thrust/advance.h>
#include <thrust/detail/destroy.h>
#include <thrust/detail/type_traits.h>

#include <algorithm>
#include <stdexcept>

#include <thrust/distance.h>
#include <thrust/iterator/iterator_traits.h>

#include <thrust/detail/raw_buffer.h>

namespace thrust
{

namespace detail
{

template<typename T, typename Alloc>
  vector_base<T,Alloc>
    ::vector_base(void)
      :mBegin(pointer(static_cast<T*>(0))),
       mSize(0),
       mCapacity(0),
       mAllocator()
{
  ;
} // end vector_base::vector_base()

template<typename T, typename Alloc>
  vector_base<T,Alloc>
    ::vector_base(size_type n, const value_type &value)
      :mBegin(pointer(static_cast<T*>(0))),
       mSize(0),
       mCapacity(0),
       mAllocator()
{
  fill_init(n,value);
} // end vector_base::vector_base()

template<typename T, typename Alloc>
  vector_base<T,Alloc>
    ::vector_base(const vector_base &v)
      :mBegin(pointer(static_cast<T*>(0))),
       mSize(0),
       mCapacity(0),
       mAllocator(v.mAllocator)
{
  // vector_base's iterator is not strictly InputHostIterator,
  // so dispatch with false_type
  range_init(v.begin(), v.end(), false_type());
} // end vector_base::vector_base()

template<typename T, typename Alloc>
  vector_base<T,Alloc> &
    vector_base<T,Alloc>
      ::operator=(const vector_base &v)
{
  if(this != &v)
  {
    assign(v.begin(), v.end());
  } // end if

  return *this;
} // end vector_base::operator=()

template<typename T, typename Alloc>
  template<typename OtherT, typename OtherAlloc>
    vector_base<T,Alloc>
      ::vector_base(const vector_base<OtherT,OtherAlloc> &v)
        :mBegin(pointer(static_cast<T*>(0))),
         mSize(0),
         mCapacity(0),
         mAllocator()
{
  // vector_base's iterator is not strictly InputHostIterator,
  // so dispatch with false_type
  range_init(v.begin(), v.end(), false_type());
} // end vector_base::vector_base()

template<typename T, typename Alloc>
  template<typename OtherT, typename OtherAlloc>
    vector_base<T,Alloc> &
      vector_base<T,Alloc>
        ::operator=(const vector_base<OtherT,OtherAlloc> &v)
{
  assign(v.begin(), v.end());

  return *this;
} // end vector_base::operator=()

template<typename T, typename Alloc>
  template<typename OtherT, typename OtherAlloc>
    vector_base<T,Alloc>
      ::vector_base(const std::vector<OtherT,OtherAlloc> &v)
        :mBegin(pointer(static_cast<T*>(0))),
         mSize(0),
         mCapacity(0),
         mAllocator()
{
  // std::vector's iterator is not strictly InputHostIterator,
  // so dispatch with false_type
  range_init(v.begin(), v.end(), false_type());
} // end vector_base::vector_base()

template<typename T, typename Alloc>
  template<typename OtherT, typename OtherAlloc>
    vector_base<T,Alloc> &
      vector_base<T,Alloc>
        ::operator=(const std::vector<OtherT,OtherAlloc> &v)
{
  assign(v.begin(), v.end());

  return *this;
} // end vector_base::operator=()

template<typename T, typename Alloc>
  template<typename IteratorOrIntegralType>
    void vector_base<T,Alloc>
      ::init_dispatch(IteratorOrIntegralType n,
                      IteratorOrIntegralType value,
                      true_type)
{
  fill_init(n,value);
} // end vector_base::init_dispatch()

template<typename T, typename Alloc>
  void vector_base<T,Alloc>
    ::fill_init(size_type n, const T &x)
{
  if(n > 0)
  {
    mBegin = mAllocator.allocate(n);
    mSize = mCapacity = n;

    thrust::uninitialized_fill(begin(), end(), x);
  } // end if
} // end vector_base::fill_init()

template<typename T, typename Alloc>
  template<typename InputIterator>
    void vector_base<T,Alloc>
      ::init_dispatch(InputIterator first,
                      InputIterator last,
                      false_type)
{
  // dispatch based on whether or not InputIterator
  // is strictly an InputHostIterator
  typedef typename thrust::iterator_traits<InputIterator>::iterator_category category;
  typedef typename thrust::detail::is_same<category, thrust::input_host_iterator_tag>::type input_host_iterator_or_not;
  range_init(first, last, input_host_iterator_or_not());
} // end vector_base::init_dispatch()

template<typename T, typename Alloc>
  template<typename InputHostIterator>
    void vector_base<T,Alloc>
      ::range_init(InputHostIterator first,
                   InputHostIterator last,
                   true_type)
{
  for(; first != last; ++first)
    push_back(*first);
} // end vector_base::range_init()

template<typename T, typename Alloc>
  template<typename ForwardIterator>
    void vector_base<T,Alloc>
      ::range_init(ForwardIterator first,
                   ForwardIterator last,
                   false_type)
{
  size_type new_size = thrust::distance(first, last);
  size_type new_capacity;
  iterator new_begin;

  allocate_and_copy(new_size, first, last, new_capacity, new_begin);

  mBegin    = new_begin;
  mSize     = new_size;
  mCapacity = new_capacity;
} // end vector_base::range_init()

template<typename T, typename Alloc>
  template<typename InputIterator>
    vector_base<T,Alloc>
      ::vector_base(InputIterator first,
                    InputIterator last)
        :mBegin(pointer(static_cast<T*>(0))),
         mSize(0),
         mCapacity(0),
         mAllocator()
{
  // check the type of InputIterator: if it's an integral type,
  // we need to interpret this call as (size_type, value_type)
  typedef thrust::detail::is_integral<InputIterator> Integer;

  init_dispatch(first, last, Integer());
} // end vector_basee::vector_base()

template<typename T, typename Alloc>
  void vector_base<T,Alloc>
    ::resize(size_type new_size, value_type x)
{
  if(new_size < size())
    erase(begin() + new_size, end());
  else
    insert(end(), new_size - size(), x);
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
    size_type new_capacity;
    iterator  new_begin;
    allocate_and_copy(n, begin(), end(), new_capacity, new_begin);

    mBegin = new_begin;
    mCapacity = new_capacity;
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
  void vector_base<T,Alloc>
    ::shrink_to_fit(void)
{
  // use the swap trick
  vector_base(*this).swap(*this);
} // end vector_base::shrink_to_fit()

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
  typename vector_base<T,Alloc>::reverse_iterator
    vector_base<T,Alloc>
      ::rbegin(void)
{
  return reverse_iterator(end());
} // end vector_base::rbegin()

template<typename T, typename Alloc>
  typename vector_base<T,Alloc>::const_reverse_iterator
    vector_base<T,Alloc>
      ::rbegin(void) const
{
  return const_reverse_iterator(end());
} // end vector_base::rbegin()

template<typename T, typename Alloc>
  typename vector_base<T,Alloc>::const_reverse_iterator
    vector_base<T,Alloc>
      ::crbegin(void) const
{
  return rbegin();
} // end vector_base::crbegin()

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
  typename vector_base<T,Alloc>::reverse_iterator
    vector_base<T,Alloc>
      ::rend(void)
{
  return reverse_iterator(begin());
} // end vector_base::rend()

template<typename T, typename Alloc>
  typename vector_base<T,Alloc>::const_reverse_iterator
    vector_base<T,Alloc>
      ::rend(void) const
{
  return const_reverse_iterator(begin());
} // end vector_base::rend()

template<typename T, typename Alloc>
  typename vector_base<T,Alloc>::const_reverse_iterator
    vector_base<T,Alloc>
      ::crend(void) const
{
  return rend();
} // end vector_base::crend()

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
  typename vector_base<T,Alloc>::pointer
    vector_base<T,Alloc>
      ::data(void)
{
  return &front();
} // end vector_base::data()

template<typename T, typename Alloc>
  typename vector_base<T,Alloc>::const_pointer
    vector_base<T,Alloc>
      ::data(void) const
{
  return &front();
} // end vector_base::data()

template<typename T, typename Alloc>
  vector_base<T,Alloc>
    ::~vector_base(void)
{
  // destroy every living thing
  thrust::detail::destroy(begin(), end());

  // deallocate
  mAllocator.deallocate(mBegin.base(), capacity());
} // end vector_base::~vector_base()

template<typename T, typename Alloc>
  void vector_base<T,Alloc>
    ::clear(void)
{
  // XXX TODO: possibly redo this
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
  insert(end(), x);
} // end vector_base::push_back()

template<typename T, typename Alloc>
  void vector_base<T,Alloc>
    ::pop_back(void)
{
  --mSize;
  thrust::detail::destroy(end(), end() + 1);
} // end vector_base::pop_back()

template<typename T, typename Alloc>
  typename vector_base<T,Alloc>::iterator vector_base<T,Alloc>
    ::erase(iterator pos)
{
  return erase(pos,pos+1);
} // end vector_base::erase()

template<typename T, typename Alloc>
  typename vector_base<T,Alloc>::iterator vector_base<T,Alloc>
    ::erase(iterator first, iterator last)
{
  // move the range [last,end()) to first
  iterator i = detail::move(last, end(), first);

  // destroy everything after i
  thrust::detail::destroy(i, end());

  // modify our size
  mSize -= (last - first);

  // return an iterator pointing to the position of the first element
  // following the erased range
  return first;
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
    ::assign(size_type n, const T &x)
{
  fill_assign(n, x);
} // end vector_base::assign()

template<typename T, typename Alloc>
  template<typename InputIterator>
    void vector_base<T,Alloc>
      ::assign(InputIterator first, InputIterator last)
{
  // we could have received assign(n, x), so disambiguate on the
  // type of InputIterator
  typedef typename thrust::detail::is_integral<InputIterator> integral;

  assign_dispatch(first, last, integral());
} // end vector_base::assign()

template<typename T, typename Alloc>
  typename vector_base<T,Alloc>::iterator
    vector_base<T,Alloc>
      ::insert(iterator position, const T &x)
{
  // find the index of the insertion
  size_type index = position - begin();

  // make the insertion
  insert(position, 1, x);

  // return an iterator pointing back to position
  return begin() + index;
} // end vector_base::insert()

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
      ::assign_dispatch(InputIterator first, InputIterator last, false_type)
{
  range_assign(first, last);
} // end vector_base::assign_dispatch()

template<typename T, typename Alloc>
  template<typename Integral>
    void vector_base<T,Alloc>
      ::assign_dispatch(Integral n, Integral x, true_type)
{
  fill_assign(n, x);
} // end vector_base::assign_dispatch()

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
        thrust::advance(mid, num_displaced_elements);

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
      size_type new_capacity = old_size + std::max(old_size, num_new_elements);

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
      size_type new_capacity = old_size + std::max(old_size, n);

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

template<typename T, typename Alloc>
  template<typename InputIterator>
    void vector_base<T,Alloc>
      ::range_assign(InputIterator first,
                     InputIterator last)
{
  // dispatch based on whether or not InputIterator
  // is strictly input_host_iterator_tag
  typedef typename thrust::iterator_traits<InputIterator>::iterator_category category;

  typedef typename thrust::detail::is_same<category, thrust::input_host_iterator_tag>::type input_host_iterator_or_not;

  range_assign(first, last, input_host_iterator_or_not());
} // end range_assign()

template<typename T, typename Alloc>
  template<typename InputHostIterator>
    void vector_base<T,Alloc>
      ::range_assign(InputHostIterator first,
                     InputHostIterator last,
                     true_type)
{
  iterator current(begin());

  // assign to elements which already exist
  for(; first != last && current != end(); ++current, ++first)
  {
    *current = *first;
  } // end for
  
  // either just the input was exhausted or both
  // the input and vector elements were exhausted
  if(first == last)
  {
    // if we exhausted the input, erase leftover elements
    erase(current, end());
  } // end if
  else
  {
    // insert the rest of the input at the end of the vector
    insert(end(), first, last);
  } // end else
} // end vector_base::range_assign()

template<typename T, typename Alloc>
  template<typename ForwardIterator>
    void vector_base<T,Alloc>
      ::range_assign(ForwardIterator first,
                     ForwardIterator last,
                     false_type)
{
  const size_type n = thrust::distance(first, last);

  if(n > capacity())
  {
    size_type new_capacity;
    iterator new_begin;

    allocate_and_copy(n, first, last, new_capacity, new_begin);

    // call destructors on the elements in the old storage
    thrust::detail::destroy(begin(), end());

    // deallocate the old storage
    mAllocator.deallocate(&*begin(), capacity());
  
    // record the vector's new parameters
    mBegin    = new_begin;
    mSize     = n;
    mCapacity = new_capacity;
  } // end if
  else if(size() >= n)
  {
    // we can already accomodate the new range
    iterator new_end = thrust::copy(first, last, begin());

    // destroy the elements we don't need
    thrust::detail::destroy(new_end, end());

    // update size
    mSize = n;
  } // end else if
  else
  {
    // range fits inside allocated storage, but some elements
    // have not been constructed yet
    
    // XXX TODO we could possibly implement this with one call
    // to transform rather than copy + uninitialized_copy

    // copy to elements which already exist
    ForwardIterator mid = first;
    thrust::advance(mid, size());
    thrust::copy(first, mid, begin());

    // uninitialize_copy to elements which must be constructed
    iterator new_end = thrust::uninitialized_copy(mid, last, end());

    // update size
    mSize = n;
  } // end else
} // end vector_base::assign()

template<typename T, typename Alloc>
  void vector_base<T,Alloc>
    ::fill_assign(size_type n, const T &x)
{
  if(n > capacity())
  {
    // XXX we should also include a copy of the allocator:
    // vector_base<T,Alloc> temp(n, x, get_allocator());
    vector_base<T,Alloc> temp(n, x);
    temp.swap(*this);
  } // end if
  else if(n > size())
  {
    // fill to existing elements
    thrust::fill(begin(), end(), x);

    // construct uninitialized elements
    thrust::uninitialized_fill(end(), end() + (n - size()), x);

    // adjust size
    mSize += (n - size());
  } // end else if
  else
  {
    // fill to existing elements
    thrust::fill(begin(), begin() + n, x);

    // erase the elements after the fill
    erase(begin() + n, end());
  } // end else
} // end vector_base::fill_assign()

template<typename T, typename Alloc>
  template<typename ForwardIterator>
    void vector_base<T,Alloc>
      ::allocate_and_copy(size_type requested_size,
                          ForwardIterator first, ForwardIterator last,
                          size_type &allocated_size,
                          iterator &new_storage)
{
  if(requested_size == 0)
  {
    allocated_size = 0;
    new_storage = iterator(pointer(static_cast<T*>(0)));
    return;
  } // end if

  // allocate exponentially larger new storage
  allocated_size = std::max<size_type>(requested_size, 2 * capacity());

  // do not exceed maximum storage
  allocated_size = std::min<size_type>(allocated_size, max_size());

  if(requested_size > allocated_size)
  {
    throw std::length_error("assignment exceeds max_size().");
  } // end if

  new_storage = mAllocator.allocate(allocated_size);

  try
  {
    // construct the range to the newly allocated storage
    thrust::uninitialized_copy(first, last, new_storage);
  } // end try
  catch(...)
  {
    // something went wrong, so destroy & deallocate the new storage 
    thrust::detail::destroy(new_storage, new_storage + requested_size);
    mAllocator.deallocate(&*new_storage, allocated_size);

    // rethrow
    throw;
  } // end catch
} // end vector_base::allocate_and_copy()

} // end detail

template<typename T, typename Alloc>
  void swap(detail::vector_base<T,Alloc> &a,
            detail::vector_base<T,Alloc> &b)
{
  a.swap(b);
} // end swap()



namespace detail
{
    
//////////////////////
// Host<->Host Path //
//////////////////////
template <typename InputIterator1, typename InputIterator2>
bool vector_equal(InputIterator1 first1, InputIterator1 last1,
                  InputIterator2 first2,
                  thrust::host_space_tag,
                  thrust::host_space_tag)
{
    return thrust::equal(first1, last1, first2);
}

//////////////////////////
// Device<->Device Path //
//////////////////////////
template <typename InputIterator1, typename InputIterator2>
bool vector_equal(InputIterator1 first1, InputIterator1 last1,
                  InputIterator2 first2,
                  thrust::device_space_tag,
                  thrust::device_space_tag)
{
    return thrust::equal(first1, last1, first2);
}

////////////////////////
// Host<->Device Path //
////////////////////////
template <typename InputIterator1, typename InputIterator2>
bool vector_equal(InputIterator1 first1, InputIterator1 last1,
                  InputIterator2 first2,
                  thrust::host_space_tag,
                  thrust::device_space_tag)
{
    typedef typename thrust::iterator_traits<InputIterator2>::value_type InputType2;
    
    // copy device sequence to host and compare on host
    raw_host_buffer<InputType2> buffer(first2, first2 + thrust::distance(first1, last1));

    return thrust::equal(first1, last1, buffer.begin());
}
  
////////////////////////
// Device<->Host Path //
////////////////////////
template <typename InputIterator1, typename InputIterator2> 
bool vector_equal(InputIterator1 first1, InputIterator1 last1,
                  InputIterator2 first2,
                  thrust::device_space_tag,
                  thrust::host_space_tag)
{
    typedef typename thrust::iterator_traits<InputIterator1>::value_type InputType1;
    
    // copy device sequence to host and compare on host
    raw_host_buffer<InputType1> buffer(first1, last1);

    return thrust::equal(buffer.begin(), buffer.end(), first2);
}

template <typename InputIterator1, typename InputIterator2>
bool vector_equal(InputIterator1 first1, InputIterator1 last1,
                  InputIterator2 first2)
{
    return vector_equal(first1, last1, first2,
            typename thrust::iterator_space< InputIterator1 >::type(),
            typename thrust::iterator_space< InputIterator2 >::type());
}

} // end namespace detail




template<typename T1, typename Alloc1,
         typename T2, typename Alloc2>
bool operator==(const detail::vector_base<T1,Alloc1>& lhs,
                const detail::vector_base<T2,Alloc2>& rhs)
{
    return lhs.size() == rhs.size() && detail::vector_equal(lhs.begin(), lhs.end(), rhs.begin());
}
    
template<typename T1, typename Alloc1,
         typename T2, typename Alloc2>
bool operator==(const detail::vector_base<T1,Alloc1>& lhs,
                const std::vector<T2,Alloc2>&         rhs)
{
    return lhs.size() == rhs.size() && detail::vector_equal(lhs.begin(), lhs.end(), rhs.begin());
}

template<typename T1, typename Alloc1,
         typename T2, typename Alloc2>
bool operator==(const std::vector<T1,Alloc1>&         lhs,
                const detail::vector_base<T2,Alloc2>& rhs)
{
    return lhs.size() == rhs.size() && detail::vector_equal(lhs.begin(), lhs.end(), rhs.begin());
}

template<typename T1, typename Alloc1,
         typename T2, typename Alloc2>
bool operator!=(const detail::vector_base<T1,Alloc1>& lhs,
                const detail::vector_base<T2,Alloc2>& rhs)
{
    return !(lhs == rhs);
}
    
template<typename T1, typename Alloc1,
         typename T2, typename Alloc2>
bool operator!=(const detail::vector_base<T1,Alloc1>& lhs,
                const std::vector<T2,Alloc2>&         rhs)
{
    return !(lhs == rhs);
}

template<typename T1, typename Alloc1,
         typename T2, typename Alloc2>
bool operator!=(const std::vector<T1,Alloc1>&         lhs,
                const detail::vector_base<T2,Alloc2>& rhs)
{
    return !(lhs == rhs);
}

} // end thrust

