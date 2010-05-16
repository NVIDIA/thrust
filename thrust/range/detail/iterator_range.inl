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

//  Copyright Neil Groves 2009.
//  Use, modification and distribution is subject to the Boost Software
//  License, Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)
//
// For more information, see http://www.boost.org/libs/range/

#pragma once

namespace thrust
{

namespace experimental
{


template<typename Iterator>
  iterator_range<Iterator>
    ::iterator_range(void)
      : m_begin(iterator()), m_end(iterator())
{
  ;
} // end iterator_range::iterator_range()


template<typename Iterator>
  template<typename OtherIterator>
    iterator_range<Iterator>
      ::iterator_range(OtherIterator begin, OtherIterator end)
        : m_begin(begin), m_end(end)
{
  ;
} // end iterator_range::iterator_range()


template<typename Iterator>
  template<typename OtherIterator>
    iterator_range<Iterator> &
      iterator_range<Iterator>
        ::operator=(const iterator_range<OtherIterator> &r)
{
  m_begin = r.begin();
  m_end   = r.end();
  return *this;
} // end iterator_range::operator=()


template<typename Iterator>
  typename iterator_range<Iterator>::iterator
    iterator_range<Iterator>
      ::begin(void) const
{
  return m_begin;
} // end iterator_range::begin()


template<typename Iterator>
  typename iterator_range<Iterator>::iterator
    iterator_range<Iterator>
      ::end(void) const
{
  return m_end;
} // end iterator_range::end()


template<typename Iterator>
  typename iterator_range<Iterator>::difference_type
    iterator_range<Iterator>
      ::size(void) const
{
  // XXX this seems not generic
  return m_end - m_begin;
} // end iterator_range::size()


template<typename Iterator>
  bool iterator_range<Iterator>
    ::empty(void) const
{
  return m_begin == m_end;
} // end iterator_range::empty()


template<typename Iterator>
  iterator_range<Iterator>
    ::operator bool (void) const
{
  return !empty();
} // end iterator_range::operator bool ()


template<typename Iterator>
  bool iterator_range<Iterator>
    ::equal(const iterator_range& r) const
{
  return m_begin == r.m_begin && m_end == r.m_end;
} // end iterator_range::equal()


template<typename Iterator>
  typename iterator_range<Iterator>::reference
    iterator_range<Iterator>
      ::front(void) const
{
  return *begin();
} // end iterator_range::front()


template<typename Iterator>
  typename iterator_range<Iterator>::reference
    iterator_range<Iterator>
      ::back(void) const
{
  iterator last(end());
  --last;
  return *last;
} // end iterator_range::front()


template<typename Iterator>
  typename iterator_range<Iterator>::reference
    iterator_range<Iterator>
      ::operator[](difference_type at) const
{
  return m_begin[at];
} // end operator[]()


template<typename Iterator>
  iterator_range<Iterator>
    make_iterator_range(Iterator begin, Iterator end)
{
  return iterator_range<Iterator>(begin,end);
} // end make_iterator_range


} // end experimental

} // end thrust

