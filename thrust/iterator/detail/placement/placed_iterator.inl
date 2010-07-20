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

#pragma once

#include <thrust/iterator/detail/placement/placed_iterator.h>

namespace thrust
{

namespace detail
{


template<typename UnplacedIterator>
  placed_iterator<UnplacedIterator>
    ::placed_iterator(void)
      : super_t(),
        m_place()
{
  ;
} // end placed_iterator::placed_iterator()

template<typename UnplacedIterator>
  placed_iterator<UnplacedIterator>
    ::placed_iterator(UnplacedIterator i, place p)
      : super_t(i),
        m_place(p)
{
  ;
} // end placed_iterator::placed_iterator()

template<typename UnplacedIterator>
  template<typename OtherIterator>
    placed_iterator<UnplacedIterator>
      ::placed_iterator(placed_iterator<OtherIterator> i, place p)
        : super_t(i.base()),
          m_place(p)
{
  ;
} // end placed_iterator::placed_iterator()

template<typename UnplacedIterator>
  void placed_iterator<UnplacedIterator>
    ::set_place(place p)
{
  m_place = p;
} // end placed_iterator::set_place()

template<typename UnplacedIterator>
  typename placed_iterator<UnplacedIterator>::place
    placed_iterator<UnplacedIterator>
      ::get_place(void) const
{
  return m_place;
} // end placed_iterator::get_place()


template<typename UnplacedIterator>
  typename placed_iterator<UnplacedIterator>::super_t::reference
    placed_iterator<UnplacedIterator>
      ::dereference(void) const
{
  return *super_t::base();
} // end placed_iterator::dereference()


template<typename UnplacedIterator>
  placed_iterator<UnplacedIterator> make_placed_iterator(UnplacedIterator i, place p)
{
  return placed_iterator<UnplacedIterator>(i,p);
} // end make_placed_iterator()

template<typename UnplacedIterator>
  placed_iterator<UnplacedIterator> make_placed_iterator(UnplacedIterator i, std::size_t p)
{
  typedef typename placed_iterator<UnplacedIterator>::place place;
  return make_placed_iterator(i, place(p));
} // end make_placed_iterator()


} // end detail

} // end thrust

