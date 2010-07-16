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


template<typename Iterator>
  placed_iterator<Iterator>
    ::placed_iterator(void)
      : super_t(),
        m_place()
{
  ;
} // end placed_iterator::placed_iterator()

template<typename Iterator>
  template<typename OtherIterator>
    placed_iterator<Iterator>
      ::placed_iterator(OtherIterator i, place p)
        : super_t(i),
          m_place(p)
{
  ;
} // end placed_iterator::placed_iterator()

template<typename Iterator>
  void placed_iterator<Iterator>
    ::set_place(place p)
{
  m_place = p;
} // end placed_iterator::set_place()

template<typename Iterator>
  typename placed_iterator<Iterator>::place
    placed_iterator<Iterator>
      ::get_place(void) const
{
  return m_place;
} // end placed_iterator::get_place()


template<typename Iterator>
  typename placed_iterator<Iterator>::super_t::reference
    placed_iterator<Iterator>
      ::dereference(void) const
{
  return *super_t::base();
} // end placed_iterator::dereference()


} // end detail

} // end thrust

