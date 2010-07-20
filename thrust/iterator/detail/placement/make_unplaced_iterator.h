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
#include <thrust/iterator/detail/normal_iterator.h>

namespace thrust
{

namespace detail
{

template<typename PlacedIterator> struct unplaced_iterator;

// specialization for placed_iterator
template<typename UnplacedIterator>
  struct unplaced_iterator<placed_iterator<UnplacedIterator> >
{
  typedef UnplacedIterator type;
};

// specialization for normal_iterator
template<typename PlacedPointer>
  struct unplaced_iterator<normal_iterator<PlacedPointer> >
{
  private:
    typedef typename unplaced_iterator<PlacedPointer>::type unplaced_pointer;

  public:
    typedef normal_iterator<unplaced_pointer> type;
};


template<typename UnplacedIterator>
  inline typename unplaced_iterator<
    placed_iterator<UnplacedIterator>
  >::type
    make_unplaced_iterator(placed_iterator<UnplacedIterator> iter)
{
  return iter.base();
} // end make_unplaced_iterator()


template<typename PlacedPointer>
  inline typename unplaced_iterator<
    normal_iterator<PlacedPointer>
  >::type
    make_unplaced_iterator(normal_iterator<PlacedPointer> iter)
{
  return make_normal_iterator(make_unplaced_iterator(iter.base()));
} // end make_unplaced_iterator()


} // end detail

} // end thrust

