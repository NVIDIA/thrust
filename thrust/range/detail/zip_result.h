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

#include <thrust/iterator/zip_iterator.h>
#include <thrust/range/iterator_range.h>

namespace thrust
{

namespace experimental
{

namespace detail
{


template<typename Range>
  struct zip1_result
{
  private:
    typedef typename range_iterator<Range>::type Iterator;
    typedef tuple<Iterator>                      IteratorTuple;
    typedef zip_iterator<IteratorTuple>          ZipIterator;

  public:
    typedef iterator_range<ZipIterator>          type;
};


template<typename Range1, typename Range2>
  struct zip2_result
{
  private:
    typedef typename range_iterator<Range1>::type Iterator1;
    typedef typename range_iterator<Range2>::type Iterator2;
    typedef tuple<Iterator1,Iterator2>            IteratorTuple;
    typedef zip_iterator<IteratorTuple>           ZipIterator;

  public:
    typedef iterator_range<ZipIterator>          type;
};


template<typename Range1, typename Range2, typename Range3>
  struct zip3_result
{
  private:
    typedef typename range_iterator<Range1>::type Iterator1;
    typedef typename range_iterator<Range2>::type Iterator2;
    typedef typename range_iterator<Range3>::type Iterator3;
    typedef tuple<Iterator1,Iterator2,Iterator3>  IteratorTuple;
    typedef zip_iterator<IteratorTuple>           ZipIterator;

  public:
    typedef iterator_range<ZipIterator>          type;
};


template<typename Range1, typename Range2, typename Range3, typename Range4>
  struct zip4_result
{
  private:
    typedef typename range_iterator<Range1>::type           Iterator1;
    typedef typename range_iterator<Range2>::type           Iterator2;
    typedef typename range_iterator<Range3>::type           Iterator3;
    typedef typename range_iterator<Range4>::type           Iterator4;
    typedef tuple<Iterator1,Iterator2,Iterator3,Iterator4>  IteratorTuple;
    typedef zip_iterator<IteratorTuple>                     ZipIterator;

  public:
    typedef iterator_range<ZipIterator>                     type;
};


template<typename Range1, typename Range2, typename Range3, typename Range4, typename Range5>
  struct zip5_result
{
  private:
    typedef typename range_iterator<Range1>::type                     Iterator1;
    typedef typename range_iterator<Range2>::type                     Iterator2;
    typedef typename range_iterator<Range3>::type                     Iterator3;
    typedef typename range_iterator<Range4>::type                     Iterator4;
    typedef typename range_iterator<Range5>::type                     Iterator5;
    typedef tuple<Iterator1,Iterator2,Iterator3,Iterator4,Iterator5>  IteratorTuple;
    typedef zip_iterator<IteratorTuple>                               ZipIterator;

  public:
    typedef iterator_range<ZipIterator>                     type;
};


template<
  typename Range1,
  typename Range2,
  typename Range3,
  typename Range4,
  typename Range5,
  typename Range6
>
  struct zip6_result
{
  private:
    typedef typename range_iterator<Range1>::type                     Iterator1;
    typedef typename range_iterator<Range2>::type                     Iterator2;
    typedef typename range_iterator<Range3>::type                     Iterator3;
    typedef typename range_iterator<Range4>::type                     Iterator4;
    typedef typename range_iterator<Range5>::type                     Iterator5;
    typedef typename range_iterator<Range6>::type                     Iterator6;

    typedef tuple<
      Iterator1,
      Iterator2,
      Iterator3,
      Iterator4,
      Iterator5,
      Iterator6
    > IteratorTuple;

    typedef zip_iterator<IteratorTuple>                               ZipIterator;

  public:
    typedef iterator_range<ZipIterator>                     type;
};


template<
  typename Range1,
  typename Range2,
  typename Range3,
  typename Range4,
  typename Range5,
  typename Range6,
  typename Range7
>
  struct zip7_result
{
  private:
    typedef typename range_iterator<Range1>::type                     Iterator1;
    typedef typename range_iterator<Range2>::type                     Iterator2;
    typedef typename range_iterator<Range3>::type                     Iterator3;
    typedef typename range_iterator<Range4>::type                     Iterator4;
    typedef typename range_iterator<Range5>::type                     Iterator5;
    typedef typename range_iterator<Range6>::type                     Iterator6;
    typedef typename range_iterator<Range7>::type                     Iterator7;

    typedef tuple<
      Iterator1,
      Iterator2,
      Iterator3,
      Iterator4,
      Iterator5,
      Iterator6,
      Iterator7
    > IteratorTuple;

    typedef zip_iterator<IteratorTuple>                               ZipIterator;

  public:
    typedef iterator_range<ZipIterator>                     type;
};


template<
  typename Range1,
  typename Range2,
  typename Range3,
  typename Range4,
  typename Range5,
  typename Range6,
  typename Range7,
  typename Range8
>
  struct zip8_result
{
  private:
    typedef typename range_iterator<Range1>::type                     Iterator1;
    typedef typename range_iterator<Range2>::type                     Iterator2;
    typedef typename range_iterator<Range3>::type                     Iterator3;
    typedef typename range_iterator<Range4>::type                     Iterator4;
    typedef typename range_iterator<Range5>::type                     Iterator5;
    typedef typename range_iterator<Range6>::type                     Iterator6;
    typedef typename range_iterator<Range7>::type                     Iterator7;
    typedef typename range_iterator<Range8>::type                     Iterator8;

    typedef tuple<
      Iterator1,
      Iterator2,
      Iterator3,
      Iterator4,
      Iterator5,
      Iterator6,
      Iterator7,
      Iterator8
    > IteratorTuple;

    typedef zip_iterator<IteratorTuple>                               ZipIterator;

  public:
    typedef iterator_range<ZipIterator>                     type;
};


template<
  typename Range1,
  typename Range2,
  typename Range3,
  typename Range4,
  typename Range5,
  typename Range6,
  typename Range7,
  typename Range8,
  typename Range9
>
  struct zip9_result
{
  private:
    typedef typename range_iterator<Range1>::type                     Iterator1;
    typedef typename range_iterator<Range2>::type                     Iterator2;
    typedef typename range_iterator<Range3>::type                     Iterator3;
    typedef typename range_iterator<Range4>::type                     Iterator4;
    typedef typename range_iterator<Range5>::type                     Iterator5;
    typedef typename range_iterator<Range6>::type                     Iterator6;
    typedef typename range_iterator<Range7>::type                     Iterator7;
    typedef typename range_iterator<Range8>::type                     Iterator8;
    typedef typename range_iterator<Range9>::type                     Iterator9;

    typedef tuple<
      Iterator1,
      Iterator2,
      Iterator3,
      Iterator4,
      Iterator5,
      Iterator6,
      Iterator7,
      Iterator8,
      Iterator9
    > IteratorTuple;

    typedef zip_iterator<IteratorTuple>                               ZipIterator;

  public:
    typedef iterator_range<ZipIterator>                     type;
};


template<
  typename Range1,
  typename Range2,
  typename Range3,
  typename Range4,
  typename Range5,
  typename Range6,
  typename Range7,
  typename Range8,
  typename Range9,
  typename Range10
>
  struct zip10_result
{
  private:
    typedef typename range_iterator<Range1>::type                     Iterator1;
    typedef typename range_iterator<Range2>::type                     Iterator2;
    typedef typename range_iterator<Range3>::type                     Iterator3;
    typedef typename range_iterator<Range4>::type                     Iterator4;
    typedef typename range_iterator<Range5>::type                     Iterator5;
    typedef typename range_iterator<Range6>::type                     Iterator6;
    typedef typename range_iterator<Range7>::type                     Iterator7;
    typedef typename range_iterator<Range8>::type                     Iterator8;
    typedef typename range_iterator<Range9>::type                     Iterator9;
    typedef typename range_iterator<Range10>::type                    Iterator10;

    typedef tuple<
      Iterator1,
      Iterator2,
      Iterator3,
      Iterator4,
      Iterator5,
      Iterator6,
      Iterator7,
      Iterator8,
      Iterator9,
      Iterator10
    > IteratorTuple;

    typedef zip_iterator<IteratorTuple>                               ZipIterator;

  public:
    typedef iterator_range<ZipIterator>                     type;
};


} // end detail

} // end experimental

} // end thrust

