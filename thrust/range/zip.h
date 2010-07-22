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
#include <thrust/range/begin.h>
#include <thrust/range/end.h>
#include <thrust/range/detail/zip_result.h>

// spawn more overloads

namespace thrust
{

namespace experimental
{


template<typename Range>
  typename detail::zip1_result<Range>::type
    zip(Range &rng)
{
  return make_iterator_range(make_zip_iterator(make_tuple(begin(rng))),
                             make_zip_iterator(make_tuple(end(rng))));
} // end zip()


template<typename Range>
  typename detail::zip1_result<const Range>::type
    zip(const Range &rng)
{
  return make_iterator_range(make_zip_iterator(make_tuple(begin(rng))),
                             make_zip_iterator(make_tuple(end(rng))));
} // end zip()



template<typename Range1, typename Range2>
  typename detail::zip2_result<Range1,Range2>::type
    zip(Range1 &rng1,
        Range2 &rng2)
{
  return make_iterator_range(make_zip_iterator(make_tuple(begin(rng1), begin(rng2))),
                             make_zip_iterator(make_tuple(end(rng1), end(rng2))));
} // end zip()


template<typename Range1, typename Range2>
  typename detail::zip2_result<Range1,const Range2>::type
    zip(Range1 &rng1,
        const Range2 &rng2)
{
  return make_iterator_range(make_zip_iterator(make_tuple(begin(rng1), begin(rng2))),
                             make_zip_iterator(make_tuple(end(rng1), end(rng2))));
} // end zip()


template<typename Range1, typename Range2>
  typename detail::zip2_result<const Range1,Range2>::type
    zip(const Range1 &rng1,
        Range2 &rng2)
{
  return make_iterator_range(make_zip_iterator(make_tuple(begin(rng1), begin(rng2))),
                             make_zip_iterator(make_tuple(end(rng1), end(rng2))));
} // end zip()


template<typename Range1, typename Range2>
  typename detail::zip2_result<const Range1,const Range2>::type
    zip(const Range1 &rng1,
        const Range2 &rng2)
{
  return make_iterator_range(make_zip_iterator(make_tuple(begin(rng1), begin(rng2))),
                             make_zip_iterator(make_tuple(end(rng1), end(rng2))));
} // end zip()


template<typename Range1, typename Range2, typename Range3>
  typename detail::zip3_result<Range1,Range2,Range3>::type
    zip(Range1 &rng1,
        Range2 &rng2,
        Range3 &rng3)
{
  return make_iterator_range(make_zip_iterator(make_tuple(begin(rng1), begin(rng2), begin(rng3))),
                             make_zip_iterator(make_tuple(end(rng1), end(rng2), end(rng3))));
} // end zip()


template<typename Range1, typename Range2, typename Range3>
  typename detail::zip3_result<Range1,Range2,const Range3>::type
    zip(Range1 &rng1,
        Range2 &rng2,
        const Range3 &rng3)
{
  return make_iterator_range(make_zip_iterator(make_tuple(begin(rng1), begin(rng2), begin(rng3))),
                             make_zip_iterator(make_tuple(end(rng1), end(rng2), end(rng3))));
} // end zip()


template<typename Range1, typename Range2, typename Range3>
  typename detail::zip3_result<Range1,const Range2,Range3>::type
    zip(Range1 &rng1,
        const Range2 &rng2,
        Range3 &rng3)
{
  return make_iterator_range(make_zip_iterator(make_tuple(begin(rng1), begin(rng2), begin(rng3))),
                             make_zip_iterator(make_tuple(end(rng1), end(rng2), end(rng3))));
} // end zip()


template<typename Range1, typename Range2, typename Range3>
  typename detail::zip3_result<Range1,const Range2,const Range3>::type
    zip(Range1 &rng1,
        const Range2 &rng2,
        const Range3 &rng3)
{
  return make_iterator_range(make_zip_iterator(make_tuple(begin(rng1), begin(rng2), begin(rng3))),
                             make_zip_iterator(make_tuple(end(rng1), end(rng2), end(rng3))));
} // end zip()


template<typename Range1, typename Range2, typename Range3>
  typename detail::zip3_result<const Range1,Range2,Range3>::type
    zip(const Range1 &rng1,
        Range2 &rng2,
        Range3 &rng3)
{
  return make_iterator_range(make_zip_iterator(make_tuple(begin(rng1), begin(rng2), begin(rng3))),
                             make_zip_iterator(make_tuple(end(rng1), end(rng2), end(rng3))));
} // end zip()


template<typename Range1, typename Range2, typename Range3>
  typename detail::zip3_result<const Range1,Range2,const Range3>::type
    zip(const Range1 &rng1,
        Range2 &rng2,
        const Range3 &rng3)
{
  return make_iterator_range(make_zip_iterator(make_tuple(begin(rng1), begin(rng2), begin(rng3))),
                             make_zip_iterator(make_tuple(end(rng1), end(rng2), end(rng3))));
} // end zip()


template<typename Range1, typename Range2, typename Range3>
  typename detail::zip3_result<const Range1,const Range2,Range3>::type
    zip(const Range1 &rng1,
        const Range2 &rng2,
        Range3 &rng3)
{
  return make_iterator_range(make_zip_iterator(make_tuple(begin(rng1), begin(rng2), begin(rng3))),
                             make_zip_iterator(make_tuple(end(rng1), end(rng2), end(rng3))));
} // end zip()


template<typename Range1, typename Range2, typename Range3>
  typename detail::zip3_result<const Range1,const Range2,const Range3>::type
    zip(const Range1 &rng1,
        const Range2 &rng2,
        const Range3 &rng3)
{
  return make_iterator_range(make_zip_iterator(make_tuple(begin(rng1), begin(rng2), begin(rng3))),
                             make_zip_iterator(make_tuple(end(rng1), end(rng2), end(rng3))));
} // end zip()


} // end experimental

} // end thrust

