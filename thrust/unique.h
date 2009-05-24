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


/*! \file unique.h
 *  \brief Move unique elements to the front of a sequence.
 */

#pragma once

#include <thrust/detail/config.h>

namespace thrust
{

/*! \addtogroup stream_compaction
 *  \{
 */

/*! For each group of consecutive elements in the range <tt>[first, last)</tt>
 *  with the same value, \p unique removes all but the first element of 
 *  the group. The return value is an iterator \c new_last such that 
 *  no two consecutive elements in the range <tt>[first, new_last)</tt> are
 *  equal. The iterators in the range <tt>[new_last, last)</tt> are all still
 *  dereferenceable, but the elements that they point to are unspecified.
 *  \p unique is stable, meaning that the relative order of elements that are
 *  not removed is unchanged.
 *
 *  This version of \p unique uses \c operator== to test for equality.
 *
 *  \param first The beginning of the input range.
 *  \param last  The end of the input range.
 *  \return The end of the unique range <tt>[first, new_last)</tt>.
 *
 *  \tparam ForwardIterator is a model of <a href="http://www.sgi.com/tech/stl/ForwardIterator.html">Forward Iterator</a>,
 *          and \p ForwardIterator is mutable,
 *          and \p ForwardIterator's \c value_type is a model of <a href="http://www.sgi.com/tech/stl/EqualityComparable.html">Equality Comparable</a>.
 *
 *  The following code snippet demonstrates how to use \p unique to
 *  compact a sequence of numbers to remove consecutive duplicates.
 *
 *  \code
 *  #include <thrust/unique.h>
 *  ...
 *  const int N = 7;
 *  int A[N] = {1, 3, 3, 3, 2, 2, 1};
 *  int *new_end = thrust::unique(A, A + N);
 *  // The first four values of A are now {1, 3, 2, 1}
 *  // Values beyond new_end are unspecified.
 *  \endcode
 *
 *  \see http://www.sgi.com/tech/stl/unique.html
 */
template <typename ForwardIterator>
ForwardIterator unique(ForwardIterator first, ForwardIterator last);

/*! For each group of consecutive elements in the range <tt>[first, last)</tt>
 *  with the same value, \p unique removes all but the first element of 
 *  the group. The return value is an iterator \c new_last such that 
 *  no two consecutive elements in the range <tt>[first, new_last)</tt> are
 *  equal. The iterators in the range <tt>[new_last, last)</tt> are all still
 *  dereferenceable, but the elements that they point to are unspecified.
 *  \p unique is stable, meaning that the relative order of elements that are
 *  not removed is unchanged.
 *
 *  This version of \p unique uses the function object \p binary_pred to test
 *  for equality.
 *
 *  \param first The beginning of the input range.
 *  \param last  The end of the input range.
 *  \param binary_pred  The binary predicate used to determine equality.
 *  \return The end of the unique range <tt>[first, new_last)</tt>
 *
 *  \tparam ForwardIterator is a model of <a href="http://www.sgi.com/tech/stl/ForwardIterator.html">Forward Iterator</a>,
 *          and \p ForwardIterator is mutable,
 *          and \p ForwardIterator's \c value_type is convertible to \p BinaryPredicate's \c first_argument_type and to \p BinaryPredicate's \c second_argument_type.
 *  \tparam BinaryPredicate is a model of <a href="http://www.sgi.com/tech/stl/BinaryPredicate.html">Binary Predicate</a>.
 *
 *  \see http://www.sgi.com/tech/stl/unique.html
 */
template <typename ForwardIterator, typename BinaryPredicate>
ForwardIterator unique(ForwardIterator first, ForwardIterator last,
                       BinaryPredicate binary_pred);

/*! \} // end stream_compaction
 */

}; // end namespace thrust

#include <thrust/detail/unique.inl>

