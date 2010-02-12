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


/*! \file is_sorted.h
 *  \brief Determine if a range is sorted.
 */

#pragma once

#include <thrust/detail/config.h>

namespace thrust
{

/*! \addtogroup reductions
 *  \{
 *  \addtogroup predicates
 *  \{
 */

/*! \p is_sorted returns \c true if the range <tt>[first, last)</tt> is
 *  sorted in ascending order, and \c false otherwise.
 *
 *  Specifically, this version of \p is_sorted returns \c false if for
 *  some iterator \c i in the range <tt>[first, last - 1)</tt> the
 *  expression <tt>*(i + 1) < *i</tt> is \c true.
 *
 *  \param first The beginning of the sequence.
 *  \param last  The end of the sequence.
 *  \return \c true, if the sequence is sorted; \c false, otherwise.
 *
 *  \tparam ForwardIterator is a model of <a href="http://www.sgi.com/tech/stl/ForwardIterator.html">Forward Iterator</a>,
 *          \p ForwardIterator's \c value_type is a model of <a href="http://www.sgi.com/tech/stl/LessThanComparable.html">LessThan Comparable</a>,
 *          and the ordering on objects of \p ForwardIterator's \c value_type is a <em>strict weak ordering</em>, as defined
 *          in the <a href="http://www.sgi.com/tech/stl/LessThanComparable.html">LessThan Comparable</a> requirements.
 *
 *
 *  The following code demonstrates how to use \p is_sorted to test whether the
 *  contents of a \c device_vector are stored in ascending order.
 *
 *  \code
 *  #include <thrust/is_sorted.h>
 *  #include <thrust/device_vector.h>
 *  #include <thrust/sort.h>
 *  ...
 *  thrust::device_vector<int> v(6);
 *  v[0] = 1;
 *  v[1] = 4;
 *  v[2] = 2;
 *  v[3] = 8;
 *  v[4] = 5;
 *  v[5] = 7;
 *
 *  bool result = thrust::is_sorted(v.begin(), v.end());
 *
 *  // result == false
 *
 *  thrust::sort(v.begin(), v.end());
 *  result = thrust::is_sorted(v.begin(), v.end());
 *
 *  // result == true
 *  \endcode
 *
 *  \see http://www.sgi.com/tech/stl/is_sorted.html
 *  \see \c sort
 *  \see \c stable_sort
 *  \see \c sorting::merge_sort
 *  \see \c sorting::stable_merge_sort
 *  \see \c sorting::radix_sort
 *  \see \c sorting::stable_radix_sort
 *  \see \c less<T>
 */
template <typename ForwardIterator>
bool is_sorted(ForwardIterator first, ForwardIterator last);

/*! \p is_sorted returns \c true if the range <tt>[first, last)</tt> is sorted in ascending 
 *  order accoring to a user-defined comparison operation, and \c false otherwise.
 *
 *  Specifically, this version of \p is_sorted returns \c false if for some iterator \c i in
 *  the range <tt>[first, last - 1)</tt> the expression <tt>comp(*(i + 1), *i)</tt> is \c true.
 *
 *  \param first The beginning of the sequence.
 *  \param last  The end of the sequence.
 *  \param comp  Comparison operator.
 *  \return \c true, if the sequence is sorted according to comp; \c false, otherwise.
 *
 *  \tparam ForwardIterator is a model of <a href="http://www.sgi.com/tech/stl/ForwardIterator.html">Forward Iterator</a>,
 *          and \p ForwardIterator's \c value_type is convertible to both \c StrictWeakOrdering's \c first_argument_type
 *          and \c second_argument_type.
 *  \tparam StrictWeakOrdering is a model of <a href="http://www.sgi.com/tech/stl/StrictWeakOrdering.html">Strict Weak Ordering</a>.
 *
 *  The following code snippet demonstrates how to use \p is_sorted to test whether the
 *  contents of a \c device_vector are stored in descending order.
 *
 *  \code
 *  #include <thrust/is_sorted.h>
 *  #include <thrust/functional.h>
 *  #include <thrust/device_vector.h>
 *  #include <thrust/sort.h>
 *  ...
 *  thrust::device_vector<int> v(6);
 *  v[0] = 1;
 *  v[1] = 4;
 *  v[2] = 2;
 *  v[3] = 8;
 *  v[4] = 5;
 *  v[5] = 7;
 *
 *  thrust::greater<int> comp;
 *  bool result = thrust::is_sorted(v.begin(), v.end(), comp);
 *
 *  // result == false
 *
 *  thrust::sort(v.begin(), v.end(), comp);
 *  result = thrust::is_sorted(v.begin(), v.end(), comp);
 *
 *  // result == true
 *  \endcode
 *
 *  \see http://www.sgi.com/tech/stl/is_sorted.html
 *  \see \c sort
 *  \see \c stable_sort
 *  \see \c sorting::merge_sort
 *  \see \c sorting::stable_merge_sort
 *  \see \c less<T>
 */
template <typename ForwardIterator, typename StrictWeakOrdering>
bool is_sorted(ForwardIterator first, ForwardIterator last, StrictWeakOrdering comp);

/*! \} // end predicates
 *  \} // end reductions
 */

}; // end thrust

#include <thrust/detail/is_sorted.inl>

