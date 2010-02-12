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


/*! \file sort.h
 *  \brief Defines the interface to various
 *         sorting functions.
 */

#pragma once

#include <thrust/detail/config.h>

namespace thrust
{


/*! \addtogroup sorting
 *  \ingroup algorithms
 *  \{
 */

/*! \p sort sorts the elements in <tt>[first, last)</tt> into
 *  ascending order, meaning that if \c i and \c j are any two valid
 *  iterators in <tt>[first, last)</tt> such that \c i precedes \c j,
 *  then \c *j is not less than \c *i. Note: \c sort is not guaranteed
 *  to be stable. That is, suppose that \c *i and \c *j are equivalent:
 *  neither one is less than the other. It is not guaranteed that the
 *  relative order of these two elements will be preserved by \p sort.
 *
 *  This version of \p sort compares objects using \c operator<.
 *
 *  \param first The beginning of the sequence.
 *  \param last The end of the sequence.
 *
 *  \tparam RandomAccessIterator is a model of <a href="http://www.sgi.com/tech/stl/RandomAccessIterator.html">Random Access Iterator</a>,
 *          \p RandomAccessIterator is mutable,
 *          and \p RandomAccessIterator's \c value_type is a model of <a href="http://www.sgi.com/tech/stl/LessThanComparable.html">LessThan Comparable</a>,
 *          and the ordering relation on \p RandomAccessIterator's \c value_type is a <em>strict weak ordering</em>, as defined in the
 *          <a href="http://www.sgi.com/tech/stl/LessThanComparable.html">LessThan Comparable</a> requirements.
 *
 *  The following code snippet demonstrates how to use \p sort to sort
 *  a sequence of integers.
 *
 *  \code
 *  #include <thrust/sort.h>
 *  ...
 *  const int N = 6;
 *  int A[N] = {1, 4, 2, 8, 5, 7};
 *  thrust::sort(A, A + N);
 *  // A is now {1, 2, 4, 5, 7, 8}
 *  \endcode
 *
 *  \see http://www.sgi.com/tech/stl/sort.html
 *  \see \p stable_sort
 *  \see \p sort_by_key
 */
template<typename RandomAccessIterator>
  void sort(RandomAccessIterator first,
            RandomAccessIterator last);

/*! \p sort sorts the elements in <tt>[first, last)</tt> into
 *  ascending order, meaning that if \c i and \c j are any two valid
 *  iterators in <tt>[first, last)</tt> such that \c i precedes \c j,
 *  then \c *j is not less than \c *i. Note: \c sort is not guaranteed
 *  to be stable. That is, suppose that \c *i and \c *j are equivalent:
 *  neither one is less than the other. It is not guaranteed that the
 *  relative order of these two elements will be preserved by \p sort.
 *
 *  This version of \p sort compares objects using a function object
 *  \p comp.
 *
 *  \param first The beginning of the sequence.
 *  \param last The end of the sequence.
 *  \param comp  Comparison operator.
 *
 *  \tparam RandomAccessIterator is a model of <a href="http://www.sgi.com/tech/stl/RandomAccessIterator.html">Random Access Iterator</a>,
 *          \p RandomAccessIterator is mutable,
 *          and \p RandomAccessIterator's \c value_type is convertible to \p StrictWeakOrdering's
 *          \c first_argument_type and \c second_argument_type.
 *  \tparam StrictWeakOrdering is a model of <a href="http://www.sgi.com/tech/stl/StrictWeakOrdering.html">Strict Weak Ordering</a>.
 *
 *  \see http://www.sgi.com/tech/stl/sort.html
 *  \see \p stable_sort
 *  \see \p sort_by_key
 */
template<typename RandomAccessIterator,
         typename StrictWeakOrdering>
  void sort(RandomAccessIterator first,
            RandomAccessIterator last,
            StrictWeakOrdering comp);

/*! \p stable_sort is much like \c sort: it sorts the elements in
 *  <tt>[first, last)</tt> into ascending order, meaning that if \c i
 *  and \c j are any two valid iterators in <tt>[first, last)</tt> such
 *  that \c i precedes \c j, then \c *j is not less than \c *i.
 *
 *  As the name suggests, \p stable_sort is stable: it preserves the
 *  relative ordering of equivalent elements. That is, if \c x and \c y
 *  are elements in <tt>[first, last)</tt> such that \c x precedes \c y,
 *  and if the two elements are equivalent (neither <tt>x < y</tt> nor
 *  <tt>y < x</tt>) then a postcondition of \p stable_sort is that \c x
 *  still precedes \c y.
 *
 *  This version of \p stable_sort compares objects using \c operator<.
 *
 *  \param first The beginning of the sequence.
 *  \param last The end of the sequence.
 *
 *  \tparam RandomAccessIterator is a model of <a href="http://www.sgi.com/tech/stl/RandomAccessIterator.html">Random Access Iterator</a>,
 *          \p RandomAccessIterator is mutable,
 *          and \p RandomAccessIterator's \c value_type is a model of <a href="http://www.sgi.com/tech/stl/LessThanComparable.html">LessThan Comparable</a>,
 *          and the ordering relation on \p RandomAccessIterator's \c value_type is a <em>strict weak ordering</em>, as defined in the
 *          <a href="http://www.sgi.com/tech/stl/LessThanComparable.html">LessThan Comparable</a> requirements.
 *
 *  The following code snippet demonstrates how to use \p sort to sort
 *  a sequence of integers.
 *
 *  \code
 *  #include <thrust/sort.h>
 *  ...
 *  const int N = 6;
 *  int A[N] = {1, 4, 2, 8, 5, 7};
 *  thrust::stable_sort(A, A + N);
 *  // A is now {1, 2, 4, 5, 7, 8}
 *  \endcode
 *
 *  \see http://www.sgi.com/tech/stl/stable_sort.html
 *  \see \p sort
 *  \see \p stable_sort_by_key
 */
template<typename RandomAccessIterator>
  void stable_sort(RandomAccessIterator first,
                   RandomAccessIterator last);

/*! \p stable_sort is much like \c sort: it sorts the elements in
 *  <tt>[first, last)</tt> into ascending order, meaning that if \c i
 *  and \c j are any two valid iterators in <tt>[first, last)</tt> such
 *  that \c i precedes \c j, then \c *j is not less than \c *i.
 *
 *  As the name suggests, \p stable_sort is stable: it preserves the
 *  relative ordering of equivalent elements. That is, if \c x and \c y
 *  are elements in <tt>[first, last)</tt> such that \c x precedes \c y,
 *  and if the two elements are equivalent (neither <tt>x < y</tt> nor
 *  <tt>y < x</tt>) then a postcondition of \p stable_sort is that \c x
 *  still precedes \c y.
 *
 *  This version of \p stable_sort compares objects using a function object
 *  \p comp.
 *
 *  \param first The beginning of the sequence.
 *  \param last The end of the sequence.
 *  \param comp Comparison operator.
 *
 *  \tparam RandomAccessIterator is a model of <a href="http://www.sgi.com/tech/stl/RandomAccessIterator.html">Random Access Iterator</a>,
 *          \p RandomAccessIterator is mutable,
 *          and \p RandomAccessIterator's \c value_type is convertible to \p StrictWeakOrdering's
 *          \c first_argument_type and \c second_argument_type.
 *  \tparam StrictWeakOrdering is a model of <a href="http://www.sgi.com/tech/stl/StrictWeakOrdering.html">Strict Weak Ordering</a>.
 *
 *  \see http://www.sgi.com/tech/stl/stable_sort.html
 *  \see \p sort
 *  \see \p stable_sort_by_key
 */
template<typename RandomAccessIterator,
         typename StrictWeakOrdering>
  void stable_sort(RandomAccessIterator first,
                   RandomAccessIterator last,
                   StrictWeakOrdering comp);

///////////////
// Key Value //
///////////////

/*! \p sort_by_key performs a key-value sort. That is, \p sort_by_key sorts the
 *  elements in <tt>[keys_first, keys_last)</tt> and <tt>[values_first,
 *  values_first + (keys_last - keys_first))</tt> into ascending key order,
 *  meaning that if \c i and \c j are any two valid iterators in <tt>[keys_first,
 *  keys_last)</tt> such that \c i precedes \c j, and \c p and \c q are iterators
 *  in <tt>[values_first, values_first + (keys_last - keys_first))</tt>
 *  corresponding to \c i and \c j respectively, then \c *j is not less than
 *  \c *i.
 *
 *  Note: \c sort_by_key is not guaranteed to be stable. That is, suppose that
 *  \c *i and \c *j are equivalent: neither one is less than the other. It is not
 *  guaranteed that the relative order of these two keys or the relative
 *  order of their corresponding values will be preserved by \p sort_by_key.
 *
 *  This version of \p sort_by_key compares key objects using \c operator<.
 *
 *  \param keys_first The beginning of the key sequence.
 *  \param keys_last The end of the key sequence.
 *  \param values_first The beginning of the value sequence.
 *
 *  \tparam RandomAccessIterator1 is a model of <a href="http://www.sgi.com/tech/stl/RandomAccessIterator.html">Random Access Iterator</a>,
 *          \p RandomAccessIterator1 is mutable,
 *          and \p RandomAccessIterator1's \c value_type is a model of <a href="http://www.sgi.com/tech/stl/LessThanComparable.html">LessThan Comparable</a>,
 *          and the ordering relation on \p RandomAccessIterator1's \c value_type is a <em>strict weak ordering</em>, as defined in the
 *          <a href="http://www.sgi.com/tech/stl/LessThanComparable.html">LessThan Comparable</a> requirements.
 *  \tparam RandomAccessIterator2 is a model of <a href="http://www.sgi.com/tech/stl/RandomAccessIterator.hml">Random Access Iterator</a>,
 *          and \p RandomAccessIterator2 is mutable.
 *
 *  The following code snippet demonstrates how to use \p sort_by_key to sort
 *  an array of characters using integers as sorting keys.
 *
 *  \code
 *  #include <thrust/sort.h>
 *  ...
 *  const int N = 6;
 *  int    keys[N] = {  1,   4,   2,   8,   5,   7};
 *  char values[N] = {'a', 'b', 'c', 'd', 'e', 'f'};
 *  thrust::sort_by_key(keys, keys + N, values);
 *  // keys is now   {  1,   2,   4,   5,   7,   8}
 *  // values is now {'a', 'c', 'b', 'e', 'f', 'd'}
 *  \endcode
 *
 *  \see http://www.sgi.com/tech/stl/sort.html
 *  \see \p stable_sort_by_key
 *  \see \p sort
 */
template<typename RandomAccessIterator1,
         typename RandomAccessIterator2>
  void sort_by_key(RandomAccessIterator1 keys_first,
                   RandomAccessIterator1 keys_last,
                   RandomAccessIterator2 values_first);

/*! \p sort_by_key performs a key-value sort. That is, \p sort_by_key sorts the
 *  elements in <tt>[keys_first, keys_last)</tt> and <tt>[values_first,
 *  values_first + (keys_last - keys_first))</tt> into ascending key order,
 *  meaning that if \c i and \c j are any two valid iterators in <tt>[keys_first,
 *  keys_last)</tt> such that \c i precedes \c j, and \c p and \c q are iterators
 *  in <tt>[values_first, values_first + (keys_last - keys_first))</tt>
 *  corresponding to \c i and \c j respectively, then \c *j is not less than
 *  \c *i.
 *
 *  Note: \c sort_by_key is not guaranteed to be stable. That is, suppose that
 *  \c *i and \c *j are equivalent: neither one is less than the other. It is not
 *  guaranteed that the relative order of these two keys or the relative
 *  order of their corresponding values will be preserved by \p sort_by_key.
 *
 *  This version of \p sort_by_key compares key objects using a function object
 *  \c comp.
 *
 *  \param keys_first The beginning of the key sequence.
 *  \param keys_last The end of the key sequence.
 *  \param values_first The beginning of the value sequence.
 *  \param comp Comparison operator.
 *
 *  \tparam RandomAccessIterator1 is a model of <a href="http://www.sgi.com/tech/stl/RandomAccessIterator.html">Random Access Iterator</a>,
 *          \p RandomAccessIterator1 is mutable,
 *          and \p RandomAccessIterator1's \c value_type is convertible to \p StrictWeakOrdering's
 *          \c first_argument_type and \c second_argument_type.
 *  \tparam RandomAccessIterator2 is a model of <a href="http://www.sgi.com/tech/stl/RandomAccessIterator.hml">Random Access Iterator</a>,
 *          and \p RandomAccessIterator2 is mutable.
 *  \tparam StrictWeakOrdering is a model of <a href="http://www.sgi.com/tech/stl/StrictWeakOrdering.html">Strict Weak Ordering</a>.
 *
 *  \see http://www.sgi.com/tech/stl/sort.html
 *  \see \p stable_sort_by_key
 *  \see \p sort
 */
template<typename RandomAccessKeyIterator,
         typename RandomAccessValueIterator,
         typename StrictWeakOrdering>
  void sort_by_key(RandomAccessKeyIterator keys_first,
                   RandomAccessKeyIterator keys_last,
                   RandomAccessValueIterator values_first,
                   StrictWeakOrdering comp);

/*! \p stable_sort_by_key performs a key-value sort. That is, \p stable_sort_by_key
 *  sorts the elements in <tt>[keys_first, keys_last)</tt> and <tt>[values_first,
 *  values_first + (keys_last - keys_first))</tt> into ascending key order,
 *  meaning that if \c i and \c j are any two valid iterators in <tt>[keys_first,
 *  keys_last)</tt> such that \c i precedes \c j, and \c p and \c q are iterators
 *  in <tt>[values_first, values_first + (keys_last - keys_first))</tt>
 *  corresponding to \c i and \c j respectively, then \c *j is not less than
 *  \c *i.
 *
 *  As the name suggests, \p stable_sort_by_key is stable: it preserves the
 *  relative ordering of equivalent elements. That is, if \c x and \c y
 *  are elements in <tt>[keys_first, keys_last)</tt> such that \c x precedes \c y,
 *  and if the two elements are equivalent (neither <tt>x < y</tt> nor
 *  <tt>y < x</tt>) then a postcondition of \p stable_sort_by_key is that \c x
 *  still precedes \c y.
 *
 *  This version of \p stable_sort_by_key compares key objects using \c operator<.
 *
 *  \param keys_first The beginning of the key sequence.
 *  \param keys_last The end of the key sequence.
 *  \param values_first The beginning of the value sequence.
 *
 *  \tparam RandomAccessIterator1 is a model of <a href="http://www.sgi.com/tech/stl/RandomAccessIterator.html">Random Access Iterator</a>,
 *          \p RandomAccessIterator1 is mutable,
 *          and \p RandomAccessIterator1's \c value_type is a model of <a href="http://www.sgi.com/tech/stl/LessThanComparable.html">LessThan Comparable</a>,
 *          and the ordering relation on \p RandomAccessIterator1's \c value_type is a <em>strict weak ordering</em>, as defined in the
 *          <a href="http://www.sgi.com/tech/stl/LessThanComparable.html">LessThan Comparable</a> requirements.
 *  \tparam RandomAccessIterator2 is a model of <a href="http://www.sgi.com/tech/stl/RandomAccessIterator.hml">Random Access Iterator</a>,
 *          and \p RandomAccessIterator2 is mutable.
 *
 *  The following code snippet demonstrates how to use \p stable_sort_by_key to sort
 *  an array of characters using integers as sorting keys.
 *
 *  \code
 *  #include <thrust/sort.h>
 *  ...
 *  const int N = 6;
 *  int    keys[N] = {  1,   4,   2,   8,   5,   7};
 *  char values[N] = {'a', 'b', 'c', 'd', 'e', 'f'};
 *  thrust::stable_sort_by_key(keys, keys + N, values);
 *  // keys is now   {  1,   2,   4,   5,   7,   8}
 *  // values is now {'a', 'c', 'b', 'e', 'f', 'd'}
 *  \endcode
 *
 *  \see http://www.sgi.com/tech/stl/sort.html
 *  \see \p sort_by_key
 *  \see \p stable_sort
 */
template<typename RandomAccessKeyIterator,
         typename RandomAccessValueIterator>
  void stable_sort_by_key(RandomAccessKeyIterator keys_first,
                          RandomAccessKeyIterator keys_last,
                          RandomAccessValueIterator values_first);

/*! \p stable_sort_by_key performs a key-value sort. That is, \p stable_sort_by_key
 *  sorts the elements in <tt>[keys_first, keys_last)</tt> and <tt>[values_first,
 *  values_first + (keys_last - keys_first))</tt> into ascending key order,
 *  meaning that if \c i and \c j are any two valid iterators in <tt>[keys_first,
 *  keys_last)</tt> such that \c i precedes \c j, and \c p and \c q are iterators
 *  in <tt>[values_first, values_first + (keys_last - keys_first))</tt>
 *  corresponding to \c i and \c j respectively, then \c *j is not less than
 *  \c *i.
 *
 *  As the name suggests, \p stable_sort_by_key is stable: it preserves the
 *  relative ordering of equivalent elements. That is, if \c x and \c y
 *  are elements in <tt>[keys_first, keys_last)</tt> such that \c x precedes \c y,
 *  and if the two elements are equivalent (neither <tt>x < y</tt> nor
 *  <tt>y < x</tt>) then a postcondition of \p stable_sort_by_key is that \c x
 *  still precedes \c y.
 *
 *  This version of \p stable_sort_by_key compares key objects using the function
 *  object \p comp.
 *
 *  \param keys_first The beginning of the key sequence.
 *  \param keys_last The end of the key sequence.
 *  \param values_first The beginning of the value sequence.
 *  \param comp Comparison operator.
 *
 *  \tparam RandomAccessIterator1 is a model of <a href="http://www.sgi.com/tech/stl/RandomAccessIterator.html">Random Access Iterator</a>,
 *          \p RandomAccessIterator1 is mutable,
 *          and \p RandomAccessIterator1's \c value_type is convertible to \p StrictWeakOrdering's
 *          \c first_argument_type and \c second_argument_type.
 *  \tparam RandomAccessIterator2 is a model of <a href="http://www.sgi.com/tech/stl/RandomAccessIterator.hml">Random Access Iterator</a>,
 *          and \p RandomAccessIterator2 is mutable.
 *  \tparam StrictWeakOrdering is a model of <a href="http://www.sgi.com/tech/stl/StrictWeakOrdering.html">Strict Weak Ordering</a>.
 *
 *  \see http://www.sgi.com/tech/stl/sort.html
 *  \see \p sort_by_key
 *  \see \p stable_sort
 */
template<typename RandomAccessKeyIterator,
         typename RandomAccessValueIterator,
         typename StrictWeakOrdering>
  void stable_sort_by_key(RandomAccessKeyIterator keys_first,
                          RandomAccessKeyIterator keys_last,
                          RandomAccessValueIterator values_first,
                          StrictWeakOrdering comp);

/*! \} // end sorting
 */

} // end namespace thrust

#include <thrust/detail/sort.inl>

