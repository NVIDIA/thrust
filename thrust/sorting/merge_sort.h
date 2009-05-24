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


/*! \file merge_sort.h
 *  \brief Defines the interface to merge sort functions.
 */

#pragma once

#include <thrust/detail/config.h>

namespace thrust
{

/*! thrust::sorting encapsulates different sorting algorithms.
 */
namespace sorting
{

/*! \addtogroup sorting
 *  \ingroup reordering
 *
 *  \addtogroup key_sorting Key Sorting
 *  \ingroup sorting
 *  \addtogroup merge_sorting Merge Sorting
 *  \ingroup key_sorting
 *  \{
 */

/*! \p merge_sort sorts the elements in <tt>[first, last)</tt> into
 *  ascending order, meaning that if \c i and \c j are any two valid
 *  iterators in <tt>[first, last)</tt> such that \c i precedes \c j,
 *  then \c *j is not less than \c *i. Note: \c merge_sort is not guaranteed
 *  to be stable. That is, suppose that \c *i and \c *j are equivalent:
 *  neither one is less than the other. It is not guaranteed that the
 *  relative order of these two elements will be preserved by
 *  \p merge_sort.
 *
 *  This version of \p merge_sort compares objects using \c operator<.
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
 *  The following code snippet demonstrates how to use \p merge_sort to sort
 *  a sequence of integers.
 *
 *  \code
 *  #include <thrust/sorting/merge_sort.h>
 *  ...
 *  const int N = 6;
 *  int A[N] = {1, 4, 2, 8, 5, 7};
 *  thrust::sorting::merge_sort(A, A + N);
 *  // A is now {1, 2, 4, 5, 7, 8}
 *  \endcode
 *
 *  \see \p sort
 *  \see \p stable_merge_sort
 *  \see \p merge_sort_by_key
 */
template<typename RandomAccessIterator>
  void merge_sort(RandomAccessIterator first,
                  RandomAccessIterator last);

/*! \p merge_sort sorts the elements in <tt>[first, last)</tt> into
 *  ascending order, meaning that if \c i and \c j are any two valid
 *  iterators in <tt>[first, last)</tt> such that \c i precedes \c j,
 *  then \c *j is not less than \c *i. Note: \c merge_sort is not guaranteed
 *  to be stable. That is, suppose that \c *i and \c *j are equivalent:
 *  neither one is less than the other. It is not guaranteed that the
 *  relative order of these two elements will be preserved by
 *  \p merge_sort.
 *
 *  This version of \p merge_sort compares objects using a function object
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
 *  \see sort
 *  \see \p stable_merge_sort
 *  \see \p merge_sort_by_key
 */
template<typename RandomAccessIterator,
         typename StrictWeakOrdering>
  void merge_sort(RandomAccessIterator first,
                  RandomAccessIterator last,
                  StrictWeakOrdering comp);

/*! \p stable_merge_sort is much like \c merge_sort: it sorts the elements in
 *  <tt>[first, last)</tt> into ascending order, meaning that if \c i
 *  and \c j are any two valid iterators in <tt>[first, last)</tt> such
 *  that \c i precedes \c j, then \c *j is not less than \c *i.
 *
 *  As the name suggests, \p stable_merge_sort is stable: it preserves the
 *  relative ordering of equivalent elements. That is, if \c x and \c y
 *  are elements in <tt>[first, last)</tt> such that \c x precedes \c y,
 *  and if the two elements are equivalent (neither <tt>x < y</tt> nor
 *  <tt>y < x</tt>) then a postcondition of \p stable_merge_sort is that \c x
 *  still precedes \c y.
 *
 *  This version of \p stable_merge_sort compares objects using \c operator<.
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
 *  The following code snippet demonstrates how to use \p stable_merge_sort to sort
 *  a sequence of integers.
 *
 *  \code
 *  #include <thrust/sorting/merge_sort.h>
 *  ...
 *  const int N = 6;
 *  int A[N] = {1, 4, 2, 8, 5, 7};
 *  thrust::sorting::stable_merge_sort(A, A + N);
 *  // A is now {1, 2, 4, 5, 7, 8}
 *  \endcode
 *
 *  \see \p stable_sort
 *  \see \p merge_sort
 *  \see \p stable_stable_sort_by_key
 */
template<typename RandomAccessIterator>
  void stable_merge_sort(RandomAccessIterator first,
                         RandomAccessIterator last);

/*! \p stable_merge_sort is much like \c merge_sort: it sorts the elements in
 *  <tt>[first, last)</tt> into ascending order, meaning that if \c i
 *  and \c j are any two valid iterators in <tt>[first, last)</tt> such
 *  that \c i precedes \c j, then \c *j is not less than \c *i.
 *
 *  As the name suggests, \p stable_merge_sort is stable: it preserves the
 *  relative ordering of equivalent elements. That is, if \c x and \c y
 *  are elements in <tt>[first, last)</tt> such that \c x precedes \c y,
 *  and if the two elements are equivalent (neither <tt>x < y</tt> nor
 *  <tt>y < x</tt>) then a postcondition of \p stable_merge_sort is that \c x
 *  still precedes \c y.
 *
 *  This version of \p stable_merge_sort compares objects using a function object
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
 *  \see stable_sort
 *  \see \p merge_sort
 *  \see \p stable_merge_sort_by_key
 */
template<typename RandomAccessIterator,
         typename StrictWeakOrdering>
  void stable_merge_sort(RandomAccessIterator first,
                         RandomAccessIterator last,
                         StrictWeakOrdering comp);

/*! \}
 */

/*! \addtogroup key_value_sorting Key-Value Sorting
 *  \ingroup sorting
 *  \addtogroup merge_sorting Merge Sorting
 *  \ingroup key_value_sorting
 *  \{
 */

/*! \p merge_sort_by_key performs a key-value sort. That is, \p merge_sort_by_key sorts the
 *  elements in <tt>[keys_first, keys_last)</tt> and <tt>[values_first,
 *  values_first + (keys_last - keys_first))</tt> into ascending key order,
 *  meaning that if \c i and \c j are any two valid iterators in <tt>[keys_first,
 *  keys_last)</tt> such that \c i precedes \c j, and \c p and \c q are iterators
 *  in <tt>[values_first, values_first + (keys_last - keys_first))</tt>
 *  corresponding to \c i and \c j respectively, then \c *j is not less than
 *  \c *i.
 *
 *  Note: \c merge_sort_by_key is not guaranteed to be stable. That is, suppose that
 *  \c *i and \c *j are equivalent: neither one is less than the other. It is not
 *  guaranteed that the relative order of these two keys or the relative
 *  order of their corresponding values will be preserved by \p merge_sort_by_key.
 *
 *  This version of \p merge_sort_by_key compares key objects using \c operator<.
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
 *  The following code snippet demonstrates how to use \p merge_sort_by_key to sort
 *  an array of characters using integers as sorting keys.
 *
 *  \code
 *  #include <thrust/sorting/merge_sort.h>
 *  ...
 *  const int N = 6;
 *  int    keys[N] = {  1,   4,   2,   8,   5,   7};
 *  char values[N] = {'a', 'b', 'c', 'd', 'e', 'f'};
 *  thrust::sorting::merge_sort_by_key(keys, keys + N, values);
 *  // keys is now   {  1,   2,   4,   5,   7,   8}
 *  // values is now {'a', 'c', 'b', 'e', 'f', 'd'}
 *  \endcode
 *
 *  \see sort_by_key
 *  \see merge_sort
 *  \see \p stable_merge_sort_by_key
 */
template<typename RandomAccessKeyIterator,
         typename RandomAccessValueIterator>
  void merge_sort_by_key(RandomAccessKeyIterator keys_first,
                         RandomAccessKeyIterator keys_last,
                         RandomAccessValueIterator values_first);

/*! \p merge_sort_by_key performs a key-value sort. That is, \p merge_sort_by_key sorts the
 *  elements in <tt>[keys_first, keys_last)</tt> and <tt>[values_first,
 *  values_first + (keys_last - keys_first))</tt> into ascending key order,
 *  meaning that if \c i and \c j are any two valid iterators in <tt>[keys_first,
 *  keys_last)</tt> such that \c i precedes \c j, and \c p and \c q are iterators
 *  in <tt>[values_first, values_first + (keys_last - keys_first))</tt>
 *  corresponding to \c i and \c j respectively, then \c *j is not less than
 *  \c *i.
 *
 *  Note: \c merge_sort_by_key is not guaranteed to be stable. That is, suppose that
 *  \c *i and \c *j are equivalent: neither one is less than the other. It is not
 *  guaranteed that the relative order of these two keys or the relative
 *  order of their corresponding values will be preserved by \p merge_sort_by_key.
 *
 *  This version of \p merge_sort_by_key compares key objects using a function object
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
 *  \see sort_by_key
 *  \see merge_sort
 *  \see \p stable_merge_sort_by_key
 */
template<typename RandomAccessKeyIterator,
         typename RandomAccessValueIterator,
         typename StrictWeakOrdering>
  void merge_sort_by_key(RandomAccessKeyIterator keys_first,
                         RandomAccessKeyIterator keys_last,
                         RandomAccessValueIterator values_first,
                         StrictWeakOrdering comp);

/*! \p stable_merge_sort_by_key performs a key-value sort. That is, \p stable_merge_sort_by_key
 *  sorts the elements in <tt>[keys_first, keys_last)</tt> and <tt>[values_first,
 *  values_first + (keys_last - keys_first))</tt> into ascending key order,
 *  meaning that if \c i and \c j are any two valid iterators in <tt>[keys_first,
 *  keys_last)</tt> such that \c i precedes \c j, and \c p and \c q are iterators
 *  in <tt>[values_first, values_first + (keys_last - keys_first))</tt>
 *  corresponding to \c i and \c j respectively, then \c *j is not less than
 *  \c *i.
 *
 *  As the name suggests, \p stable_merge_sort_by_key is stable: it preserves the
 *  relative ordering of equivalent elements. That is, if \c x and \c y
 *  are elements in <tt>[keys_first, keys_last)</tt> such that \c x precedes \c y,
 *  and if the two elements are equivalent (neither <tt>x < y</tt> nor
 *  <tt>y < x</tt>) then a postcondition of \p stable_merge_sort_by_key is that \c x
 *  still precedes \c y.
 *
 *  This version of \p stable_merge_sort_by_key compares key objects using \c operator<.
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
 *  The following code snippet demonstrates how to use \p stable_merge_sort_by_key to sort
 *  an array of characters using integers as sorting keys.
 *
 *  \code
 *  #include <thrust/sorting/merge_sort.h>
 *  ...
 *  const int N = 6;
 *  int    keys[N] = {  1,   4,   2,   8,   5,   7};
 *  char values[N] = {'a', 'b', 'c', 'd', 'e', 'f'};
 *  thrust::sorting::stable_merge_sort_by_key(keys, keys + N, values);
 *  // keys is now   {  1,   2,   4,   5,   7,   8}
 *  // values is now {'a', 'c', 'b', 'e', 'f', 'd'}
 *  \endcode
 *
 *  \see stable_sort_by_key
 *  \see \p merge_sort_by_key
 *  \see \p stable_merge_sort
 */
template<typename RandomAccessKeyIterator,
         typename RandomAccessValueIterator>
  void stable_merge_sort_by_key(RandomAccessKeyIterator keys_first,
                                RandomAccessKeyIterator keys_last,
                                RandomAccessValueIterator values_first);

/*! \p stable_merge_sort_by_key performs a key-value sort. That is, \p stable_merge_sort_by_key
 *  sorts the elements in <tt>[keys_first, keys_last)</tt> and <tt>[values_first,
 *  values_first + (keys_last - keys_first))</tt> into ascending key order,
 *  meaning that if \c i and \c j are any two valid iterators in <tt>[keys_first,
 *  keys_last)</tt> such that \c i precedes \c j, and \c p and \c q are iterators
 *  in <tt>[values_first, values_first + (keys_last - keys_first))</tt>
 *  corresponding to \c i and \c j respectively, then \c *j is not less than
 *  \c *i.
 *
 *  As the name suggests, \p stable_merge_sort_by_key is stable: it preserves the
 *  relative ordering of equivalent elements. That is, if \c x and \c y
 *  are elements in <tt>[keys_first, keys_last)</tt> such that \c x precedes \c y,
 *  and if the two elements are equivalent (neither <tt>x < y</tt> nor
 *  <tt>y < x</tt>) then a postcondition of \p stable_merge_sort_by_key is that \c x
 *  still precedes \c y.
 *
 *  This version of \p stable_merge_sort_by_key compares key objects using the function
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
 *  \see stable_sort_by_key
 *  \see \p merge_sort_by_key
 *  \see \p stable_merge_sort
 */
template<typename RandomAccessKeyIterator,
         typename RandomAccessValueIterator,
         typename StrictWeakOrdering>
  void stable_merge_sort_by_key(RandomAccessKeyIterator keys_first,
                                RandomAccessKeyIterator keys_last,
                                RandomAccessValueIterator values_first,
                                StrictWeakOrdering comp);

/*! \}
 */

} // end namespace sorting

} // end namespace thrust

#include <thrust/sorting/detail/merge_sort.inl>

