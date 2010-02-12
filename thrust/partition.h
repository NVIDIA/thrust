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


/*! \file partition.h
 *  \brief Defines the interface to a function performing
 *         a stream compaction computation.
 */

#pragma once

#include <thrust/detail/config.h>

namespace thrust
{

/*! \addtogroup reordering
 *  \ingroup algorithms
 *
 *  \addtogroup partitioning
 *  \ingroup reordering
 *  \{
 */

/*! \p partition reorders the elements <tt>[first, last)</tt> based on the function
 *  object \p pred, such that all of the elements that satisfy \p pred precede the
 *  elements that fail to satisfy it. The postcondition is that, for some iterator
 *  \c middle in the range <tt>[first, last)</tt>, <tt>pred(*i)</tt> is \c true for every
 *  iterator \c i in the range <tt>[first,middle)</tt> and \c false for every iterator
 *  \c i in the range <tt>[middle, last)</tt>. The return value of \p partition is
 *  \c middle.
 *
 *  Note that the relative order of elements in the two reordered sequences is not
 *  necessarily the same as it was in the original sequence. A different algorithm,
 *  \ref stable_partition, does guarantee to preserve the relative order.
 *
 *  \param first The beginning of the sequence to reorder.
 *  \param last The end of the sequence to reorder.
 *  \param pred A function object which decides to which partition each element of the
 *              sequence <tt>[first, last)</tt> belongs.
 *  \return An iterator referring to the first element of the second partition, that is,
 *          the sequence of the elements which do not satisfy \p pred.
 *
 *  \tparam ForwardIterator is a model of <a href="http://www.sgi.com/tech/stl/ForwardIterator.html">Forward Iterator</a>,
 *          and \p ForwardIterator's \c value_type is convertible to \p Predicate's \c argument_type,
 *          and \p ForwardIterator is mutable.
 *  \tparam Predicate is a model of <a href="http://www.sgi.com/tech/stl/Predicate.html">Predicate</a>.
 *
 *  The following code snippet demonstrates how to use \p partition to reorder a
 *  sequence so that even numbers precede odd numbers.
 *
 *  \code
 *  #include <thrust/partition.h>
 *  ...
 *  struct is_even
 *  {
 *    __host__ __device__
 *    bool operator()(const int &x)
 *    {
 *      return (x % 2) == 0;
 *    }
 *  };
 *  ...
 *  int A[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
 *  const int N = sizeof(A)/sizeof(int);
 *  thrust::partition(A, A + N,
 *                     is_even());
 *  // A is now {2, 4, 6, 8, 10, 1, 3, 5, 7, 9}
 *  \endcode
 *
 *  \see http://www.sgi.com/tech/stl/partition.html
 *  \see \p stable_partition
 *  \see \p partition_copy
 */
template<typename ForwardIterator,
         typename Predicate>
  ForwardIterator partition(ForwardIterator first,
                            ForwardIterator last,
                            Predicate pred);

namespace experimental
{
/*! \p partition_copy differs from \ref partition only in that the reordered
 *  sequence is written to a difference output sequence, rather than in place.
 *
 *  \p partition_copy reorders the elements <tt>[first, last)</tt> based on the
 *  function object \p pred, such that all of the elements that satisfy \p pred precede the
 *  elements that fail to satisfy it. The postcondition is that, for some iterator
 *  \c middle in the range <tt>[first, last)</tt>, <tt>pred(*i)</tt> is \c true for every
 *  iterator \c i in the range <tt>[first,middle)</tt> and \c false for every iterator
 *  \c i in the range <tt>[middle, last)</tt>. The return value of \p partition_copy is
 *  \c middle.
 *
 *  \param first The beginning of the sequence to reorder.
 *  \param last The end of the sequence to reorder.
 *  \param result The destination of the resulting sequence.
 *  \param pred A function object which decides to which partition each element of the
 *              sequence <tt>[first, last)</tt> belongs.
 *  \return An iterator referring to the first element of the second partition, that is,
 *          the sequence of the elements which do not satisfy \p pred.
 *
 *  \tparam ForwardIterator1 is a model of <a href="http://www.sgi.com/tech/stl/ForwardIterator.html">Forward Iterator</a>,
 *          and \p ForwardIterator's \c value_type is convertible to \p Predicate's \c argument_type.
 *  \tparam ForwardIterator2 is a model of <a href="http://www.sgi.com/tech/stl/ForwardIterator.html">Forward Iterator</a>.
 *  \tparam Predicate is a model of <a href="http://www.sgi.com/tech/stl/Predicate.html">Predicate</a>.
 *
 *  The following code snippet demonstrates how to use \p partition_copy to reorder a
 *  sequence so that even numbers precede odd numbers.
 *
 *  \code
 *  #include <thrust/partition.h>
 *  ...
 *  struct is_even
 *  {
 *    __host__ __device__
 *    bool operator()(const int &x)
 *    {
 *      return (x % 2) == 0;
 *    }
 *  };
 *  ...
 *  int A[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
 *  int result[10];
 *  const int N = sizeof(A)/sizeof(int);
 *  thrust::partition_copy(A, A + N, result,
 *                          is_even());
 *  // A remains {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
 *  // result is now {2, 4, 6, 8, 10, 1, 3, 5, 7, 9}
 *  \endcode
 *
 *  \note The relative order of elements in the two reordered sequences is not
 *  necessarily the same as it was in the original sequence. A different algorithm,
 *  \ref stable_partition_copy, does guarantee to preserve the relative order.
 *
 *  \note \p partition_copy's interface differs from the proposed C++ STL function
 *  <a href="http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2569.pdf">std::partition_copy</a>
 *  due to the absence of a priori knowledge about the size of the two resulting sequences.
 *
 *  \see http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2569.pdf
 *  \see \p stable_partition_copy
 *  \see \p partition
 */
template<typename ForwardIterator1,
         typename ForwardIterator2,
         typename Predicate>
  ForwardIterator2 partition_copy(ForwardIterator1 first,
                                  ForwardIterator1 last,
                                  ForwardIterator2 result,
                                  Predicate pred);

} // end namespace experimental

/*! \p stable_partition is much like \ref partition : it reorders the elements in the
 *  range <tt>[first, last)</tt> based on the function object \p pred, such that all of
 *  the elements that satisfy \p pred precede all of the elements that fail to satisfy
 *  it. The postcondition is that, for some iterator \p middle in the range
 *  <tt>[first, last)</tt>, <tt>pred(*i)</tt> is \c true for every iterator \c i in the
 *  range <tt>[first,middle)</tt> and \c false for every iterator \c i in the range
 *  <tt>[middle, last)</tt>. The return value of \p stable_partition is \c middle.
 *
 *  \p stable_partition differs from \ref partition in that \p stable_partition is
 *  guaranteed to preserve relative order. That is, if \c x and \c y are elements in
 *  <tt>[first, last)</tt>, such that <tt>pred(x) == pred(y)</tt>, and if \c x precedes
 *  \c y, then it will still be true after \p stable_partition that \c x precedes \c y.
 *
 *  \param first The first element of the sequence to reorder.
 *  \param last One position past the last element of the sequence to reorder.
 *  \param pred A function object which decides to which partition each element of the
 *              sequence <tt>[first, last)</tt> belongs.
 *  \return An iterator referring to the first element of the second partition, that is,
 *          the sequence of the elements which do not satisfy pred.
 *
 *  \tparam ForwardIterator is a model of <a href="http://www.sgi.com/tech/stl/ForwardIterator.html">Forward Iterator</a>,
 *          and \p ForwardIterator's \c value_type is convertible to \p Predicate's \c argument_type,
 *          and \p ForwardIterator is mutable.
 *  \tparam Predicate is a model of <a href="http://www.sgi.com/tech/stl/Predicate.html">Predicate</a>.
 *
 *  The following code snippet demonstrates how to use \p stable_partition to reorder a
 *  sequence so that even numbers precede odd numbers.
 *
 *  \code
 *  #include <thrust/partition.h>
 *  ...
 *  struct is_even
 *  {
 *    __host__ __device__
 *    bool operator()(const int &x)
 *    {
 *      return (x % 2) == 0;
 *    }
 *  };
 *  ...
 *  int A[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
 *  const int N = sizeof(A)/sizeof(int);
 *  thrust::stable_partition(A, A + N,
 *                            is_even());
 *  // A is now {2, 4, 6, 8, 10, 1, 3, 5, 7, 9}
 *  \endcode
 *
 *  \see http://www.sgi.com/tech/stl/stable_partition.html
 *  \see \p partition
 *  \see \p stable_partition_copy
 */
template<typename ForwardIterator,
         typename Predicate>
  ForwardIterator stable_partition(ForwardIterator first,
                                   ForwardIterator last,
                                   Predicate pred);


namespace experimental
{
/*! \p stable_partition_copy is much like \ref partition_copy : it reorders the elements
 *  in the range <tt>[first, last)</tt> based on the function object \p pred, such that
 *  all of the elements that satisfy \p pred precede all of the elements that fail to
 *  satisfy it. The postcondition is that, for some iterator \p middle in the range
 *  <tt>[first, last)</tt>, <tt>pred(*i)</tt> is \c true for every iterator \c i in the
 *  range <tt>[first,middle)</tt> and \c false for every iterator \c i in the range
 *  <tt>[middle, last)</tt>. The return value of \p stable_partition_copy is \c middle.
 *
 *  \p stable_partition_copy differs from \ref partition_copy in that
 *  \p stable_partition_copy is guaranteed to preserve relative order. That is, if
 *  \c x and \c y are elements in <tt>[first, last)</tt>, such that
 *  <tt>pred(x) == pred(y)</tt>, and if \c x precedes \c y, then it will still be true
 *  after \p stable_partition_copy that \c x precedes \c y.
 *
 *  \param first The first element of the sequence to reorder.
 *  \param last One position past the last element of the sequence to reorder.
 *  \param result The destination of the resulting sequence.
 *  \param pred A function object which decides to which partition each element of the
 *              sequence [first, last) belongs.
 *  \return An iterator referring to the first element of the second partition, that is,
 *          the sequence of the elements which do not satisfy pred.
 *
 *  \tparam ForwardIterator1 is a model of <a href="http://www.sgi.com/tech/stl/ForwardIterator.html">Forward Iterator</a>,
 *          and \p ForwardIterator1's \c value_type is convertible to \p Predicate's \c argument_type.
 *  \tparam ForwardIterator2 is a model of <a href="http://www.sgi.com/tech/stl/ForwardIterator.html">Forward Iterator</a>,
 *          and \p ForwardIterator2 is mutable.
 *  \tparam Predicate is a model of <a href="http://www.sgi.com/tech/stl/Predicate.html">Predicate</a>.
 *
 *  The following code snippet demonstrates how to use \p stable_partition_copy to
 *  reorder a sequence so that even numbers precede odd numbers.
 *
 *  \code
 *  #include <thrust/partition.h>
 *  ...
 *  struct is_even
 *  {
 *    __host__ __device__
 *    bool operator()(const int &x)
 *    {
 *      return (x % 2) == 0;
 *    }
 *  };
 *  ...
 *  int A[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
 *  int result[10];
 *  const int N = sizeof(A)/sizeof(int);
 *  thrust::stable_partition_copy(A, A + N, result,
 *                                 is_even());
 *  // A remains {1, 2, 3, 4, 5, 6, 7, 8, 8, 10}
 *  // result is now {2, 4, 6, 8, 10, 1, 3, 5, 7, 9}
 *  \endcode
 *
 *  \see \p partition_copy
 *  \see \p stable_partition
 */
template<typename ForwardIterator1,
         typename ForwardIterator2,
         typename Predicate>
  ForwardIterator2 stable_partition_copy(ForwardIterator1 first,
                                         ForwardIterator1 last,
                                         ForwardIterator2 result,
                                         Predicate pred);

} // end namespace experimental

/*! \} // end stream_compaction
 */

/*! \} // end reordering
 */

} // end thrust

#include <thrust/detail/partition.inl>

