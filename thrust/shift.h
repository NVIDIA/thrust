/*
 *  Copyright 2022 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a shift_left of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */


/*! \file thrust/shift.h
 *  \brief Shifts the elements of a range to the front / end
 */

#pragma once

#include <thrust/detail/config.h>
#include <thrust/detail/execution_policy.h>
#include <thrust/iterator/iterator_traits.h>

THRUST_NAMESPACE_BEGIN

/*! \addtogroup algorithms
 */

/*! \addtogroup stream_compaction
 *  \{
 */


/*! \p shift_left shifts the elements from the range [\p first, \p last) to the left. That is, it performs
 *  the assignments *\p first = *(\p first + \p n), *(\p first + \c 1) = *(\p first + \p n + \c 1),
 *  and so on. Generally, for every integer \c m from \c 0 to \p last - \p first - n, \p shift_left
 *  performs the assignment *(\p first + \c m) = *(\p first + \c m + \p n). Unlike \c std::shift_left,
 *  \p shift_left offers no guarantee on order of operation. Elements that are in the original range but
 *  not the new range are left in a valid but unspecified state. If \p n is equal to zero or greater than
 *  (\p last - \p first) the algorithm has no effect.
 *
 *  The return value is the end of the resulting range \p first + (\p last - \p first - \p n).
 *
 *  The algorithm's execution is parallelized as determined by \p exec.
 *
 *  \param exec The execution policy to use for parallelization.
 *  \param first The beginning of the sequence to shift.
 *  \param last The end of the sequence to shift.
 *  \param n The number of positions to shift.
 *  \return The end of the resulting sequence.
 *  \see https://en.cppreference.com/w/cpp/algorithm/shift
 *
 *  \tparam DerivedPolicy The name of the derived execution policy.
 *  \tparam ForwardIterator must be a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a>.
 *
 *  \pre \p n may be equal to equal to zero or greater than (\p last - \p first), but \p n shall not be negative
 *
 *
 *  The following code snippet demonstrates how to use \p shift_left
 *  on a range using the \p thrust::device parallelization policy:
 *
 *  \code
 *  #include <thrust/shift.h>
 *  #include <thrust/device_vector.h>
 *  #include <thrust/execution_policy.h>
 *  ...
 *
 *  thrust::device_vector<int> vec = {1, 2, 3, 4, 5, 6};
 *  ...
 *
 *  thrust::shift_left(thrust::device, vec.begin(), vec.end(), 2);
 *
 *  // vec is now equal to {3, 4, 5, 6, unspecified, unspecified }
 *  \endcode
 */
template <typename DerivedPolicy, typename ForwardIterator>
__host__ __device__
  ForwardIterator shift_left(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                             ForwardIterator first,
                             ForwardIterator last,
                             typename thrust::iterator_traits<ForwardIterator>::difference_type n);


/*! \p shift_left shifts the elements from the range [\p first, \p last) to the left. That is, it performs
 *  the assignments *\p first = *(\p first + \p n), *(\p first + \c 1) = *(\p first + \p n + \c 1),
 *  and so on. Generally, for every integer \c m from \c 0 to \p last - \p first - n, \p shift_left
 *  performs the assignment *(\p first + \c m) = *(\p first + \c m + \p n). Elements that are in the
 *  original range but not the new range are left in a valid but unspecified state. If \p n is equal
 *  to zero or greater than (\p last - \p first) the algorithm has no effect.
 *
 *  The return value is the end of the resulting range \p first + (\p last - \p first - \p n).
 *
 *  \param first The beginning of the sequence to shift.
 *  \param last The end of the sequence to shift.
 *  \param n The number of positions to shift.
 *  \return The end of the resulting sequence.
 *  \see https://en.cppreference.com/w/cpp/algorithm/shift
 *
 *  \tparam ForwardIterator must be a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a>.
 *
 *  \pre \p n may be equal to equal to zero or greater than (\p last - \p first), but \p n shall not be negative
 *
 *  The following code snippet demonstrates how to use \p shift_left on a range:
 *
 *  \code
 *  #include <thrust/shift.h>
 *  #include <thrust/device_vector.h>
 *  ...
 *
 *  thrust::device_vector<int> vec = {1, 2, 3, 4, 5, 6};
 *  ...
 *
 *  thrust::shift_left(vec.begin(), vec.end(), 2);
 *
 *  // vec is now equal to {3, 4, 5, 6, unspecified, unspecified }
 *  \endcode
 */
template <typename ForwardIterator>
  ForwardIterator shift_left(ForwardIterator first,
                             ForwardIterator last,
                             typename thrust::iterator_traits<ForwardIterator>::difference_type n);


/*! \p shift_right shifts the elements from the range [\p first, \p last) to the right. That is, it performs
 *  the assignments *(\p first + \p n) = *\p first, *(\p first + \p n + \c 1) = *(\p first + \c 1),
 *  and so on. Generally, for every integer \c m from \c 0 to \p last - \p first - n, \p shift_right
 *  performs the assignment *(\p first + \c m + \p n) = *(\p first + \c m). Unlike \c std::shift_right,
 *  \p shift_right offers no guarantee on order of operation. Elements that are in the original range but
 *  not the new range are left in a valid but unspecified state. If \p n is equal to zero or greater than
 *  (\p last - \p first) the algorithm has no effect.
 *
 *  The return value is the beginning of the resulting range \p first + \p n.
 *
 *  The algorithm's execution is parallelized as determined by \p exec.
 *
 *  \param exec The execution policy to use for parallelization.
 *  \param first The beginning of the sequence to shift.
 *  \param last The end of the sequence to shift.
 *  \param n The number of positions to shift.
 *  \return The beginning of the resulting sequence.
 *  \see https://en.cppreference.com/w/cpp/algorithm/shift
 *
 *  \tparam DerivedPolicy The name of the derived execution policy.
 *  \tparam ForwardIterator must be a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a>.
 *
 *  \pre \p n may be equal to equal to zero or greater than (\p last - \p first), but \p n shall not be negative
 *
 *  The following code snippet demonstrates how to use \p shift_right on the elements of a range
 *  using the \p thrust::device parallelization policy:
 *
 *  \code
 *  #include <thrust/shift.h>
 *  #include <thrust/device_vector.h>
 *  #include <thrust/execution_policy.h>
 *  ...
 *
 *  thrust::device_vector<int> vec = {1, 2, 3, 4, 5, 6};
 *  ...
 *
 *  thrust::shift_right(thrust::device, vec.begin(), vec.end(), 2);
 *
 *  // vec is now equal to {unspecified, unspecified, 1, 2, 3, 4 }
 *  \endcode
 */
template <typename DerivedPolicy, typename ForwardIterator>
__host__ __device__
  ForwardIterator shift_right(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                      ForwardIterator first,
                      ForwardIterator last,
                      typename thrust::iterator_traits<ForwardIterator>::difference_type n);


/*! \p shift_right shifts the elements from the range [\p first, \p last) to the right. That is, it performs
 *  the assignments *(\p first + \p n) = *\p first, *(\p first + \p n + \c 1) = *(\p first + \c 1),
 *  and so on. Generally, for every integer \c m from \c 0 to \p last - \p first - n, \p shift_right
 *  performs the assignment *(\p first + \c m + \p n) = *(\p first + \c m). Elements that are in the original
 *  range but not the new range are left in a valid but unspecified state. If \p n is equal to zero or greater
 *  than (\p last - \p first) the algorithm has no effect.
 *
 *  The return value is the beginning of the resulting range \p first + \p n.
 *
 *  \param first The beginning of the sequence to shift.
 *  \param last The end of the sequence to shift.
 *  \param n The number of positions to shift.
 *  \return The beginning of the resulting sequence.
 *  \see https://en.cppreference.com/w/cpp/algorithm/shift
 *
 *  \tparam ForwardIterator must be a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a>.
 *
 *  \pre \p n may be equal to equal to zero or greater than (\p last - \p first), but \p n shall not be negative
 *
 *  The following code snippet demonstrates how to use \p shift_right on the elements of a range:
 *
 *  \code
 *  #include <thrust/shift.h>
 *  #include <thrust/device_vector.h>
 *  ...
 *
 *  thrust::device_vector<int> vec = {1, 2, 3, 4, 5, 6};
 *  ...
 *
 *  thrust::shift_right(vec.begin(), vec.end(), 2);
 *
 *  // vec is now equal to {unspecified, unspecified, 1, 2, 3, 4 }
 *  \endcode
 */
template <typename ForwardIterator>
  ForwardIterator shift_right(ForwardIterator first,
                              ForwardIterator last,
                              typename thrust::iterator_traits<ForwardIterator>::difference_type n);

/*! \} // end stream_compaction
 */

THRUST_NAMESPACE_END

#include <thrust/detail/shift.inl>
