/*
 *  Copyright 2008-2011 NVIDIA Corporation
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


#include <thrust/iterator/iterator_traits.h>
#include <thrust/functional.h>

#include <thrust/detail/dispatch/extrema.h>

namespace thrust
{

template<typename T, typename BinaryPredicate>
  T min THRUST_PREVENT_MACRO_SUBSTITUTION (const T &lhs, const T &rhs, BinaryPredicate comp)
{
  return comp(rhs, lhs) ? rhs : lhs;
} // end min()

template<typename T>
  T min THRUST_PREVENT_MACRO_SUBSTITUTION (const T &lhs, const T &rhs)
{
  return rhs < lhs ? rhs : lhs;
} // end min()

template<typename T, typename BinaryPredicate>
  T max THRUST_PREVENT_MACRO_SUBSTITUTION (const T &lhs, const T &rhs, BinaryPredicate comp)
{
  return comp(lhs,rhs) ? rhs : lhs;
} // end max()

template<typename T>
  T max THRUST_PREVENT_MACRO_SUBSTITUTION (const T &lhs, const T &rhs)
{
  return lhs < rhs ? rhs : lhs;
} // end max()


template <typename ForwardIterator>
ForwardIterator min_element(ForwardIterator first, ForwardIterator last)
{
    // use < predicate by default
    typedef typename thrust::iterator_traits<ForwardIterator>::value_type InputType;
    return thrust::min_element(first, last, thrust::less<InputType>());
} // end min_element()


template <typename ForwardIterator, typename BinaryPredicate>
ForwardIterator min_element(ForwardIterator first, ForwardIterator last,
                            BinaryPredicate comp)
{
    // dispatch on space
    return thrust::detail::dispatch::min_element(first, last, comp,
            typename thrust::iterator_space<ForwardIterator>::type());
} // end min_element()


template <typename ForwardIterator>
ForwardIterator max_element(ForwardIterator first, ForwardIterator last)
{
    // use < predicate by default
    typedef typename thrust::iterator_traits<ForwardIterator>::value_type InputType;
    return thrust::max_element(first, last, thrust::less<InputType>());
} // end max_element()


template <typename ForwardIterator, typename BinaryPredicate>
ForwardIterator max_element(ForwardIterator first, ForwardIterator last,
                            BinaryPredicate comp)
{
    // dispatch on space
    return thrust::detail::dispatch::max_element(first, last, comp,
            typename thrust::iterator_space<ForwardIterator>::type());
} // end max_element()


template <typename ForwardIterator>
thrust::pair<ForwardIterator,ForwardIterator> 
                minmax_element(ForwardIterator first, ForwardIterator last)
{
    // use < predicate by default
    typedef typename thrust::iterator_traits<ForwardIterator>::value_type InputType;
    return thrust::minmax_element(first, last, thrust::less<InputType>());
} // end minmax_element()


template <typename ForwardIterator, typename BinaryPredicate>
thrust::pair<ForwardIterator,ForwardIterator> 
                minmax_element(ForwardIterator first, ForwardIterator last,
                               BinaryPredicate comp)
{
    // dispatch on space
    return thrust::detail::dispatch::minmax_element(first, last, comp,
            typename thrust::iterator_space<ForwardIterator>::type());
} // end minmax_element()


} // end namespace thrust

