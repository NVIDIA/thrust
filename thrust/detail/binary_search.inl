/*
 *  Copyright 2008-2012 NVIDIA Corporation
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


/*! \file binary_search.inl
 *  \brief Inline file for binary_search.h.
 */

#include <thrust/functional.h>

#include <thrust/iterator/iterator_traits.h>
#include <thrust/detail/backend/binary_search.h>

namespace thrust
{

//////////////////////
// Scalar Functions //
//////////////////////

template <class ForwardIterator, class LessThanComparable>
ForwardIterator lower_bound(ForwardIterator first, 
                            ForwardIterator last,
                            const LessThanComparable& value)
{
    return thrust::lower_bound(first, last, value, thrust::less<LessThanComparable>());
}

template <class ForwardIterator, class T, class StrictWeakOrdering>
ForwardIterator lower_bound(ForwardIterator first,
                            ForwardIterator last,
                            const T& value, 
                            StrictWeakOrdering comp)
{
    return thrust::detail::backend::lower_bound(first, last, value, comp);
}

template <class ForwardIterator, class LessThanComparable>
ForwardIterator upper_bound(ForwardIterator first, 
                            ForwardIterator last,
                            const LessThanComparable& value)
{
    return thrust::upper_bound(first, last, value, thrust::less<LessThanComparable>());
}

template <class ForwardIterator, class T, class StrictWeakOrdering>
ForwardIterator upper_bound(ForwardIterator first,
                            ForwardIterator last,
                            const T& value, 
                            StrictWeakOrdering comp)
{
    return thrust::detail::backend::upper_bound(first, last, value, comp);
}

template <class ForwardIterator, class LessThanComparable>
bool binary_search(ForwardIterator first, 
                   ForwardIterator last,
                   const LessThanComparable& value)
{
    return thrust::binary_search(first, last, value, thrust::less<LessThanComparable>());
}

template <class ForwardIterator, class T, class StrictWeakOrdering>
bool binary_search(ForwardIterator first,
                   ForwardIterator last,
                   const T& value, 
                   StrictWeakOrdering comp)
{
    return thrust::detail::backend::binary_search(first, last, value, comp);
}

template <class ForwardIterator, class LessThanComparable>
thrust::pair<ForwardIterator, ForwardIterator>
equal_range(ForwardIterator first,
            ForwardIterator last,
            const LessThanComparable& value)
{
    return thrust::equal_range(first, last, value, thrust::less<LessThanComparable>());
}

template <class ForwardIterator, class T, class StrictWeakOrdering>
thrust::pair<ForwardIterator, ForwardIterator>
equal_range(ForwardIterator first,
            ForwardIterator last,
            const T& value,
            StrictWeakOrdering comp)
{
    ForwardIterator lb = thrust::lower_bound(first, last, value, comp);
    ForwardIterator ub = thrust::upper_bound(first, last, value, comp);
    return thrust::make_pair(lb, ub);
}

//////////////////////
// Vector Functions //
//////////////////////

template <class ForwardIterator, class InputIterator, class OutputIterator>
OutputIterator lower_bound(ForwardIterator first, 
                           ForwardIterator last,
                           InputIterator values_first, 
                           InputIterator values_last,
                           OutputIterator output)
{
    typedef typename thrust::iterator_traits<InputIterator>::value_type ValueType;

    return thrust::lower_bound(first, last, values_first, values_last, output, thrust::less<ValueType>());
}

template <class ForwardIterator, class InputIterator, class OutputIterator, class StrictWeakOrdering>
OutputIterator lower_bound(ForwardIterator first, 
                           ForwardIterator last,
                           InputIterator values_first, 
                           InputIterator values_last,
                           OutputIterator output,
                           StrictWeakOrdering comp)
{
    return thrust::detail::backend::lower_bound(first, last, values_first, values_last, output, comp);
}
    
template <class ForwardIterator, class InputIterator, class OutputIterator>
OutputIterator upper_bound(ForwardIterator first, 
                           ForwardIterator last,
                           InputIterator values_first, 
                           InputIterator values_last,
                           OutputIterator output)
{
    typedef typename thrust::iterator_traits<InputIterator>::value_type ValueType;

    return thrust::upper_bound(first, last, values_first, values_last, output, thrust::less<ValueType>());
}

template <class ForwardIterator, class InputIterator, class OutputIterator, class StrictWeakOrdering>
OutputIterator upper_bound(ForwardIterator first, 
                           ForwardIterator last,
                           InputIterator values_first, 
                           InputIterator values_last,
                           OutputIterator output,
                           StrictWeakOrdering comp)
{
    return thrust::detail::backend::upper_bound(first, last, values_first, values_last, output, comp);
}

template <class ForwardIterator, class InputIterator, class OutputIterator>
OutputIterator binary_search(ForwardIterator first, 
                             ForwardIterator last,
                             InputIterator values_first, 
                             InputIterator values_last,
                             OutputIterator output)
{
    typedef typename thrust::iterator_traits<InputIterator>::value_type ValueType;

    return thrust::binary_search(first, last, values_first, values_last, output, thrust::less<ValueType>());
}

template <class ForwardIterator, class InputIterator, class OutputIterator, class StrictWeakOrdering>
OutputIterator binary_search(ForwardIterator first, 
                             ForwardIterator last,
                             InputIterator values_first, 
                             InputIterator values_last,
                             OutputIterator output,
                             StrictWeakOrdering comp)
{
    return thrust::detail::backend::binary_search(first, last, values_first, values_last, output, comp);
}

} // end namespace thrust

