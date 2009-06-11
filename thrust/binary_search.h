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


/*! \file binary_search.h
 *  \brief Search for values in sorted ranges.
 */

#pragma once

#include <thrust/detail/config.h>
#include <thrust/pair.h>

namespace thrust
{

/*! \addtogroup searching
 *  \{
 *  \addtogroup binary_search Binary Search
 *  \ingroup copying
 *  \{
 */


//////////////////////   
// Scalar Functions //
//////////////////////


/*! \p lower_bound
 */
template <class ForwardIterator, class LessThanComparable>
ForwardIterator lower_bound(ForwardIterator begin, 
                            ForwardIterator end,
                            const LessThanComparable& value);

/*! \p lower_bound
 */
template <class ForwardIterator, class T, class StrictWeakOrdering>
ForwardIterator lower_bound(ForwardIterator begin,
                            ForwardIterator end,
                            const T& value, 
                            StrictWeakOrdering comp);

/*! \p upper_bound
 */
template <class ForwardIterator, class LessThanComparable>
ForwardIterator upper_bound(ForwardIterator begin, 
                            ForwardIterator end,
                            const LessThanComparable& value);

/*! \p upper_bound
 */
template <class ForwardIterator, class T, class StrictWeakOrdering>
ForwardIterator upper_bound(ForwardIterator begin,
                            ForwardIterator end,
                            const T& value, 
                            StrictWeakOrdering comp);

/*! \p binary_search
 */
template <class ForwardIterator, class LessThanComparable>
bool binary_search(ForwardIterator begin, 
                   ForwardIterator end,
                   const LessThanComparable& value);

/*! \p binary_search
 */
template <class ForwardIterator, class T, class StrictWeakOrdering>
bool binary_search(ForwardIterator begin,
                   ForwardIterator end,
                   const T& value, 
                   StrictWeakOrdering comp);

/*! \p equal_range
 */
template <class ForwardIterator, class LessThanComparable>
thrust::pair<ForwardIterator, ForwardIterator>
equal_range(ForwardIterator first,
            ForwardIterator last,
            const LessThanComparable& value);

/*! \p equal_range
 */
template <class ForwardIterator, class T, class StrictWeakOrdering>
thrust::pair<ForwardIterator, ForwardIterator>
equal_range(ForwardIterator first,
            ForwardIterator last,
            const T& value,
            StrictWeakOrdering comp);


//////////////////////
// Vector Functions //
//////////////////////

/*! \p lower_bound
 */
template <class ForwardIterator, class InputIterator, class OutputIterator>
OutputIterator lower_bound(ForwardIterator begin, 
                           ForwardIterator end,
                           InputIterator values_begin, 
                           InputIterator values_end,
                           OutputIterator output);

/*! \p lower_bound
 */
template <class ForwardIterator, class InputIterator, class OutputIterator, class StrictWeakOrdering>
OutputIterator lower_bound(ForwardIterator begin, 
                           ForwardIterator end,
                           InputIterator values_begin, 
                           InputIterator values_end,
                           OutputIterator output,
                           StrictWeakOrdering comp);

/*! \p upper_bound
 */
template <class ForwardIterator, class InputIterator, class OutputIterator>
OutputIterator upper_bound(ForwardIterator begin, 
                           ForwardIterator end,
                           InputIterator values_begin, 
                           InputIterator values_end,
                           OutputIterator output);

/*! \p upper_bound
 */
template <class ForwardIterator, class InputIterator, class OutputIterator, class StrictWeakOrdering>
OutputIterator upper_bound(ForwardIterator begin, 
                           ForwardIterator end,
                           InputIterator values_begin, 
                           InputIterator values_end,
                           OutputIterator output,
                           StrictWeakOrdering comp);

/*! \p binary_search
 */
template <class ForwardIterator, class InputIterator, class OutputIterator>
OutputIterator binary_search(ForwardIterator begin, 
                             ForwardIterator end,
                             InputIterator values_begin, 
                             InputIterator values_end,
                             OutputIterator output);

/*! \p binary_search
 */
template <class ForwardIterator, class InputIterator, class OutputIterator, class StrictWeakOrdering>
OutputIterator binary_search(ForwardIterator begin, 
                             ForwardIterator end,
                             InputIterator values_begin, 
                             InputIterator values_end,
                             OutputIterator output,
                             StrictWeakOrdering comp);

/*! \} // binary_search
 *  \} // searching
 */

}; // end namespace thrust

#include <thrust/detail/binary_search.inl>

