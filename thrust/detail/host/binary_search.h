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


/*! \file binary_search.h
 *  \brief Search for values in sorted ranges [host]
 */

#pragma once

#include <algorithm>

#include <thrust/detail/type_traits.h>

namespace thrust
{

namespace detail
{

namespace host
{

namespace detail
{

//////////////
// Functors //
//////////////

template <class BooleanType>
struct bound_postprocess
{
};

template <>
struct bound_postprocess<thrust::detail::true_type>
{
    template <class ForwardIterator>
    typename thrust::iterator_traits<ForwardIterator>::difference_type operator()(ForwardIterator final, ForwardIterator begin){
        return final - begin;
    }
};

template <>
struct bound_postprocess<thrust::detail::false_type>
{
    template <class ForwardIterator>
    ForwardIterator operator()(ForwardIterator final, ForwardIterator begin){
        return final;
    }
};

} // end namespace detail


//////////////////////
// Scalar Functions //
//////////////////////
template <class ForwardIterator, class T, class StrictWeakOrdering>
ForwardIterator lower_bound(ForwardIterator begin,
                            ForwardIterator end,
                            const T& value, 
                            StrictWeakOrdering comp)
{
    return std::lower_bound(begin, end, value, comp);
}


template <class ForwardIterator, class T, class StrictWeakOrdering>
ForwardIterator upper_bound(ForwardIterator begin,
                            ForwardIterator end,
                            const T& value, 
                            StrictWeakOrdering comp)
{
    return std::upper_bound(begin, end, value, comp);
}

template <class ForwardIterator, class T, class StrictWeakOrdering>
bool binary_search(ForwardIterator begin,
                   ForwardIterator end,
                   const T& value, 
                   StrictWeakOrdering comp)
{
    return std::binary_search(begin, end, value, comp);
}

//////////////////////
// Vector Functions //
//////////////////////

template <class ForwardIterator, class InputIterator, class OutputIterator, class StrictWeakOrdering>
OutputIterator lower_bound(ForwardIterator begin, 
                           ForwardIterator end,
                           InputIterator values_begin, 
                           InputIterator values_end,
                           OutputIterator output,
                           StrictWeakOrdering comp)
{
    typedef typename thrust::iterator_traits<OutputIterator>::value_type OutputType;
    
    detail::bound_postprocess< typename thrust::detail::is_integral<OutputType>::type > postprocess;

    while(values_begin != values_end)
        *output++ = postprocess(std::lower_bound(begin, end, *values_begin++, comp), begin);

    return output;
}

template <class ForwardIterator, class InputIterator, class OutputIterator, class StrictWeakOrdering>
OutputIterator upper_bound(ForwardIterator begin, 
                           ForwardIterator end,
                           InputIterator values_begin, 
                           InputIterator values_end,
                           OutputIterator output,
                           StrictWeakOrdering comp)
{
    typedef typename thrust::iterator_traits<OutputIterator>::value_type OutputType;
    
    detail::bound_postprocess< typename thrust::detail::is_integral<OutputType>::type > postprocess;

    while(values_begin != values_end)
        *output++ = postprocess(std::upper_bound(begin, end, *values_begin++, comp), begin);

    return output;
}

template <class ForwardIterator, class InputIterator, class OutputIterator, class StrictWeakOrdering>
OutputIterator binary_search(ForwardIterator begin, 
                             ForwardIterator end,
                             InputIterator values_begin, 
                             InputIterator values_end,
                             OutputIterator output,
                             StrictWeakOrdering comp)
{
    while(values_begin != values_end)
        *output++ = std::binary_search(begin, end, *values_begin++, comp);

    return output;
}

} // end namespace host

} // end namespace detail

} // end namespace thrust

