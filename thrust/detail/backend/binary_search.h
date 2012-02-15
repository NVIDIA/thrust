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


/*! \file binary_search.h
 *  \brief Backend interface to binary search functions.
 */

#pragma once

#include <thrust/iterator/iterator_traits.h>
#include <thrust/iterator/detail/minimum_space.h>

#include <thrust/detail/backend/cpp/binary_search.h>
#include <thrust/detail/backend/generic/binary_search.h>


namespace thrust
{
namespace detail
{
namespace backend
{
namespace dispatch
{

template <typename ForwardIterator, typename T, typename StrictWeakOrdering, typename Backend>
ForwardIterator lower_bound(ForwardIterator begin,
                            ForwardIterator end,
                            const T& value, 
                            StrictWeakOrdering comp,
                            Backend)
{
    return thrust::detail::backend::generic::lower_bound(begin, end, value, comp);
}

template <typename ForwardIterator, typename T, typename StrictWeakOrdering>
ForwardIterator lower_bound(ForwardIterator begin,
                            ForwardIterator end,
                            const T& value, 
                            StrictWeakOrdering comp,
                            thrust::host_space_tag)
{
    return thrust::detail::backend::cpp::lower_bound(begin, end, value, comp);
}

template <typename ForwardIterator, typename T, typename StrictWeakOrdering, typename Backend>
ForwardIterator upper_bound(ForwardIterator begin,
                            ForwardIterator end,
                            const T& value, 
                            StrictWeakOrdering comp,
                            Backend)
{
    return thrust::detail::backend::generic::upper_bound(begin, end, value, comp);
}

template <typename ForwardIterator, typename T, typename StrictWeakOrdering>
ForwardIterator upper_bound(ForwardIterator begin,
                            ForwardIterator end,
                            const T& value, 
                            StrictWeakOrdering comp,
                            thrust::host_space_tag)
{
    return thrust::detail::backend::cpp::upper_bound(begin, end, value, comp);
}


template <typename ForwardIterator, typename T, typename StrictWeakOrdering, typename Backend>
bool binary_search(ForwardIterator begin,
                   ForwardIterator end,
                   const T& value, 
                   StrictWeakOrdering comp,
                   Backend)
{
    return thrust::detail::backend::generic::binary_search(begin, end, value, comp);
}

template <typename ForwardIterator, typename T, typename StrictWeakOrdering>
bool binary_search(ForwardIterator begin,
                   ForwardIterator end,
                   const T& value, 
                   StrictWeakOrdering comp,
                   thrust::host_space_tag)
{
    return thrust::detail::backend::cpp::binary_search(begin, end, value, comp);
}


template <typename ForwardIterator, typename InputIterator, typename OutputIterator, typename StrictWeakOrdering, typename Backend>
OutputIterator lower_bound(ForwardIterator begin, 
                           ForwardIterator end,
                           InputIterator values_begin, 
                           InputIterator values_end,
                           OutputIterator output,
                           StrictWeakOrdering comp,
                           Backend)
{
    return thrust::detail::backend::generic::lower_bound(begin, end, values_begin, values_end, output, comp);
}

template <typename ForwardIterator, typename InputIterator, typename OutputIterator, typename StrictWeakOrdering, typename Backend>
OutputIterator upper_bound(ForwardIterator begin, 
                           ForwardIterator end,
                           InputIterator values_begin, 
                           InputIterator values_end,
                           OutputIterator output,
                           StrictWeakOrdering comp,
                           Backend)
{
    return thrust::detail::backend::generic::upper_bound(begin, end, values_begin, values_end, output, comp);
}

template <typename ForwardIterator, typename InputIterator, typename OutputIterator, typename StrictWeakOrdering, typename Backend>
OutputIterator binary_search(ForwardIterator begin, 
                             ForwardIterator end,
                             InputIterator values_begin, 
                             InputIterator values_end,
                             OutputIterator output,
                             StrictWeakOrdering comp,
                             Backend)
{
    return thrust::detail::backend::generic::binary_search(begin, end, values_begin, values_end, output, comp);
}

} // end namespace dispatch


template <typename ForwardIterator, typename T, typename StrictWeakOrdering>
ForwardIterator lower_bound(ForwardIterator begin,
                            ForwardIterator end,
                            const T& value, 
                            StrictWeakOrdering comp)
{
    return thrust::detail::backend::dispatch::lower_bound(begin, end, value, comp,
        typename thrust::iterator_space<ForwardIterator>::type());
}

template <typename ForwardIterator, typename T, typename StrictWeakOrdering>
ForwardIterator upper_bound(ForwardIterator begin,
                            ForwardIterator end,
                            const T& value, 
                            StrictWeakOrdering comp)
{
    return thrust::detail::backend::dispatch::upper_bound(begin, end, value, comp,
        typename thrust::iterator_space<ForwardIterator>::type());
}


template <typename ForwardIterator, typename T, typename StrictWeakOrdering>
bool binary_search(ForwardIterator begin,
                   ForwardIterator end,
                   const T& value, 
                   StrictWeakOrdering comp)
{
    return thrust::detail::backend::dispatch::binary_search(begin, end, value, comp,
        typename thrust::iterator_space<ForwardIterator>::type());
}


template <typename ForwardIterator, typename InputIterator, typename OutputIterator, typename StrictWeakOrdering>
OutputIterator lower_bound(ForwardIterator begin, 
                           ForwardIterator end,
                           InputIterator values_begin, 
                           InputIterator values_end,
                           OutputIterator output,
                           StrictWeakOrdering comp)
{
    return thrust::detail::backend::dispatch::lower_bound(begin, end, values_begin, values_end, output, comp,
        typename thrust::detail::minimum_space<
          typename thrust::iterator_space<ForwardIterator>::type,
          typename thrust::iterator_space<InputIterator>::type,
          typename thrust::iterator_space<OutputIterator>::type
        >::type());
}

template <typename ForwardIterator, typename InputIterator, typename OutputIterator, typename StrictWeakOrdering>
OutputIterator upper_bound(ForwardIterator begin, 
                           ForwardIterator end,
                           InputIterator values_begin, 
                           InputIterator values_end,
                           OutputIterator output,
                           StrictWeakOrdering comp)
{
    return thrust::detail::backend::dispatch::upper_bound(begin, end, values_begin, values_end, output, comp,
        typename thrust::detail::minimum_space<
          typename thrust::iterator_space<ForwardIterator>::type,
          typename thrust::iterator_space<InputIterator>::type,
          typename thrust::iterator_space<OutputIterator>::type
        >::type());
}

template <typename ForwardIterator, typename InputIterator, typename OutputIterator, typename StrictWeakOrdering>
OutputIterator binary_search(ForwardIterator begin, 
                             ForwardIterator end,
                             InputIterator values_begin, 
                             InputIterator values_end,
                             OutputIterator output,
                             StrictWeakOrdering comp)
{
    return thrust::detail::backend::dispatch::binary_search(begin, end, values_begin, values_end, output, comp,
        typename thrust::detail::minimum_space<
          typename thrust::iterator_space<ForwardIterator>::type,
          typename thrust::iterator_space<InputIterator>::type,
          typename thrust::iterator_space<OutputIterator>::type
        >::type());
}

} // end namespace backend
} // end namespace detail
} // end namespace thrust

