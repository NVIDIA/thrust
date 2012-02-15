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


/*! \file extrema.h
 *  \brief Interface to extrema functions.
 */

#pragma once

#include <thrust/pair.h>
#include <thrust/detail/backend/generic/extrema.h>
#include <thrust/detail/backend/cpp/extrema.h>
#include <thrust/iterator/iterator_traits.h>

namespace thrust
{
namespace detail
{
namespace backend
{
namespace dispatch
{



template <typename ForwardIterator, typename BinaryPredicate, typename Backend>
ForwardIterator min_element(ForwardIterator first,
                            ForwardIterator last,
                            BinaryPredicate comp,
                            Backend)
{
    return thrust::detail::backend::generic::min_element(first, last, comp);
}

template <typename ForwardIterator, typename BinaryPredicate>
ForwardIterator min_element(ForwardIterator first,
                            ForwardIterator last,
                            BinaryPredicate comp,
                            thrust::host_space_tag)
{
    return thrust::detail::backend::cpp::min_element(first, last, comp);
}


template <typename ForwardIterator, typename BinaryPredicate, typename Backend>
ForwardIterator max_element(ForwardIterator first,
                            ForwardIterator last,
                            BinaryPredicate comp,
                            Backend)
{
    return thrust::detail::backend::generic::max_element(first, last, comp);
}

template <typename ForwardIterator, typename BinaryPredicate>
ForwardIterator max_element(ForwardIterator first,
                            ForwardIterator last,
                            BinaryPredicate comp,
                            thrust::host_space_tag)
{
    return thrust::detail::backend::cpp::max_element(first, last, comp);
}


template <typename ForwardIterator, typename BinaryPredicate, typename Backend>
thrust::pair<ForwardIterator,ForwardIterator> minmax_element(ForwardIterator first, 
                                                             ForwardIterator last,
                                                             BinaryPredicate comp,
                                                             Backend)
{
    return thrust::detail::backend::generic::minmax_element(first, last, comp);
}

template <typename ForwardIterator, typename BinaryPredicate>
thrust::pair<ForwardIterator,ForwardIterator> minmax_element(ForwardIterator first, 
                                                             ForwardIterator last,
                                                             BinaryPredicate comp,
                                                             thrust::host_space_tag)
{
    return thrust::detail::backend::cpp::minmax_element(first, last, comp);
}



} // end namespace dispatch



template <typename ForwardIterator, typename BinaryPredicate>
ForwardIterator min_element(ForwardIterator first,
                            ForwardIterator last,
                            BinaryPredicate comp)
{
    return thrust::detail::backend::dispatch::min_element(first, last, comp,
        typename thrust::iterator_space<ForwardIterator>::type());
}


template <typename ForwardIterator, typename BinaryPredicate>
ForwardIterator max_element(ForwardIterator first,
                            ForwardIterator last,
                            BinaryPredicate comp)
{
    return thrust::detail::backend::dispatch::max_element(first, last, comp,
        typename thrust::iterator_space<ForwardIterator>::type());
}


template <typename ForwardIterator, typename BinaryPredicate>
thrust::pair<ForwardIterator,ForwardIterator> minmax_element(ForwardIterator first, 
                                                             ForwardIterator last,
                                                             BinaryPredicate comp)
{
    return thrust::detail::backend::dispatch::minmax_element(first, last, comp,
        typename thrust::iterator_space<ForwardIterator>::type());
}



} // end namespace backend
} // end namespace detail
} // end namespace thrust

