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


/*! \file advance.h
 *  \brief Dispatch layer to advance function.
 */

#pragma once

#include <iterator>

#include <thrust/system/detail/cpp/tag.h>
#include <thrust/system/cuda/memory.h>
#include <thrust/system/omp/memory.h>

namespace thrust
{

namespace detail
{

namespace dispatch
{

// XXX We really ought to dispatch on thrust::iterator_traversal instead

template<typename InputIterator, typename Distance>
void advance(InputIterator &i, Distance n,
             thrust::cpp::tag)
{
    std::advance(i, n);
}

template<typename InputIterator, typename Distance>
void advance(InputIterator &i, Distance n,
             thrust::omp::tag)
{
    // omp iterators are random access
    i += n;
}

template<typename InputIterator, typename Distance>
void advance(InputIterator &i, Distance n,
             thrust::cuda::tag)
{
    // cuda iterators are random access
    i += n;
}

template<typename InputIterator, typename Distance>
void advance(InputIterator &i, Distance n,
             thrust::any_space_tag)
{
    // any space iterators are random access
    i += n;
}

} // end namespace dispatch

} // end namespace detail

} // end namespace thrust

