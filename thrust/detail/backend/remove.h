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


/*! \file remove.h
 *  \brief Entry points for remove backend.
 */

#pragma once

#include <thrust/detail/backend/generic/remove.h>
#include <thrust/detail/backend/cpp/remove.h>

namespace thrust
{
namespace detail
{
namespace backend
{
namespace dispatch
{



template<typename ForwardIterator,
         typename Predicate,
         typename Backend>
  ForwardIterator remove_if(ForwardIterator first,
                            ForwardIterator last,
                            Predicate pred,
                            Backend)
{
  return thrust::detail::backend::generic::remove_if(first, last, pred);
}


template<typename ForwardIterator,
         typename Predicate>
  ForwardIterator remove_if(ForwardIterator first,
                            ForwardIterator last,
                            Predicate pred,
                            thrust::host_space_tag)
{
  return thrust::detail::backend::cpp::remove_if(first, last, pred);
}

template<typename ForwardIterator,
         typename InputIterator,
         typename Predicate,
         typename Backend>
  ForwardIterator remove_if(ForwardIterator first,
                            ForwardIterator last,
                            InputIterator stencil,
                            Predicate pred,
                            Backend)
{
  return thrust::detail::backend::generic::remove_if(first, last, stencil, pred);
}

template<typename ForwardIterator,
         typename InputIterator,
         typename Predicate>
  ForwardIterator remove_if(ForwardIterator first,
                            ForwardIterator last,
                            InputIterator stencil,
                            Predicate pred,
                            thrust::host_space_tag)
{
  return thrust::detail::backend::cpp::remove_if(first, last, stencil, pred);
}


template<typename InputIterator,
         typename OutputIterator,
         typename Predicate,
         typename Backend>
  OutputIterator remove_copy_if(InputIterator first,
                                InputIterator last,
                                OutputIterator result,
                                Predicate pred,
                                Backend)
{
  return thrust::detail::backend::generic::remove_copy_if(first, last, result, pred);
}

template<typename InputIterator,
         typename OutputIterator,
         typename Predicate>
  OutputIterator remove_copy_if(InputIterator first,
                                InputIterator last,
                                OutputIterator result,
                                Predicate pred,
                                thrust::host_space_tag)
{
  return thrust::detail::backend::cpp::remove_copy_if(first, last, result, pred);
}


template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator,
         typename Predicate,
         typename Backend>
  OutputIterator remove_copy_if(InputIterator1 first,
                                InputIterator1 last,
                                InputIterator2 stencil,
                                OutputIterator result,
                                Predicate pred,
                                Backend)
{
  return thrust::detail::backend::generic::remove_copy_if(first, last, stencil, result, pred);
}

template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator,
         typename Predicate>
  OutputIterator remove_copy_if(InputIterator1 first,
                                InputIterator1 last,
                                InputIterator2 stencil,
                                OutputIterator result,
                                Predicate pred,
                                thrust::host_space_tag)
{
  return thrust::detail::backend::cpp::remove_copy_if(first, last, stencil, result, pred);
}



} // end namespace dispatch


template<typename ForwardIterator,
         typename Predicate>
  ForwardIterator remove_if(ForwardIterator first,
                            ForwardIterator last,
                            Predicate pred)
{
  return thrust::detail::backend::dispatch::remove_if(first, last, pred,
    typename thrust::iterator_space<ForwardIterator>::type());
}


template<typename ForwardIterator,
         typename InputIterator,
         typename Predicate>
  ForwardIterator remove_if(ForwardIterator first,
                            ForwardIterator last,
                            InputIterator stencil,
                            Predicate pred)
{
  return thrust::detail::backend::dispatch::remove_if(first, last, stencil, pred,
    typename thrust::detail::minimum_space<
      typename thrust::iterator_space<ForwardIterator>::type,
      typename thrust::iterator_space<InputIterator>::type
    >::type());
}


template<typename InputIterator,
         typename OutputIterator,
         typename Predicate>
  OutputIterator remove_copy_if(InputIterator first,
                                InputIterator last,
                                OutputIterator result,
                                Predicate pred)
{
  return thrust::detail::backend::dispatch::remove_copy_if(first, last, result, pred,
    typename thrust::detail::minimum_space<
      typename thrust::iterator_space<InputIterator>::type,
      typename thrust::iterator_space<OutputIterator>::type
    >::type());
}


template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator,
         typename Predicate>
  OutputIterator remove_copy_if(InputIterator1 first,
                                InputIterator1 last,
                                InputIterator2 stencil,
                                OutputIterator result,
                                Predicate pred)
{
  return thrust::detail::backend::dispatch::remove_copy_if(first, last, stencil, result, pred,
    typename thrust::detail::minimum_space<
      typename thrust::iterator_space<InputIterator1>::type,
      typename thrust::iterator_space<InputIterator2>::type,
      typename thrust::iterator_space<OutputIterator>::type
    >::type());
}


} // end namespace backend
} // end namespace detail
} // end namespace thrust

