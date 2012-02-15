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


#pragma once

#include <thrust/iterator/iterator_traits.h>
#include <thrust/detail/backend/generic/find.h>
#include <thrust/detail/backend/cpp/find.h>

namespace thrust
{
namespace detail
{
namespace backend
{
namespace dispatch
{


template <typename InputIterator, typename Predicate, typename Backend>
InputIterator find_if(InputIterator first,
                      InputIterator last,
                      Predicate pred,
                      Backend)
{
  return thrust::detail::backend::generic::find_if(first, last, pred);
}

template <typename InputIterator, typename Predicate>
InputIterator find_if(InputIterator first,
                      InputIterator last,
                      Predicate pred,
                      thrust::host_space_tag)
{
  return thrust::detail::backend::cpp::find_if(first, last, pred);
}



} // end dispatch


template <typename InputIterator, typename Predicate>
InputIterator find_if(InputIterator first,
                      InputIterator last,
                      Predicate pred)
{
  return thrust::detail::backend::dispatch::find_if(first, last, pred,
      typename thrust::iterator_space<InputIterator>::type());
}


} // end namespace backend
} // end namespace detail
} // end namespace thrust

