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

#pragma once

#include <thrust/detail/type_traits.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/iterator/detail/minimum_space.h>
#include <thrust/detail/backend/cpp/copy.h>
#include <thrust/detail/backend/omp/copy.h>
#include <thrust/detail/backend/cuda/copy.h>

#include <thrust/system/cpp/detail/tag.h>
#include <thrust/system/cuda/detail/tag.h>
#include <thrust/system/omp/detail/tag.h>

namespace thrust
{
namespace detail
{
namespace backend
{
namespace dispatch
{

namespace detail
{

// dispatch procedure:
// 1. if either space is cuda, dispatch to cuda::copy
// 2. else if either space is omp, dispatch to omp::copy
// 3. else if either space is cpp, dispatch to cpp::copy
// 4. else error

template<typename Space1, typename Space2>
  struct copy_case
    : thrust::detail::eval_if<
        thrust::detail::or_<
          thrust::detail::is_same<Space1,thrust::cuda::tag>,
          thrust::detail::is_same<Space2,thrust::cuda::tag>
        >::value,
        thrust::detail::identity_<thrust::cuda::tag>,
        thrust::detail::eval_if<
          thrust::detail::or_<
            thrust::detail::is_same<Space1,thrust::omp::tag>,
            thrust::detail::is_same<Space2,thrust::omp::tag>
          >::value,
          thrust::detail::identity_<thrust::omp::tag>,
          thrust::detail::eval_if<
            thrust::detail::or_<
              thrust::detail::is_same<Space1,thrust::cpp::tag>,
              thrust::detail::is_same<Space2,thrust::cpp::tag>
            >::value,
            thrust::detail::identity_<thrust::cpp::tag>,
            thrust::detail::identity_<void>
          >
        >
      >
{};

} // end detail



template<typename InputIterator,
         typename OutputIterator>
  OutputIterator copy(InputIterator first,
                      InputIterator last,
                      OutputIterator result,
                      thrust::cpp::tag)
{
  return thrust::detail::backend::cpp::copy(first,last,result);
} // end copy()

template<typename InputIterator,
         typename Size,
         typename OutputIterator>
  OutputIterator copy_n(InputIterator first,
                        Size n,
                        OutputIterator result,
                        thrust::cpp::tag)
{
  return thrust::detail::backend::cpp::copy_n(first,n,result);
} // end copy_n()



template<typename InputIterator,
         typename OutputIterator>
  OutputIterator copy(InputIterator first,
                      InputIterator last,
                      OutputIterator result,
                      thrust::omp::tag)
{
  return thrust::detail::backend::omp::copy(first, last, result);
} // end copy()

template<typename InputIterator,
         typename Size,
         typename OutputIterator>
  OutputIterator copy_n(InputIterator first,
                        Size n,
                        OutputIterator result,
                        thrust::omp::tag)
{
  return thrust::detail::backend::omp::copy_n(first, n, result);
} // end copy_n()



template<typename InputIterator,
         typename OutputIterator>
  OutputIterator copy(InputIterator first,
                      InputIterator last,
                      OutputIterator result,
                      thrust::cuda::tag)
{
  return thrust::detail::backend::cuda::copy(first, last, result);
} // end copy()


template<typename InputIterator,
         typename Size,
         typename OutputIterator>
  OutputIterator copy_n(InputIterator first,
                        Size n,
                        OutputIterator result,
                        thrust::cuda::tag)
{
  return thrust::detail::backend::cuda::copy_n(first, n, result);
} // end copy_n()



// entry points

template<typename InputIterator,
         typename OutputIterator>
  OutputIterator copy(InputIterator first,
                      InputIterator last,
                      OutputIterator result)
{
  return thrust::detail::backend::dispatch::copy(first,last,result,
    typename thrust::detail::backend::dispatch::detail::copy_case<
      typename thrust::iterator_space<InputIterator>::type,
      typename thrust::iterator_space<OutputIterator>::type
    >::type());
} // end copy()

template<typename InputIterator,
         typename Size,
         typename OutputIterator>
  OutputIterator copy_n(InputIterator  first, 
                        Size n, 
                        OutputIterator result)
{
  return thrust::detail::backend::dispatch::copy_n(first, n, result,
    typename thrust::detail::backend::dispatch::detail::copy_case<
      typename thrust::iterator_space<InputIterator>::type,
      typename thrust::iterator_space<OutputIterator>::type
    >::type());
} // end copy_n()


} // end namespace dispatch
} // end namespace backend
} // end namespace detail
} // end namespace thrust

