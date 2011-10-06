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

#include <thrust/detail/config.h>
#include <thrust/detail/backend/omp/uninitialized_fill.h>
#include <thrust/detail/backend/omp/for_each.h>
#include <thrust/fill.h>
#include <thrust/detail/internal_functional.h>
#include <thrust/iterator/iterator_traits.h>

namespace thrust
{
namespace detail
{
namespace backend
{
namespace omp
{
namespace detail
{

template<typename ForwardIterator,
         typename Size,
         typename T>
  ForwardIterator uninitialized_fill_n(ForwardIterator first,
                                       Size n,
                                       const T &x,
                                       thrust::detail::true_type) // has_trivial_copy_constructor
{
  std::cout << "cuda::uninitialized_fill_n(trivial)" << std::endl;
  return thrust::fill_n(first, n, x);
} // end uninitialized_fill_n()

// non-trivial copy constructor path
template<typename ForwardIterator,
         typename Size,
         typename T>
  ForwardIterator uninitialized_fill_n(ForwardIterator first,
                                       Size n,
                                       const T &x,
                                       thrust::detail::false_type) // has_trivial_copy_constructor
{
  std::cout << "cuda::uninitialized_fill_n(non-trivial)" << std::endl;
  typedef typename iterator_traits<ForwardIterator>::value_type ValueType;

  return thrust::detail::backend::omp::detail::for_each_n(first, n, thrust::detail::uninitialized_fill_functor<ValueType>(x));
} // end uninitialized_fill_n()

} // end detail

template<typename ForwardIterator,
         typename Size,
         typename T>
  ForwardIterator uninitialized_fill_n(tag,
                                       ForwardIterator first,
                                       Size n,
                                       const T &x)
{
  typedef typename thrust::iterator_value<ForwardIterator>::type ValueType;

  typedef thrust::detail::has_trivial_copy_constructor<ValueType> ValueTypeHasTrivialCopyConstructor;

  return thrust::detail::backend::omp::detail::uninitialized_fill_n(first, n, x,
    ValueTypeHasTrivialCopyConstructor());
} // end generate_n()

} // end omp
} // end backend
} // end detail
} // end thrust

