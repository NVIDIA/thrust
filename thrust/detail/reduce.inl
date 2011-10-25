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


/*! \file reduce.inl
 *  \brief Inline file for reduce.h.
 */

#include <thrust/reduce.h>
#include <thrust/detail/backend/generic/select_system.h>
#include <thrust/detail/backend/generic/reduce.h>
#include <thrust/detail/backend/generic/reduce_by_key.h>
#include <thrust/iterator/iterator_traits.h>

// XXX make the backend-specific versions of reduce available
// XXX try to eliminate the need for these
#include <thrust/system/cpp/detail/reduce.h>
#include <thrust/detail/backend/omp/reduce.h>
#include <thrust/detail/backend/cuda/reduce.h>

// XXX make the backend-specific versions of reduce_by_key available
// XXX try to eliminate the need for these
#include <thrust/system/cpp/detail/reduce_by_key.h>
#include <thrust/detail/backend/omp/reduce_by_key.h>
#include <thrust/detail/backend/cuda/reduce_by_key.h>

namespace thrust
{

template<typename InputIterator>
typename thrust::iterator_traits<InputIterator>::value_type
  reduce(InputIterator first,
         InputIterator last)
{
  using thrust::detail::backend::generic::select_system;
  using thrust::detail::backend::generic::reduce;

  typedef typename thrust::iterator_space<InputIterator>::type space;

  return reduce(select_system(space()), first, last);
}

template<typename InputIterator,
         typename T>
   T reduce(InputIterator first,
            InputIterator last,
            T init)
{
  using thrust::detail::backend::generic::select_system;
  using thrust::detail::backend::generic::reduce;

  typedef typename thrust::iterator_space<InputIterator>::type space;

  return reduce(select_system(space()), first, last, init);
}


template<typename InputIterator,
         typename T,
         typename BinaryFunction>
   T reduce(InputIterator first,
            InputIterator last,
            T init,
            BinaryFunction binary_op)
{
  using thrust::detail::backend::generic::select_system;
  using thrust::detail::backend::generic::reduce;

  typedef typename thrust::iterator_space<InputIterator>::type space;

  return reduce(select_system(space()), first, last, init, binary_op);
}

template <typename InputIterator1,
          typename InputIterator2,
          typename OutputIterator1,
          typename OutputIterator2>
  thrust::pair<OutputIterator1,OutputIterator2>
  reduce_by_key(InputIterator1 keys_first, 
                InputIterator1 keys_last,
                InputIterator2 values_first,
                OutputIterator1 keys_output,
                OutputIterator2 values_output)
{
  using thrust::detail::backend::generic::select_system;
  using thrust::detail::backend::generic::reduce_by_key;

  typedef typename thrust::iterator_space<InputIterator1>::type  space1;
  typedef typename thrust::iterator_space<InputIterator2>::type  space2;
  typedef typename thrust::iterator_space<OutputIterator1>::type space3;
  typedef typename thrust::iterator_space<OutputIterator2>::type space4;

  return reduce_by_key(select_system(space1(),space2(),space3(),space4()), keys_first, keys_last, values_first, keys_output, values_output);
}

template <typename InputIterator1,
          typename InputIterator2,
          typename OutputIterator1,
          typename OutputIterator2,
          typename BinaryPredicate>
  thrust::pair<OutputIterator1,OutputIterator2>
  reduce_by_key(InputIterator1 keys_first, 
                InputIterator1 keys_last,
                InputIterator2 values_first,
                OutputIterator1 keys_output,
                OutputIterator2 values_output,
                BinaryPredicate binary_pred)
{
  using thrust::detail::backend::generic::select_system;
  using thrust::detail::backend::generic::reduce_by_key;

  typedef typename thrust::iterator_space<InputIterator1>::type  space1;
  typedef typename thrust::iterator_space<InputIterator2>::type  space2;
  typedef typename thrust::iterator_space<OutputIterator1>::type space3;
  typedef typename thrust::iterator_space<OutputIterator2>::type space4;

  return reduce_by_key(select_system(space1(),space2(),space3(),space4()), keys_first, keys_last, values_first, keys_output, values_output, binary_pred);
}

template <typename InputIterator1,
          typename InputIterator2,
          typename OutputIterator1,
          typename OutputIterator2,
          typename BinaryPredicate,
          typename BinaryFunction>
  thrust::pair<OutputIterator1,OutputIterator2>
  reduce_by_key(InputIterator1 keys_first, 
                InputIterator1 keys_last,
                InputIterator2 values_first,
                OutputIterator1 keys_output,
                OutputIterator2 values_output,
                BinaryPredicate binary_pred,
                BinaryFunction binary_op)
{
  using thrust::detail::backend::generic::select_system;
  using thrust::detail::backend::generic::reduce_by_key;

  typedef typename thrust::iterator_space<InputIterator1>::type  space1;
  typedef typename thrust::iterator_space<InputIterator2>::type  space2;
  typedef typename thrust::iterator_space<OutputIterator1>::type space3;
  typedef typename thrust::iterator_space<OutputIterator2>::type space4;

  return reduce_by_key(select_system(space1(),space2(),space3(),space4()), keys_first, keys_last, values_first, keys_output, values_output, binary_pred, binary_op);
}

} // end namespace thrust

