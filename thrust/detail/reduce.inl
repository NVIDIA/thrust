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


#include <thrust/functional.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/detail/type_traits.h>
#include <thrust/detail/type_traits/iterator/is_output_iterator.h>

#include <thrust/detail/dispatch/reduce.h>

namespace thrust
{

template<typename InputIterator>
typename thrust::iterator_traits<InputIterator>::value_type
  reduce(InputIterator first,
         InputIterator last)
{
  typedef typename thrust::iterator_traits<InputIterator>::value_type InputType;

  // use InputType(0) as init by default
  return thrust::reduce(first, last, InputType(0));
}

template<typename InputIterator,
         typename T>
   T reduce(InputIterator first,
            InputIterator last,
            T init)
{
    // use plus<T> by default
    return thrust::reduce(first, last, init, thrust::plus<T>());
}


template<typename InputIterator,
         typename T,
         typename BinaryFunction>
   T reduce(InputIterator first,
            InputIterator last,
            T init,
            BinaryFunction binary_op)
{
    //dispatch on space 
    return thrust::detail::dispatch::reduce(first, last, init, binary_op,
            typename thrust::iterator_space<InputIterator>::type());
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
    typedef typename thrust::iterator_value<InputIterator1>::type KeyType;

    // use equal_to<KeyType> as default BinaryPredicate
    return thrust::reduce_by_key(keys_first, keys_last, 
                                 values_first,
                                 keys_output,
                                 values_output,
                                 thrust::equal_to<KeyType>());
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
    typedef typename thrust::detail::eval_if<
      thrust::detail::is_output_iterator<OutputIterator2>::value,
      thrust::iterator_value<InputIterator2>,
      thrust::iterator_value<OutputIterator2>
    >::type T;

    // use plus<T> as default BinaryFunction
    return thrust::reduce_by_key(keys_first, keys_last, 
                                 values_first,
                                 keys_output,
                                 values_output,
                                 binary_pred,
                                 thrust::plus<T>());
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
    //dispatch on space 
    return thrust::detail::dispatch::reduce_by_key
            (keys_first, keys_last, 
             values_first,
             keys_output,
             values_output,
             binary_pred,
             binary_op,
             typename thrust::iterator_space<InputIterator1>::type(),
             typename thrust::iterator_space<InputIterator2>::type(),
             typename thrust::iterator_space<OutputIterator1>::type(),
             typename thrust::iterator_space<OutputIterator2>::type());
}

} // end namespace thrust

