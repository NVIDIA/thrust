/*
 *  Copyright 2008-2009 NVIDIA Corporation
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


} // end namespace thrust

