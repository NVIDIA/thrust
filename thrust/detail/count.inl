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


/*! \file count.inl
 *  \brief Inline file for count.h.
 */

#include <thrust/count.h>
#include <thrust/functional.h>
#include <thrust/transform_reduce.h>

namespace thrust
{

namespace detail
{

template <typename InputType1, typename InputType2, typename CountType>
struct count_make_predicate
{
  __host__ __device__ 
  count_make_predicate(const InputType2 &val) : rhs(val){}

  __host__ __device__
  CountType operator()(const InputType1 &lhs) const 
  {
    if(lhs == rhs)
      return 1;
    else
      return 0;
  } // end operator()()

  const InputType2 rhs;
}; // end count_make_predicate


// TODO fix this
template <typename InputType, typename Predicate, typename CountType>
struct count_predicate
{
  __host__ __device__ 
  count_predicate(const Predicate& pred) : op(pred){}

  __host__ __device__
  CountType operator()(const InputType &val) const 
  {
    if(op(val))
      return 1;
    else
      return 0;
  } // end operator()

  const Predicate op;
}; // end count_predicate()

} // end detail

template <typename InputIterator, typename EqualityComparable>
typename thrust::iterator_traits<InputIterator>::difference_type
count(InputIterator first, InputIterator last, const EqualityComparable& value)
{
  typedef typename thrust::iterator_traits<InputIterator>::value_type InputType;
  typedef typename thrust::iterator_traits<InputIterator>::difference_type CountType;
  
  thrust::detail::count_make_predicate<InputType, EqualityComparable, CountType> op(value);
  thrust::plus<CountType> bin_op;
  return thrust::transform_reduce(first, last, op, CountType(0), bin_op);
} // end count()

template <typename InputIterator, typename Predicate>
typename thrust::iterator_traits<InputIterator>::difference_type
count_if(InputIterator first, InputIterator last, Predicate pred)
{
  typedef typename thrust::iterator_traits<InputIterator>::value_type InputType;
  typedef typename thrust::iterator_traits<InputIterator>::difference_type CountType;
  
  thrust::detail::count_predicate<InputType, Predicate, CountType> op(pred);
  thrust::plus<CountType> bin_op;
  return thrust::transform_reduce(first, last, op, CountType(0), bin_op);
} // end count_if()

}; // end thrust

