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


/*! \file remove.inl
 *  \brief Inline file for remove.h.
 */

#include <thrust/remove.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/detail/dispatch/remove.h>

namespace thrust
{

namespace detail
{

template<typename T>
  struct equal_to_value
{
  equal_to_value(const T &v):value(v){}

  __host__ __device__
  inline bool operator()(const T &x) { return x == value; }

  const T value;
}; // end equal_to_value

} // end detail

template<typename ForwardIterator,
         typename T>
  ForwardIterator remove(ForwardIterator begin,
                         ForwardIterator end,
                         const T &value)
{
  detail::equal_to_value<T> pred(value);

  return thrust::remove_if(begin,end,pred);
} // end remove()

template<typename InputIterator,
         typename OutputIterator,
         typename T>
  OutputIterator remove_copy(InputIterator begin,
                             InputIterator end,
                             OutputIterator result,
                             const T &value)
{
  detail::equal_to_value<T> pred(value);

  return thrust::remove_copy_if(begin,end,result,pred);
} // end remove_copy()

template<typename ForwardIterator,
         typename Predicate>
  ForwardIterator remove_if(ForwardIterator begin,
                            ForwardIterator end,
                            Predicate pred)
{
  return detail::dispatch::remove_if(begin, end, pred,
    typename thrust::iterator_traits<ForwardIterator>::iterator_category());
} // end remove_if()

template<typename InputIterator,
         typename OutputIterator,
         typename Predicate>
  OutputIterator remove_copy_if(InputIterator begin,
                                InputIterator end,
                                OutputIterator result,
                                Predicate pred)
{
  return detail::dispatch::remove_copy_if(begin, end, result, pred,
    typename thrust::iterator_traits<InputIterator>::iterator_category(),
    typename thrust::iterator_traits<OutputIterator>::iterator_category());
} // end remove_copy_if()

} // end thrust

