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

#include <thrust/iterator/permutation_iterator.h>

namespace thrust
{


namespace detail
{

// XXX remove this when we no longer need device::dereference
struct permutation_iterator_friend
{
  template<typename ElementIterator, typename IndexIterator>
    static inline __host__ __device__
      typename backend::dereference_result< thrust::permutation_iterator<ElementIterator,IndexIterator> >::type
        dereference(const thrust::permutation_iterator<ElementIterator,IndexIterator> &iter)
  {
    return thrust::detail::backend::dereference(iter.m_element_iterator + thrust::detail::backend::dereference(iter.base()));
  }
};




namespace backend
{


template<typename ElementIterator, typename IndexIterator>
  struct dereference_result< thrust::permutation_iterator<ElementIterator, IndexIterator> >
    : dereference_result<ElementIterator>
{
}; // end dereference_result


template<typename ElementIterator, typename IndexIterator>
  inline __host__ __device__
    typename dereference_result< thrust::permutation_iterator<ElementIterator, IndexIterator> >::type
      dereference(const thrust::permutation_iterator<ElementIterator, IndexIterator> &iter)
{
  return permutation_iterator_friend::dereference(iter);
} // end dereference()


template<typename ElementIterator, typename IndexIterator, typename IndexType>
  inline __host__ __device__
    typename dereference_result< thrust::permutation_iterator<ElementIterator, IndexIterator> >::type
      dereference(const thrust::permutation_iterator<ElementIterator, IndexIterator> &iter, IndexType n)
{
  // XXX the result of these operations is undefined if dereference() returns a reference to
  //     a member of iter
  iter += n;
  return dereference(iter);
} // end dereference()


} // end backend

} // end detail

} // end thrust

