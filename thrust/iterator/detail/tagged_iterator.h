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

#include <thrust/detail/config.h>
#include <thrust/iterator/iterator_adaptor.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/detail/backend/dereference.h>
#include <thrust/detail/use_default.h>

namespace thrust
{
namespace experimental
{

// forward declaration of iterator_adaptor
template <typename Derived,
          typename Base,
          typename Pointer,
          typename Value,
          typename Space,
          typename Traversal,
          typename Reference,
          typename Difference
>
class iterator_adaptor;

} // end experimental

namespace detail
{

template <typename,typename> class tagged_iterator;

template<typename Iterator, typename Tag>
  struct tagged_iterator_base
{
  typedef thrust::experimental::iterator_adaptor<
    tagged_iterator<Iterator,Tag>,
    Iterator,
    typename thrust::iterator_pointer<Iterator>::type,
    typename thrust::iterator_value<Iterator>::type,
    Tag,
    typename thrust::iterator_traversal<Iterator>::type,
    typename thrust::iterator_reference<Iterator>::type,
    typename thrust::iterator_difference<Iterator>::type
  > type;
}; // end tagged_iterator_base

template<typename Iterator, typename Tag>
  class tagged_iterator
    : public tagged_iterator_base<Iterator,Tag>::type
{
  private:
    typedef typename tagged_iterator_base<Iterator,Tag>::type super_t;

  public:
    __host__ __device__
    tagged_iterator(void) {}

    __host__ __device__
    explicit tagged_iterator(Iterator x)
      : super_t(x) {}
}; // end tagged_iterator


namespace backend
{


template<typename Iterator, typename Tag>
  struct dereference_result< thrust::detail::tagged_iterator<Iterator,Tag> >
    : dereference_result<Iterator>
{
}; // end dereference_result


template<typename Iterator, typename Tag>
  inline __host__ __device__
    typename dereference_result< thrust::detail::tagged_iterator<Iterator,Tag> >::type
      dereference(const thrust::detail::tagged_iterator<Iterator,Tag> &iter)
{
  return dereference(iter.base());
} // end dereference()


template<typename Iterator, typename Tag, typename IndexType>
  inline __host__ __device__
    typename dereference_result< thrust::detail::tagged_iterator<Iterator,Tag> >::type
      dereference(const thrust::detail::tagged_iterator<Iterator,Tag> &iter, IndexType n)
{
  return dereference(iter.base(), n);
} // end dereference()


} // end backend

} // end detail

} // end thrust

