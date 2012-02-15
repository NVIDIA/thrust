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

/*! \file forced_iterator.h
 *  \brief Adapts an existing iterator but forces the space or traversal type.
 */

#include <thrust/iterator/iterator_adaptor.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/detail/backend/dereference.h>

namespace thrust
{

namespace detail
{

template <typename,typename> class forced_iterator;

template<typename Iterator, typename Space>
  struct forced_iterator_base
{
  typedef thrust::experimental::iterator_adaptor<
    forced_iterator<Iterator,Space>,
    Iterator,
    typename thrust::iterator_pointer<Iterator>::type,
    typename thrust::iterator_value<Iterator>::type,
    Space,
    typename thrust::iterator_traversal<Iterator>::type,
    typename thrust::iterator_reference<Iterator>::type
  > type;
}; // end forced_iterator_base

template<typename Iterator, typename Space>
  class forced_iterator
    : public forced_iterator_base<Iterator,Space>::type
{
  private:
    typedef typename forced_iterator_base<Iterator,Space>::type super_t;

    friend class thrust::experimental::iterator_core_access;

  public:
    __host__ __device__
    forced_iterator(void) {}

    __host__ __device__
    explicit forced_iterator(Iterator x)
      : super_t(x) {}
}; // end forced_iterator

template <typename Iterator, typename Space>
  forced_iterator<Iterator,Space>
    make_forced_iterator(Iterator x, Space)
{
  return forced_iterator<Iterator,Space>(x);
} // end make_forced_iterator


namespace backend
{


template<typename Iterator, typename Space>
  struct dereference_result< thrust::detail::forced_iterator<Iterator,Space> >
    : dereference_result<Iterator>
{
}; // end dereference_result


template<typename Iterator, typename Space>
  inline __host__ __device__
    typename dereference_result< thrust::detail::forced_iterator<Iterator,Space> >::type
      dereference(const thrust::detail::forced_iterator<Iterator,Space> &iter)
{
  return dereference(iter.base());
} // end dereference()


template<typename Iterator, typename Space, typename IndexType>
  inline __host__ __device__
    typename dereference_result< thrust::detail::forced_iterator<Iterator,Space> >::type
      dereference(const thrust::detail::forced_iterator<Iterator,Space> &iter, IndexType n)
{
  return dereference(iter.base(), n);
} // end dereference()


} // end backend

} // end detail

} // end thrust

