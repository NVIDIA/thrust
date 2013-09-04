/*
 *  Copyright 2008-2013 NVIDIA Corporation
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

#include <thrust/system/cuda/detail/bulk/detail/config.hpp>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/tuple.h>
#include <thrust/functional.h>


BULK_NAMESPACE_PREFIX
namespace bulk
{
namespace detail
{


template<typename RandomAccessIterator,
         typename BinaryPredicate = thrust::equal_to<typename thrust::iterator_value<RandomAccessIterator>::type>,
         typename ValueType = bool,
         typename IndexType = typename thrust::iterator_difference<RandomAccessIterator>::type>
//  class tail_flags
  class tail_flags_
{
  // XXX WAR cudafe issue
  //private:
  public:
    struct tail_flag_functor
    {
      BinaryPredicate binary_pred; // this must be the first member for performance reasons
      IndexType n;

      typedef ValueType result_type;

      __host__ __device__
      tail_flag_functor(IndexType n)
        : binary_pred(), n(n)
      {}

      __host__ __device__
      tail_flag_functor(IndexType n, BinaryPredicate binary_pred)
        : binary_pred(binary_pred), n(n)
      {}

      template<typename Tuple>
      __host__ __device__ __thrust_forceinline__
      result_type operator()(const Tuple &t)
      {
        const IndexType i = thrust::get<0>(t);

        // note that we do not dereference the tuple's 2nd element when i >= n
        // and therefore do not dereference a bad location at the boundary
        return (i == (n - 1) || !binary_pred(thrust::get<1>(t), thrust::get<2>(t)));
      }
    };

    typedef thrust::counting_iterator<IndexType> counting_iterator;

  public:
    typedef thrust::transform_iterator<
      tail_flag_functor,
      thrust::zip_iterator<thrust::tuple<counting_iterator,RandomAccessIterator,RandomAccessIterator> >
    > iterator;

    __bulk_hd_warning_disable__
    __host__ __device__
    //tail_flags(RandomAccessIterator first, RandomAccessIterator last)
    tail_flags_(RandomAccessIterator first, RandomAccessIterator last)
      : m_begin(thrust::make_transform_iterator(thrust::make_zip_iterator(thrust::make_tuple(thrust::counting_iterator<IndexType>(0), first, first + 1)),
                                                tail_flag_functor(last - first))),
        m_end(m_begin + (last - first))
    {}

    __bulk_hd_warning_disable__
    __host__ __device__
    //tail_flags(RandomAccessIterator first, RandomAccessIterator last, BinaryPredicate binary_pred)
    tail_flags_(RandomAccessIterator first, RandomAccessIterator last, BinaryPredicate binary_pred)
      : m_begin(thrust::make_transform_iterator(thrust::make_zip_iterator(thrust::make_tuple(thrust::counting_iterator<IndexType>(0), first, first + 1)),
                                                tail_flag_functor(last - first, binary_pred))),
        m_end(m_begin + (last - first))
    {}

    __host__ __device__
    iterator begin() const
    {
      return m_begin;
    }

    __host__ __device__
    iterator end() const
    {
      return m_end;
    }

    template<typename OtherIndex>
    __host__ __device__
    typename iterator::reference operator[](OtherIndex i)
    {
      return *(begin() + i);
    }

  private:
    iterator m_begin, m_end;
};


template<typename RandomAccessIterator, typename BinaryPredicate>
__host__ __device__
//tail_flags<RandomAccessIterator, BinaryPredicate>
tail_flags_<RandomAccessIterator, BinaryPredicate>
  make_tail_flags(RandomAccessIterator first, RandomAccessIterator last, BinaryPredicate binary_pred)
{
//  return tail_flags<RandomAccessIterator, BinaryPredicate>(first, last, binary_pred);
  return tail_flags_<RandomAccessIterator, BinaryPredicate>(first, last, binary_pred);
}


template<typename RandomAccessIterator>
__host__ __device__
//tail_flags<RandomAccessIterator>
tail_flags_<RandomAccessIterator>
  make_tail_flags(RandomAccessIterator first, RandomAccessIterator last)
{
//  return tail_flags<RandomAccessIterator>(first, last);
  return tail_flags_<RandomAccessIterator>(first, last);
}


} // end detail
} // end bulk
BULK_NAMESPACE_SUFFIX

