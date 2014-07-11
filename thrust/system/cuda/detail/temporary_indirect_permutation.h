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

#include <thrust/detail/temporary_array.h>
#include <thrust/sequence.h>
#include <thrust/gather.h>
#include <thrust/detail/function.h>

namespace thrust
{
namespace system
{
namespace cuda
{
namespace detail
{


template<typename DerivedPolicy, typename RandomAccessIterator>
  struct temporary_indirect_permutation
{
  private:
    typedef unsigned int size_type;
    typedef thrust::detail::temporary_array<size_type, DerivedPolicy> array_type;

  public:
    __host__ __device__
    temporary_indirect_permutation(thrust::execution_policy<DerivedPolicy> &exec, RandomAccessIterator first, RandomAccessIterator last)
      : m_exec(derived_cast(exec)),
        m_src_first(first),
        m_src_last(last),
        m_permutation(0, m_exec, last - first)
    {
      // generate sorted index sequence
      thrust::sequence(exec, m_permutation.begin(), m_permutation.end());
    }

    __host__ __device__
    ~temporary_indirect_permutation()
    {
      // permute the source array using the indices
      typedef typename thrust::iterator_value<RandomAccessIterator>::type value_type;
      thrust::detail::temporary_array<value_type, DerivedPolicy> temp(m_exec, m_src_first, m_src_last);
      thrust::gather(m_exec, m_permutation.begin(), m_permutation.end(), temp.begin(), m_src_first);
    }

    typedef typename array_type::iterator iterator;

    __host__ __device__
    iterator begin()
    {
      return m_permutation.begin();
    }

    __host__ __device__
    iterator end()
    {
      return m_permutation.end();
    }

  private:
    DerivedPolicy &m_exec;
    RandomAccessIterator m_src_first, m_src_last;
    thrust::detail::temporary_array<size_type, DerivedPolicy> m_permutation;
};


template<typename DerivedPolicy, typename RandomAccessIterator>
  struct iterator_range_with_execution_policy
{
  __host__ __device__
  iterator_range_with_execution_policy(thrust::execution_policy<DerivedPolicy> &exec, RandomAccessIterator first, RandomAccessIterator last)
    : m_exec(derived_cast(exec)), m_first(first), m_last(last)
  {}

  typedef RandomAccessIterator iterator;

  __host__ __device__
  iterator begin()
  {
    return m_first;
  }

  __host__ __device__
  iterator end()
  {
    return m_last;
  }

  __host__ __device__
  DerivedPolicy &exec()
  {
    return m_exec;
  }

  DerivedPolicy &m_exec;
  RandomAccessIterator m_first, m_last;
};


template<typename Condition, typename DerivedPolicy, typename RandomAccessIterator>
  struct conditional_temporary_indirect_permutation
    : thrust::detail::eval_if<
        Condition::value,
        thrust::detail::identity_<temporary_indirect_permutation<DerivedPolicy, RandomAccessIterator> >,
        thrust::detail::identity_<iterator_range_with_execution_policy<DerivedPolicy, RandomAccessIterator> >
      >::type
{
  typedef typename thrust::detail::eval_if<
    Condition::value,
    thrust::detail::identity_<temporary_indirect_permutation<DerivedPolicy, RandomAccessIterator> >,
    thrust::detail::identity_<iterator_range_with_execution_policy<DerivedPolicy, RandomAccessIterator> >
  >::type super_t;

  __host__ __device__
  conditional_temporary_indirect_permutation(thrust::execution_policy<DerivedPolicy> &exec, RandomAccessIterator first, RandomAccessIterator last)
    : super_t(exec, first, last)
  {}
};


template<typename DerivedPolicy, typename RandomAccessIterator, typename Compare>
  struct temporary_indirect_ordering
    : temporary_indirect_permutation<DerivedPolicy,RandomAccessIterator>
{
  private:
    typedef temporary_indirect_permutation<DerivedPolicy,RandomAccessIterator> super_t;

  public:
    __host__ __device__
    temporary_indirect_ordering(thrust::execution_policy<DerivedPolicy> &exec, RandomAccessIterator first, RandomAccessIterator last, Compare comp)
      : super_t(exec, first, last),
        m_comp(first, comp)
    {}

    struct compare
    {
      RandomAccessIterator first;

      thrust::detail::wrapped_function<
        Compare,
        bool
      > comp;

      __host__ __device__
      compare(RandomAccessIterator first, Compare comp)
        : first(first), comp(comp)
      {}

      template<typename Integral>
      __host__ __device__
      bool operator()(Integral a, Integral b)
      {
        return comp(first[a], first[b]);
      }
    };

    __host__ __device__
    compare comp() const
    {
      return m_comp;
    }

  private:
    compare m_comp;
};


template<typename DerivedPolicy, typename RandomAccessIterator, typename Compare>
  struct iterator_range_with_execution_policy_and_compare
    : iterator_range_with_execution_policy<DerivedPolicy, RandomAccessIterator>
{
  typedef iterator_range_with_execution_policy<DerivedPolicy, RandomAccessIterator> super_t;

  __host__ __device__
  iterator_range_with_execution_policy_and_compare(thrust::execution_policy<DerivedPolicy> &exec, RandomAccessIterator first, RandomAccessIterator last, Compare comp)
    : super_t(exec, first, last), m_comp(comp)
  {}

  typedef Compare compare;

  __host__ __device__
  compare comp()
  {
    return m_comp;
  }

  Compare m_comp;
};


template<typename Condition, typename DerivedPolicy, typename RandomAccessIterator, typename Compare>
  struct conditional_temporary_indirect_ordering
    : thrust::detail::eval_if<
        Condition::value,
        thrust::detail::identity_<temporary_indirect_ordering<DerivedPolicy, RandomAccessIterator, Compare> >,
        thrust::detail::identity_<iterator_range_with_execution_policy_and_compare<DerivedPolicy, RandomAccessIterator, Compare> >
      >::type
{
  typedef typename thrust::detail::eval_if<
    Condition::value,
    thrust::detail::identity_<temporary_indirect_ordering<DerivedPolicy, RandomAccessIterator, Compare> >,
    thrust::detail::identity_<iterator_range_with_execution_policy_and_compare<DerivedPolicy, RandomAccessIterator, Compare> >
  >::type super_t;

  __host__ __device__
  conditional_temporary_indirect_ordering(thrust::execution_policy<DerivedPolicy> &exec, RandomAccessIterator first, RandomAccessIterator last, Compare comp)
    : super_t(exec, first, last, comp)
  {}
};


} // end detail
} // end cuda
} // end system
} // end thrust

