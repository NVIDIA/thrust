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


template<typename System, typename RandomAccessIterator>
  struct temporary_indirect_permutation
{
  private:
    typedef unsigned int size_type;
    typedef thrust::detail::temporary_array<size_type, System> array_type;

  public:
    temporary_indirect_permutation(System &system, RandomAccessIterator first, RandomAccessIterator last)
      : m_system(system),
        m_src_first(first),
        m_src_last(last),
        m_permutation(0, m_system, last - first)
    {
      // generate sorted index sequence
      thrust::sequence(system, m_permutation.begin(), m_permutation.end());
    }

    ~temporary_indirect_permutation()
    {
      // permute the source array using the indices
      typedef typename thrust::iterator_value<RandomAccessIterator>::type value_type;
      thrust::detail::temporary_array<value_type, System> temp(m_system, m_src_first, m_src_last);
      thrust::gather(m_system, m_permutation.begin(), m_permutation.end(), temp.begin(), m_src_first);
    }

    typedef typename array_type::iterator iterator;

    iterator begin()
    {
      return m_permutation.begin();
    }

    iterator end()
    {
      return m_permutation.end();
    }

  private:
    System &m_system;
    RandomAccessIterator m_src_first, m_src_last;
    thrust::detail::temporary_array<size_type, System> m_permutation;
};


template<typename System, typename RandomAccessIterator>
  struct iterator_range_with_system
{
  iterator_range_with_system(System &system, RandomAccessIterator first, RandomAccessIterator last)
    : m_system(system), m_first(first), m_last(last)
  {}

  typedef RandomAccessIterator iterator;

  iterator begin()
  {
    return m_first;
  }

  iterator end()
  {
    return m_last;
  }

  System &system()
  {
    return m_system;
  }

  System &m_system;
  RandomAccessIterator m_first, m_last;
};


template<typename Condition, typename System, typename RandomAccessIterator>
  struct conditional_temporary_indirect_permutation
    : thrust::detail::eval_if<
        Condition::value,
        thrust::detail::identity_<temporary_indirect_permutation<System, RandomAccessIterator> >,
        thrust::detail::identity_<iterator_range_with_system<System, RandomAccessIterator> >
      >::type
{
  typedef typename thrust::detail::eval_if<
    Condition::value,
    thrust::detail::identity_<temporary_indirect_permutation<System, RandomAccessIterator> >,
    thrust::detail::identity_<iterator_range_with_system<System, RandomAccessIterator> >
  >::type super_t;

  conditional_temporary_indirect_permutation(System &system, RandomAccessIterator first, RandomAccessIterator last)
    : super_t(system, first, last)
  {}
};


template<typename System, typename RandomAccessIterator, typename Compare>
  struct temporary_indirect_ordering
    : temporary_indirect_permutation<System,RandomAccessIterator>
{
  private:
    typedef temporary_indirect_permutation<System,RandomAccessIterator> super_t;

  public:
    temporary_indirect_ordering(System &system, RandomAccessIterator first, RandomAccessIterator last, Compare comp)
      : super_t(system, first, last),
        m_comp(first, comp)
    {}

    struct compare
    {
      RandomAccessIterator first;

      thrust::detail::host_device_function<
        Compare,
        bool
      > comp;

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

    compare comp() const
    {
      return m_comp;
    }

  private:
    compare m_comp;
};


template<typename System, typename RandomAccessIterator, typename Compare>
  struct iterator_range_with_system_and_compare
    : iterator_range_with_system<System, RandomAccessIterator>
{
  typedef iterator_range_with_system<System, RandomAccessIterator> super_t;

  iterator_range_with_system_and_compare(System &system, RandomAccessIterator first, RandomAccessIterator last, Compare comp)
    : super_t(system, first, last), m_comp(comp)
  {}

  typedef Compare compare;

  compare comp()
  {
    return m_comp;
  }

  Compare m_comp;
};


template<typename Condition, typename System, typename RandomAccessIterator, typename Compare>
  struct conditional_temporary_indirect_ordering
    : thrust::detail::eval_if<
        Condition::value,
        thrust::detail::identity_<temporary_indirect_ordering<System, RandomAccessIterator, Compare> >,
        thrust::detail::identity_<iterator_range_with_system_and_compare<System, RandomAccessIterator, Compare> >
      >::type
{
  typedef typename thrust::detail::eval_if<
    Condition::value,
    thrust::detail::identity_<temporary_indirect_ordering<System, RandomAccessIterator, Compare> >,
    thrust::detail::identity_<iterator_range_with_system_and_compare<System, RandomAccessIterator, Compare> >
  >::type super_t;

  conditional_temporary_indirect_ordering(System &system, RandomAccessIterator first, RandomAccessIterator last, Compare comp)
    : super_t(system, first, last, comp)
  {}
};


} // end detail
} // end cuda
} // end system
} // end thrust

