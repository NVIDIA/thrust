/*
 *  Copyright 2008-2010 NVIDIA Corporation
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

/*! \file segmented_iterator.h
 *  \brief An iterator that iterates across a range of ranges.
 *         Inspired by the implementation in ASL.
 */

#include <thrust/detail/config.h>
#include <thrust/iterator/detail/segmented_iterator_base.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/range/detail/iterator.h>

namespace thrust
{

namespace detail
{

// an iterator the iterates across a range of ranges
template<typename Iterator>
  class segmented_iterator
    : public segmented_iterator_base<Iterator>::type
{
  public:
    __host__ __device__
    inline segmented_iterator(void);

    // XXX this needs an enable_if_convertible<OtherIterator,Iterator>
    template<typename OtherIterator>
    __host__ __device__
    inline segmented_iterator(OtherIterator first, OtherIterator last);

    __host__ __device__
    inline segmented_iterator(const segmented_iterator &x);

  private:
    // shorthand parent types
    typedef typename segmented_iterator_base<Iterator>::type super_t;
    typedef typename super_t::reference                      reference;

    // an iterator that iterates across buckets
    typedef Iterator                                         bucket_iterator;

    // an iterator that iterates locally to a bucket
    typedef typename thrust::experimental::range_iterator<
      typename iterator_value<bucket_iterator>::type
    >::type                                                  local_iterator;

    // the current bucket
    bucket_iterator m_current_bucket;

    // the end of the buckets range
    bucket_iterator m_buckets_end;

    // the current element within the current bucket
    local_iterator  m_current;

    // iterator core interface follows
    
    friend class thrust::experimental::iterator_core_access;
    
    reference dereference(void) const;

    __host__ __device__
    bool equal(const segmented_iterator &x) const;

    __host__ __device__
    void increment(void);

    __host__ __device__
    void decrement(void);
}; // end segmented_iterator

} // end detail

} // end thrust

#include <thrust/iterator/detail/segmented_iterator.inl>

