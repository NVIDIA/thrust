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

#include <thrust/range/begin.h>
#include <thrust/range/end.h>

namespace thrust
{

namespace detail
{


template<typename Iterator>
  segmented_iterator<Iterator>
    ::segmented_iterator(void)
      : m_current_bucket(),
        m_buckets_end(),
        m_current()
{
} // end segmented_iterator::segmented_iterator()


template<typename Iterator>
  template<typename OtherIterator>
    segmented_iterator<Iterator>
      ::segmented_iterator(OtherIterator first, OtherIterator last)
        : m_current_bucket(first),
          m_buckets_end(last)
{
  // skip past empty buckets
  while(m_current_bucket != m_buckets_end && thrust::experimental::empty(*m_current_bucket))
  {
    ++m_current_bucket;
  } // end while

  if(m_current_bucket != m_buckets_end)
  {
    m_current = thrust::experimental::begin(*m_current_bucket);
  } // end if
} // end segmented_iterator::segmented_iterator()


template<typename Iterator>
  segmented_iterator<Iterator>
    ::segmented_iterator(const segmented_iterator &x)
      : m_current_bucket(x.m_current_bucket),
        m_buckets_end(x.m_buckets_end),
        m_current(x.m_current)
{
} // end segmented_iterator::segmented_iterator()


template<typename Iterator>
  typename segmented_iterator<Iterator>::reference
    segmented_iterator<Iterator>
      ::dereference(void) const
{
  return *m_current;
} // end segmented_iterator::dereference()


template<typename Iterator>
  bool segmented_iterator<Iterator>
    ::equal(const segmented_iterator &x) const
{
  // check the end of the buckets range, if they are equal, compare the current buckets
  // the iterators are equal if we're at the end of the range or if the local iterators match
  return (m_buckets_end == x.m_buckets_end) && (m_current_bucket == x.m_current_bucket)
    && ((m_current_bucket == m_buckets_end) || (m_current == x.m_current));
} // end segmented_iterator::equal()


template<typename Iterator>
  void segmented_iterator<Iterator>
    ::increment(void)
{
  // increment the local iterator
  ++m_current;

  // skip past the current bucket if we're at the end and all empty buckets
  while(m_current == thrust::experimental::end(*m_current_bucket))
  {
    // increment the bucket iterator
    ++m_current_bucket;

    // have we exhausted all buckets?
    if(m_current_bucket == m_buckets_end) break;

    // point to the beginning of the current bucket
    m_current = thrust::experimental::begin(*m_current_bucket);
  } // end while
} // end segmented_iterator::increment()


template<typename Iterator>
  void segmented_iterator<Iterator>
    ::decrement(void)
{
  // skip past the current bucket if we're at the beginning and all empty buckets
  while(m_current_bucket == m_buckets_end || m_current == thrust::experimental::begin(*m_current_bucket))
  {
    --m_current_bucket;
    m_current = thrust::experimental::end(*m_current_bucket);
  } // end while

  // decrement the local iterator
  --m_current_bucket;
} // end segmented_iterator::decrement()


} // end detail

} // end thrust

