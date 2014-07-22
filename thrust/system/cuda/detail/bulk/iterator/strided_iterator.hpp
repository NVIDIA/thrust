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
#include <thrust/iterator/iterator_adaptor.h>


BULK_NAMESPACE_PREFIX
namespace bulk
{


template<typename Iterator,
         typename Size = typename thrust::iterator_difference<Iterator>::type>
class strided_iterator
  : public thrust::iterator_adaptor<
      strided_iterator<Iterator>,
      Iterator
    >
{
  private:
    typedef thrust::iterator_adaptor<strided_iterator<Iterator>,Iterator> super_t;

  public:
    typedef Size stride_type;

    inline __host__ __device__
    strided_iterator()
      : super_t(), m_stride(1)
    {}

    inline __host__ __device__
    strided_iterator(const strided_iterator& other)
      : super_t(other), m_stride(other.m_stride)
    {}

    inline __host__ __device__
    strided_iterator(const Iterator &base, stride_type stride)
      : super_t(base), m_stride(stride)
    {}

    inline __host__ __device__
    stride_type stride() const
    {
      return m_stride;
    }

  private:
    friend class thrust::iterator_core_access;

    __host__ __device__
    void increment()
    {
      super_t::base_reference() += stride();
    }

    __host__ __device__
    void decrement()
    {
      super_t::base_reference() -= stride();
    }

    __host__ __device__
    void advance(typename super_t::difference_type n)
    {
      super_t::base_reference() += n * stride();
    }

    template<typename OtherIterator>
    __host__ __device__
    typename super_t::difference_type distance_to(const strided_iterator<OtherIterator> &other) const
    {
      if(other.base() >= this->base())
      {
        return (other.base() - this->base() + (stride() - 1)) / stride();
      }

      return (other.base() - this->base() - (stride() - 1)) / stride();
    }

    stride_type m_stride;
};


template<typename Iterator, typename Size>
__host__ __device__
strided_iterator<Iterator,Size> make_strided_iterator(Iterator iter, Size stride)
{
  return strided_iterator<Iterator,Size>(iter, stride);
}


} // end bulk
BULK_NAMESPACE_SUFFIX

