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

#include <thrust/detail/config.h>
#include <thrust/pair.h>

namespace thrust
{
namespace system
{
namespace cuda
{
namespace detail
{


template<typename Size>
class trivial_decomposition
{
  public:
    typedef Size size_type;

    typedef thrust::pair<size_type,size_type> range;

    __host__ __device__
    trivial_decomposition()
      : m_n(0)
    {}

    __host__ __device__
    trivial_decomposition(size_type n)
      : m_n(n)
    {}

    __host__ __device__
    range operator[](size_type) const
    {
      return range(0, n());
    }

    __host__ __device__
    size_type size() const
    {
      return 1;
    }

    // XXX think of a better name for this
    __host__ __device__
    size_type n() const
    {
      return m_n;
    }

  private:
    Size m_n;
};


template<typename Size>
__host__ __device__
trivial_decomposition<Size> make_trivial_decomposition(Size n)
{
  return trivial_decomposition<Size>(n);
}


template<typename Size>
class blocked_decomposition
{
  public:
    typedef Size size_type;

    typedef thrust::pair<size_type,size_type> range;

    __host__ __device__
    blocked_decomposition()
      : m_n(0),
        m_block_size(0),
        m_num_partitions(0)
    {}

    __host__ __device__
    blocked_decomposition(size_type n, Size block_size)
      : m_n(n),
        m_block_size(block_size),
        m_num_partitions((n + block_size - 1) / block_size)
    {}

    __host__ __device__
    range operator[](size_type i) const
    {
      size_type first = i * m_block_size;
      size_type last  = thrust::min(m_n, first + m_block_size);

      return range(first, last);
    }

    __host__ __device__
    size_type size() const
    {
      return m_num_partitions;
    }

    // XXX think of a better name for this
    __host__ __device__
    size_type n() const
    {
      return m_n;
    }

  private:
    Size m_n;
    Size m_block_size;
    Size m_num_partitions;
};


template<typename Size>
__host__ __device__
blocked_decomposition<Size> make_blocked_decomposition(Size n, Size block_size)
{
  return blocked_decomposition<Size>(n,block_size);
}


template<typename Size>
class uniform_decomposition
  : public blocked_decomposition<Size>
{
  private:
    typedef blocked_decomposition<Size> super_t;

  public:
    __host__ __device__
    uniform_decomposition()
      : super_t()
    {}

    __host__ __device__
    uniform_decomposition(Size n, Size num_partitions)
      : super_t(n, n / num_partitions)
    {}
};


template<typename Size>
__host__ __device__
uniform_decomposition<Size> make_uniform_decomposition(Size n, Size num_partitions)
{
  return uniform_decomposition<Size>(n,num_partitions);
}


template<typename Size>
class aligned_decomposition
{
  public:
    typedef Size size_type;

    typedef thrust::pair<size_type,size_type> range;

    __host__ __device__
    aligned_decomposition()
      : m_n(0),
        m_num_partitions(0),
        m_tile_size(0)
    {}

    __host__ __device__
    aligned_decomposition(Size n, Size num_partitions, Size aligned_size)
      : m_n(n),
        m_num_partitions(num_partitions),
        m_tile_size(aligned_size)
    {
      size_type num_tiles = (n + m_tile_size - 1) / m_tile_size;

      m_num_tiles_per_partition = num_tiles / size();
      m_last_partial_tile_size  =  num_tiles % size();
    }

    __host__ __device__
    range operator[](Size i) const
    {
      range result = range_in_tiles(i);
      result.first *= m_tile_size;
      result.second = thrust::min<size_type>(m_n, result.second * m_tile_size);
      return result;
    }

    __host__ __device__
    size_type size() const
    {
      return m_num_partitions;
    }

    // XXX think of a better name for this
    __host__ __device__
    size_type n() const
    {
      return m_n;
    }

  private:
    __host__ __device__
    range range_in_tiles(size_type i) const
    {
      range result;

      result.first = m_num_tiles_per_partition * i;
      result.first += thrust::min<size_type>(i, m_last_partial_tile_size);

      result.second = result.first + m_num_tiles_per_partition + (i < m_last_partial_tile_size);

      return result;
    }

    size_type m_n;
    size_type m_num_partitions;
    size_type m_num_tiles_per_partition;
    size_type m_tile_size;
    size_type m_last_partial_tile_size;
};


template<typename Size>
__host__ __device__
aligned_decomposition<Size> make_aligned_decomposition(Size n, Size num_partitions, Size aligned_size)
{
  return aligned_decomposition<Size>(n,num_partitions,aligned_size);
}


} // end detail
} // end cuda
} // end system
} // end thrust

