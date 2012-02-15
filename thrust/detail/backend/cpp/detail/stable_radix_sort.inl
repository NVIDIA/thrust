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


#include <limits>

#include <thrust/copy.h>
#include <thrust/host_vector.h>
#include <thrust/functional.h>
#include <thrust/iterator/iterator_traits.h>

#include <thrust/detail/uninitialized_array.h>
#include <thrust/detail/cstdint.h>

namespace thrust
{
namespace detail
{
namespace backend
{
namespace cpp
{
namespace detail
{
namespace detail
{

template <typename T>
struct RadixEncoder : public thrust::identity<T>
{};

template <>
struct RadixEncoder<char> : public thrust::unary_function<char, unsigned char>
{
  unsigned char operator()(char x) const
  {
    if(std::numeric_limits<char>::is_signed)
      return x ^ static_cast<unsigned char>(1) << (8 * sizeof(unsigned char) - 1);
    else
      return x;
  }
};

template <>
struct RadixEncoder<signed char> : public thrust::unary_function<signed char, unsigned char>
{
  unsigned char operator()(signed char x) const
  {
    return x ^ static_cast<unsigned char>(1) << (8 * sizeof(unsigned char) - 1);
  }
};

template <>
struct RadixEncoder<short> : public thrust::unary_function<short, unsigned short>
{
  unsigned short operator()(short x) const
  {
    return x ^ static_cast<unsigned short>(1) << (8 * sizeof(unsigned short) - 1);
  }
};

template <>
struct RadixEncoder<int> : public thrust::unary_function<int, unsigned int>
{
  unsigned long operator()(long x) const
  {
    return x ^ static_cast<unsigned int>(1) << (8 * sizeof(unsigned int) - 1);
  }
};

template <>
struct RadixEncoder<long> : public thrust::unary_function<long, unsigned long>
{
  unsigned long operator()(long x) const
  {
    return x ^ static_cast<unsigned long>(1) << (8 * sizeof(unsigned long) - 1);
  }
};

template <>
struct RadixEncoder<long long> : public thrust::unary_function<long long, unsigned long long>
{
  unsigned long long operator()(long long x) const
  {
    return x ^ static_cast<unsigned long long>(1) << (8 * sizeof(unsigned long long) - 1);
  }
};

// ideally we'd use uint32 here and uint64 below
template <>
struct RadixEncoder<float> : public thrust::unary_function<float, thrust::detail::uint32_t>
{
  thrust::detail::uint32_t operator()(float x) const
  {
    union { float f; thrust::detail::uint32_t i; } u;
    u.f = x;
    thrust::detail::uint32_t mask = -static_cast<thrust::detail::int32_t>(u.i >> 31) | (static_cast<thrust::detail::uint32_t>(1) << 31);
    return u.i ^ mask;
  }
};

template <>
struct RadixEncoder<double> : public thrust::unary_function<double, thrust::detail::uint64_t>
{
  thrust::detail::uint64_t operator()(double x) const
  {
    union { double f; thrust::detail::uint64_t i; } u;
    u.f = x;
    thrust::detail::uint64_t mask = -static_cast<thrust::detail::int64_t>(u.i >> 63) | (static_cast<thrust::detail::uint64_t>(1) << 63);
    return u.i ^ mask;
  }
};


template <unsigned int RadixBits,
          bool HasValues,
          typename RandomAccessIterator1,
          typename RandomAccessIterator2,
          typename RandomAccessIterator3,
          typename RandomAccessIterator4>
void radix_sort(RandomAccessIterator1 keys1,
                RandomAccessIterator2 keys2,
                RandomAccessIterator3 vals1,
                RandomAccessIterator4 vals2,
                const size_t N)
{
  typedef typename thrust::iterator_value<RandomAccessIterator1>::type KeyType;

  typedef RadixEncoder<KeyType> Encoder;
  typedef typename Encoder::result_type EncodedType;

  static const unsigned int NumHistograms = (8 * sizeof(EncodedType) + (RadixBits - 1)) / RadixBits;
  static const unsigned int HistogramSize =  1 << RadixBits;

  static const EncodedType BitMask = static_cast<EncodedType>((1 << RadixBits) - 1);
  
  Encoder encode;

  // storage for histograms
  size_t histograms[NumHistograms][HistogramSize] = {{0}};

  // see which passes can be eliminated
  bool skip_shuffle[NumHistograms] = {false};
  
  // false if most recent data is stored in (keys1,vals1)
  bool flip = false;
    
  // compute histograms
  for (size_t i = 0; i < N; i++)
  {
    const EncodedType x = encode(keys1[i]);

    for (unsigned int j = 0; j < NumHistograms; j++)
    {
      const EncodedType BitShift = RadixBits * j;
      histograms[j][(x >> BitShift) & BitMask]++;
    }
  }

  // scan histograms
  for (unsigned int i = 0; i < NumHistograms; i++)
  {
    size_t sum = 0;

    for (unsigned int j = 0; j < HistogramSize; j++)
    {
      size_t bin = histograms[i][j];

      if (bin == N)
        skip_shuffle[i] = true;

      histograms[i][j] = sum;

      sum = sum + bin;
    }
  }

  // shuffle keys and (optionally) values 
  for (unsigned int i = 0; i < NumHistograms; i++)
  {
    const EncodedType BitShift = static_cast<EncodedType>(RadixBits * i);

    if (!skip_shuffle[i])
    {
      if (flip)
      {
        for (size_t j = 0; j < N; j++)
        {
          const EncodedType x = encode(keys2[j]);
          size_t position = histograms[i][(x >> BitShift) & BitMask]++;
          keys1[position] = keys2[j];
          if (HasValues)
            vals1[position] = vals2[j];
        }
      }
      else
      {
        for (size_t j = 0; j < N; j++)
        {
          const EncodedType x = encode(keys1[j]);
          size_t position = histograms[i][(x >> BitShift) & BitMask]++;
          keys2[position] = keys1[j];
          if (HasValues)
            vals2[position] = vals1[j];
        }
      }
        
      flip = (flip) ? false : true;
    }
  }
 
  // ensure final values are in (keys1,vals1)
  if (flip)
  {
    thrust::copy(keys2, keys2 + N, keys1);
    if (HasValues)
      thrust::copy(vals2, vals2 + N, vals1);
  }
}


// Select best radix sort parameters based on sizeof(T) and input size
// These particular values were determined through empirical testing on a Core i7 950 CPU
template <size_t KeySize>
struct radix_sort_dispatcher
{
};

template <>
struct radix_sort_dispatcher<1>
{
  template <typename RandomAccessIterator1, typename RandomAccessIterator2>
  void operator()(RandomAccessIterator1 keys1, RandomAccessIterator2 keys2, const size_t N)
  {
    detail::radix_sort<8,false>(keys1, keys2, static_cast<int *>(0), static_cast<int *>(0), N);
  }
  template <typename RandomAccessIterator1, typename RandomAccessIterator2, typename RandomAccessIterator3, typename RandomAccessIterator4>
  void operator()(RandomAccessIterator1 keys1, RandomAccessIterator2 keys2, RandomAccessIterator3 vals1, RandomAccessIterator4 vals2, const size_t N)
  {
    detail::radix_sort<8,true>(keys1, keys2, vals1, vals2, N);
  }
};

template <>
struct radix_sort_dispatcher<2>
{
  template <typename RandomAccessIterator1, typename RandomAccessIterator2>
  void operator()(RandomAccessIterator1 keys1, RandomAccessIterator2 keys2, const size_t N)
  {
    if (N < (1 << 16))
      detail::radix_sort<8,false>(keys1, keys2, static_cast<int *>(0), static_cast<int *>(0), N);
    else
      detail::radix_sort<16,false>(keys1, keys2, static_cast<int *>(0), static_cast<int *>(0), N);
  }
  template <typename RandomAccessIterator1, typename RandomAccessIterator2, typename RandomAccessIterator3, typename RandomAccessIterator4>
  void operator()(RandomAccessIterator1 keys1, RandomAccessIterator2 keys2, RandomAccessIterator3 vals1, RandomAccessIterator4 vals2, const size_t N)
  {
    if (N < (1 << 15))
      detail::radix_sort<8,true>(keys1, keys2, vals1, vals2, N);
    else
      detail::radix_sort<16,true>(keys1, keys2, vals1, vals2, N);
  }
};

template <>
struct radix_sort_dispatcher<4>
{
  template <typename RandomAccessIterator1, typename RandomAccessIterator2>
  void operator()(RandomAccessIterator1 keys1, RandomAccessIterator2 keys2, const size_t N)
  {
    if (N < (1 << 22))
      detail::radix_sort<8,false>(keys1, keys2, static_cast<int *>(0), static_cast<int *>(0), N);
    else
      detail::radix_sort<4,false>(keys1, keys2, static_cast<int *>(0), static_cast<int *>(0), N);
  }
  template <typename RandomAccessIterator1, typename RandomAccessIterator2, typename RandomAccessIterator3, typename RandomAccessIterator4>
  void operator()(RandomAccessIterator1 keys1, RandomAccessIterator2 keys2, RandomAccessIterator3 vals1, RandomAccessIterator4 vals2, const size_t N)
  {
    if (N < (1 << 22))
      detail::radix_sort<8,true>(keys1, keys2, vals1, vals2, N);
    else
      detail::radix_sort<3,true>(keys1, keys2, vals1, vals2, N);
  }
};

template <>
struct radix_sort_dispatcher<8>
{
  template <typename RandomAccessIterator1, typename RandomAccessIterator2>
  void operator()(RandomAccessIterator1 keys1, RandomAccessIterator2 keys2, const size_t N)
  {
    if (N < (1 << 21))
      detail::radix_sort<8,false>(keys1, keys2, static_cast<int *>(0), static_cast<int *>(0), N);
    else
      detail::radix_sort<4,false>(keys1, keys2, static_cast<int *>(0), static_cast<int *>(0), N);
  }
  template <typename RandomAccessIterator1, typename RandomAccessIterator2, typename RandomAccessIterator3, typename RandomAccessIterator4>
  void operator()(RandomAccessIterator1 keys1, RandomAccessIterator2 keys2, RandomAccessIterator3 vals1, RandomAccessIterator4 vals2, const size_t N)
  {
    if (N < (1 << 21))
      detail::radix_sort<8,true>(keys1, keys2, vals1, vals2, N);
    else
      detail::radix_sort<3,true>(keys1, keys2, vals1, vals2, N);
  }
};

template <typename RandomAccessIterator1,
          typename RandomAccessIterator2>
void radix_sort(RandomAccessIterator1 keys1,
                RandomAccessIterator2 keys2,
                const size_t N)
{
  typedef typename thrust::iterator_value<RandomAccessIterator1>::type KeyType;
  radix_sort_dispatcher<sizeof(KeyType)>()(keys1, keys2, N);
}

template <typename RandomAccessIterator1,
          typename RandomAccessIterator2,
          typename RandomAccessIterator3,
          typename RandomAccessIterator4>
void radix_sort(RandomAccessIterator1 keys1,
                RandomAccessIterator2 keys2,
                RandomAccessIterator3 vals1,
                RandomAccessIterator4 vals2,
                const size_t N)
{
  typedef typename thrust::iterator_value<RandomAccessIterator1>::type KeyType;
  radix_sort_dispatcher<sizeof(KeyType)>()(keys1, keys2, vals1, vals2, N);
}

} // namespace detail

//////////////
// Key Sort //
//////////////

template <typename RandomAccessIterator>
void stable_radix_sort(RandomAccessIterator first,
                       RandomAccessIterator last)
{
  typedef typename thrust::iterator_value<RandomAccessIterator>::type KeyType;

  size_t N = last - first;
  
  thrust::detail::uninitialized_array<KeyType, thrust::host_space_tag> temp(N);
  
  detail::radix_sort(first, temp.begin(), N);
}


////////////////////
// Key-Value Sort //
////////////////////

template <typename RandomAccessIterator1,
          typename RandomAccessIterator2>
void stable_radix_sort_by_key(RandomAccessIterator1 first1,
                              RandomAccessIterator1 last1,
                              RandomAccessIterator2 first2)
{
  typedef typename thrust::iterator_value<RandomAccessIterator1>::type KeyType;
  typedef typename thrust::iterator_value<RandomAccessIterator2>::type ValueType;

  size_t N = last1 - first1;
  
  thrust::detail::uninitialized_array<KeyType, thrust::host_space_tag> temp1(N);
  thrust::host_vector<ValueType>                                       temp2(N);

  detail::radix_sort(first1, temp1.begin(), first2, temp2.begin(), N);
}

} // end namespace detail
} // end namespace cpp
} // end namespace backend
} // end namespace detail
} // end namespace thrust

