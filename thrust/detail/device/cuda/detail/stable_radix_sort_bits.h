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

// TODO make this dispatch work the same way as the other parts of RS

namespace thrust
{
namespace detail
{
namespace device
{
namespace cuda
{
namespace detail
{

// Default encoding works for unsigned types
template <typename T>
struct encode_uint
{
    __host__ __device__ unsigned int operator()(const T& x) const {
        return x;
    }
};

template <typename T>
struct decode_uint
{
    __host__ __device__ T operator()(const unsigned int& x) const {
        return x;
    }
};


// 8 bit keys
template <>
struct encode_uint<signed char>
{
    __host__ __device__ unsigned int operator()(const signed char& x) const {
        return static_cast<unsigned int>(static_cast<unsigned char>(x ^ 0x80));
    }
};

template <>
struct decode_uint<signed char>
{
    __host__ __device__ signed char operator()(const unsigned int& x) const {
        return static_cast<signed char>(static_cast<unsigned char>(x) ^ 0x80);
    }
};

template <>
struct encode_uint<char>
{
    __host__ __device__ unsigned int operator()(const char& x) const {
        if( std::numeric_limits<char>::is_signed )
            return encode_uint<signed char>()(x);
        else
            return encode_uint<unsigned char>()(x);
    }
};

template <>
struct decode_uint<char>
{
    __host__ __device__ char operator()(const unsigned int& x) const {
        if( std::numeric_limits<char>::is_signed )
            return decode_uint<signed char>()(x);
        else
            return decode_uint<unsigned char>()(x);
    }
};

// 16-bit types
template <>
struct encode_uint<short>
{
    __host__ __device__ unsigned int operator()(const short& x) const {
        return static_cast<unsigned int>(static_cast<unsigned short>(x ^ 0x8000));
    }
};

template <>
struct decode_uint<short>
{
    __host__ __device__ short operator()(const unsigned int& x) const {
        return static_cast<short>(static_cast<unsigned short>(x) ^ 0x8000);
    }
};


// 32-bit keys
template <>
 struct encode_uint<int> 
{
    __host__ __device__ unsigned int operator()(const unsigned int& i) const {
	    return i ^ 0x80000000;
    }
};

template <>
 struct decode_uint<int>
{
    __host__ __device__ unsigned int operator()(const unsigned int& i) const {
	    return i ^ 0x80000000;
    }
};

template <>
 struct encode_uint<long> 
{
    __host__ __device__ unsigned int operator()(const long & i) const {
        // this is only called if sizeof(long) == sizeof(uint)
	    return i ^ 0x80000000;
    }
};

template <>
 struct decode_uint<long>
{
    __host__ __device__ unsigned int operator()(const long & i) const {
        // this is only called if sizeof(long) == sizeof(uint)
	    return i ^ 0x80000000;
    }
};

template <>
 struct encode_uint<float> 
{
    __host__ __device__ unsigned int operator()(const unsigned int & f) const {
        unsigned int mask = -int(f >> 31) | 0x80000000;
	    return f ^ mask;
    }
};

template <>
 struct decode_uint<float>
{
    __host__ __device__ unsigned int operator()(const unsigned int& f) const {
        unsigned int mask = ((f >> 31) - 1) | 0x80000000;
	    return f ^ mask;
    }
};

// 64-bit keys
template <typename T>
  struct lower_32_bits
{
  __host__ __device__ unsigned int operator()(const T &x) const {
      return x & 0xffffffff;
  }
};

template <typename T>
  struct upper_32_bits
{
  __host__ __device__ unsigned int operator()(const T &x) const {
      return x >> 32;
  }
};

__host__ __device__
inline unsigned long long flip_double(const unsigned long long& x){
    // we really want uint64 here
    const long long mask = -static_cast<long long>(x >> 63) | 0x8000000000000000;
    return x ^ mask;
}


template <>
  struct lower_32_bits<double>
{
  __host__ __device__ unsigned int operator()(const unsigned long long &x) const {
      return static_cast<unsigned int>(flip_double(x) & 0xffffffff);
  }
};

template <>
  struct upper_32_bits<double>
{
  __host__ __device__ unsigned int operator()(const unsigned long long &x) const {
      return static_cast<unsigned int>(flip_double(x) >> 32);
  }
};


} // end namespace detail
} // end namespace cuda
} // end namespace device
} // end namespace detail
} // end namespace thrust

