/******************************************************************************
 * 
 * Copyright 2010 Duane Merrill
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License. 
 * 
 * 
 * 
 * 
 * AUTHORS' REQUEST: 
 * 
 * 		If you use|reference|benchmark this code, please cite our Technical 
 * 		Report (http://www.cs.virginia.edu/~dgm4d/papers/RadixSortTR.pdf):
 * 
 *		@TechReport{ Merrill:Sorting:2010,
 *        	author = "Duane Merrill and Andrew Grimshaw",
 *        	title = "Revisiting Sorting for GPGPU Stream Architectures",
 *        	year = "2010",
 *        	institution = "University of Virginia, Department of Computer Science",
 *        	address = "Charlottesville, VA, USA",
 *        	number = "CS2010-03"
 *		}
 * 
 * For more information, see our Google Code project site: 
 * http://code.google.com/p/back40computing/
 * 
 * Thanks!
 * 
 ******************************************************************************/


/******************************************************************************
 * Functors for converting signed and floating point types to unsigned types
 * suitable for radix sorting  
 ******************************************************************************/

#pragma once

namespace thrust  {
namespace system  {
namespace cuda    {
namespace detail  {
namespace detail  {
namespace b40c_thrust   {


//
// Do-nothing functors
//

template <typename T>
struct NopFunctor{
    template <typename ConvertedKeyType>
	__device__ __host__ __forceinline__ void operator()(ConvertedKeyType &converted_key) {}
	__device__ __host__ __forceinline__ static bool MustApply(){ return false;}
};

//
// Do-nothing functors that indicate a mandatory pass
//

template <typename T>
struct MandatoryPassNopFunctor{
    template <typename ConvertedKeyType>
	__device__ __host__ __forceinline__ void operator()(ConvertedKeyType &converted_key) {}
	__device__ __host__ __forceinline__ static bool MustApply(){ return false;}
};


//
// Conversion for generic unsigned types
//

template <typename T> struct KeyConversion {
	typedef T UnsignedBits;
};

template <typename T>
struct PreprocessKeyFunctor{
    template <typename ConvertedKeyType>
	__device__ __host__ __forceinline__ void operator()(ConvertedKeyType &converted_key) {}
	__device__ __host__ __forceinline__ static bool MustApply(){ return false;}
};

template <typename T>
struct PostprocessKeyFunctor {
    template <typename ConvertedKeyType>
	__device__ __host__ __forceinline__ void operator()(ConvertedKeyType &converted_key) {}
	__device__ __host__ __forceinline__ static bool MustApply(){ return false;}
};



//
// Conversion for floats
//

template <> struct KeyConversion<float> {
	typedef unsigned int UnsignedBits;
};

template <>
struct PreprocessKeyFunctor<float> {
	__device__ __host__ __forceinline__ void operator()(unsigned int &converted_key) {

		unsigned int mask = (converted_key & 0x80000000) ? 0xffffffff : 0x80000000; 
		converted_key ^= mask;
	}
	__device__ __host__ __forceinline__ static bool MustApply(){ return true;}
};

template <>
struct PostprocessKeyFunctor<float> {
	__device__ __host__ __forceinline__ void operator()(unsigned int &converted_key) {

		unsigned int mask = (converted_key & 0x80000000) ? 0x80000000 : 0xffffffff; 
		converted_key ^= mask;
    }
	__device__ __host__ __forceinline__ static bool MustApply(){ return true;}
};



//
// Conversion for doubles
//

template <> struct KeyConversion<double> {
	typedef unsigned long long UnsignedBits;
};

template <>
struct PreprocessKeyFunctor<double> {
	__device__ __host__ __forceinline__ void operator()(unsigned long long &converted_key) {

		unsigned long long mask = (converted_key & 0x8000000000000000) ? 0xffffffffffffffff : 0x8000000000000000; 
		converted_key ^= mask;
	}
	__device__ __host__ __forceinline__ static bool MustApply(){ return true;}
};

template <>
struct PostprocessKeyFunctor<double> {
	__device__ __host__ __forceinline__ void operator()(unsigned long long &converted_key)  {
		unsigned long long mask = (converted_key & 0x8000000000000000) ? 0x8000000000000000 : 0xffffffffffffffff; 
        converted_key ^= mask;
    }
	__device__ __host__ __forceinline__ static bool MustApply(){ return true;}
};


//
// Conversion for signed chars
//

template <> struct KeyConversion<char> {
  typedef unsigned char UnsignedBits;
};

template <>
struct PreprocessKeyFunctor<char> {
  __device__ __host__ __forceinline__ void operator()(unsigned char &converted_key) {
    // char is unsigned on some platforms, so we have to check
    if(std::numeric_limits<char>::is_signed)
    {
      const unsigned int SIGN_MASK = 1u << ((sizeof(char) * 8) - 1);
      converted_key ^= SIGN_MASK;	
    }
  }
  __device__ __host__ __forceinline__ static bool MustApply(){ return std::numeric_limits<char>::is_signed;}
};

template <>
struct PostprocessKeyFunctor<char> {
  __device__ __host__ __forceinline__ void operator()(unsigned char &converted_key)  {
    // char is unsigned on some platforms, so we have to check
    if(std::numeric_limits<char>::is_signed)
    {
      const unsigned int SIGN_MASK = 1u << ((sizeof(char) * 8) - 1);
      converted_key ^= SIGN_MASK;	
    }
  }
  __device__ __host__ __forceinline__ static bool MustApply(){ return std::numeric_limits<char>::is_signed;}
};


// TODO handle this more gracefully
template <> struct KeyConversion<signed char> {
	typedef unsigned char UnsignedBits;
};

template <>
struct PreprocessKeyFunctor<signed char> {
	__device__ __host__ __forceinline__ void operator()(unsigned char &converted_key) {
		const unsigned int SIGN_MASK = 1u << ((sizeof(char) * 8) - 1);
		converted_key ^= SIGN_MASK;	
	}
	__device__ __host__ __forceinline__ static bool MustApply(){ return true;}
};

template <>
struct PostprocessKeyFunctor<signed char> {
	__device__ __host__ __forceinline__ void operator()(unsigned char &converted_key)  {
		const unsigned int SIGN_MASK = 1u << ((sizeof(char) * 8) - 1);
		converted_key ^= SIGN_MASK;	
    }
	__device__ __host__ __forceinline__ static bool MustApply(){ return true;}
};


//
// Conversion for signed shorts
//

template <> struct KeyConversion<short> {
	typedef unsigned short UnsignedBits;
};

template <>
struct PreprocessKeyFunctor<short> {
	__device__ __host__ __forceinline__ void operator()(unsigned short &converted_key) {
		const unsigned int SIGN_MASK = 1u << ((sizeof(short) * 8) - 1);
		converted_key ^= SIGN_MASK;	
	}
	__device__ __host__ __forceinline__ static bool MustApply(){ return true;}
};

template <>
struct PostprocessKeyFunctor<short> {
	__device__ __host__ __forceinline__ void operator()(unsigned short &converted_key)  {
		const unsigned int SIGN_MASK = 1u << ((sizeof(short) * 8) - 1);
		converted_key ^= SIGN_MASK;	
    }
	__device__ __host__ __forceinline__ static bool MustApply(){ return true;}
};



//
// Conversion for signed ints
//

template <> struct KeyConversion<int> {
	typedef unsigned int UnsignedBits;
};

template <>
struct PreprocessKeyFunctor<int> {
	__device__ __host__ __forceinline__ void operator()(unsigned int &converted_key) {
		const unsigned int SIGN_MASK = 1u << ((sizeof(int) * 8) - 1);
		converted_key ^= SIGN_MASK;	
	}
	__device__ __host__ __forceinline__ static bool MustApply(){ return true;}
};

template <>
struct PostprocessKeyFunctor<int> {
	__device__ __host__ __forceinline__ void operator()(unsigned int &converted_key)  {
		const unsigned int SIGN_MASK = 1u << ((sizeof(int) * 8) - 1);
		converted_key ^= SIGN_MASK;	
    }
	__device__ __host__ __forceinline__ static bool MustApply(){ return true;}
};



//
// Conversion for signed longs
//

// TODO rework this with metaprogramming
template <> struct KeyConversion<unsigned long> {
#if ULONG_MAX == UINT_MAX
    typedef unsigned int UnsignedBits;
#else
    typedef unsigned long long UnsignedBits;
#endif
};

// TODO rework this with metaprogramming
template <> struct KeyConversion<long> {
#if ULONG_MAX == UINT_MAX
    typedef unsigned int UnsignedBits;
#else
    typedef unsigned long long UnsignedBits;
#endif
};

template <>
struct PreprocessKeyFunctor<long> {
	__device__ __host__ __forceinline__ void operator()(typename KeyConversion<long>::UnsignedBits& converted_key) {
		const typename KeyConversion<long>::UnsignedBits SIGN_MASK = 1ul << ((sizeof(long) * 8) - 1);
		converted_key ^= SIGN_MASK;	
	}
	__device__ __host__ __forceinline__ static bool MustApply(){ return true;}
};

template <>
struct PostprocessKeyFunctor<long> {
	__device__ __host__ __forceinline__ void operator()(typename KeyConversion<long>::UnsignedBits& converted_key) {
		const typename KeyConversion<long>::UnsignedBits SIGN_MASK = 1ul << ((sizeof(long) * 8) - 1);
		converted_key ^= SIGN_MASK;	
    }
	__device__ __host__ __forceinline__ static bool MustApply(){ return true;}
};



//
// Conversion for signed long longs 
//

template <> struct KeyConversion<long long> {
	typedef unsigned long long UnsignedBits;
};

template <>
struct PreprocessKeyFunctor<long long> {
	__device__ __host__ __forceinline__ void operator()(unsigned long long &converted_key) {
		const unsigned long long SIGN_MASK = 1ull << ((sizeof(long long) * 8) - 1);
		converted_key ^= SIGN_MASK;	
	}
	__device__ __host__ __forceinline__ static bool MustApply(){ return true;}
};

template <>
struct PostprocessKeyFunctor<long long> {
	__device__ __host__ __forceinline__ void operator()(unsigned long long &converted_key)  {
		const unsigned long long SIGN_MASK = 1ull << ((sizeof(long long) * 8) - 1);
		converted_key ^= SIGN_MASK;	
    }
	__device__ __host__ __forceinline__ static bool MustApply(){ return true;}
};


} // end namespace b40c_thrust
} // end namespace detail
} // end namespace detail
} // end namespace cuda
} // end namespace system
} // end namespace thrust

