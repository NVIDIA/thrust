// Copyright (c) 2014 M.A. (Thijs) van den Berg
// Distributed under the Boost Software License, Version 1.0. (See
// accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)

/*
 *  Copyright 2008-2019 NVIDIA Corporation
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

// This code implements the Threefry-4x64 counter-based pseudorandom number generators described in http://www.thesalmons.org/john/random123/papers/random123sc11.pdf

// This code is adapted from https://github.com/sitmo/threefry
// Its dependency on Boost has been eliminated, some non-standard functionality has been removed, it has been adapted for CUDA C++, and it has been updated for C++ >= 2011.

/*! \file threefry4x64_engine.h
 *  \brief A pseudorandom number engine based on cryptographic hashes.
 */

#pragma once

// XXX this code needs >= C++11
//     what is our preferred way to check for that?

#include <thrust/detail/config.h>
#include <cstdint>
#include <iostream>


// Some of this code causes nvcc to complain based on the template parameters of threefry4x64_engine
// The code in question is benign
// To suppress these warnings, uncomment the #pragmas below.
// Be aware that it's not possible to reenable these warnings once suppressed.
// XXX Eliminate this comment once these nvbugs are fixed
//#if defined(__NVCC__)
//#pragma diag_suppress = subscript_out_of_range // nvbug 2512048
//#pragma diag_suppress = code_is_unreachable    // nvbug 1902675
//#pragma diag_suppress = initialization_not_reachable // nvbug ???
//#endif // __NVCC__


namespace thrust
{
namespace random
{
namespace detail
{


constexpr static std::uint_least64_t threefry4x64_tweak = 0x1BD11BDAA9FC1A22;


// primary template
template<class UIntType, std::size_t w>
struct extract4x64_impl;


// specialization
template<class UIntType>
struct extract4x64_impl<UIntType,64>
{
  __host__ __device__
  constexpr static UIntType zth(const std::uint_least64_t (&output)[4])
  {
    return output[0];
  }

  __host__ __device__
  inline static UIntType nth(const std::uint_least64_t (&output)[4], std::size_t n)
  {
    return output[n];
  }

  __host__ __device__
  constexpr static UIntType w_max()
  {
    return 0xFFFFFFFFFFFFFFFF;
  }
};


template<class UIntType>
struct extract4x64_impl<UIntType,32>
{
  __host__ __device__
  constexpr static UIntType zth(const std::uint_least64_t (&output)[4])
  {
    return output[0] & 0xFFFFFFFF;
  }

  __host__ __device__
  inline static UIntType nth(const std::uint_least64_t (&output)[4], std::size_t n)
  {
    return (output[n>>1] >> ((n&1)<<5)) & 0xFFFFFFFF;
  }

  __host__ __device__
  constexpr static UIntType w_max()
  {
    return 0xFFFFFFFF;
  }
};


template<class UIntType>
struct extract4x64_impl<UIntType,16>
{
  __host__ __device__
  constexpr static UIntType zth(const std::uint_least64_t (&output)[4])
  {
    return output[0] & 0xFFFF;
  }

  __host__ __device__
  inline static UIntType nth(const std::uint_least64_t (&output)[4], std::size_t n)
  {
    return (output[n>>2] >> ((n&3)<<4)) & 0xFFFF;
  }

  __host__ __device__
  constexpr static UIntType w_max()
  {
    return 0xFFFF;
  }
};


template<class UIntType>
struct extract4x64_impl<UIntType,8>
{
  __host__ __device__
  constexpr static UIntType zth(const std::uint_least64_t (&output)[4])
  {
    return output[0] & 0xFF;
  }

  __host__ __device__
  inline static UIntType nth(const std::uint_least64_t (&output)[4], std::size_t n)
  {
    return (output[n>>3] >> ((n&7)<<3)) & 0xFF;
  }

  __host__ __device__
  constexpr static UIntType w_max()
  {
    return 0xFF;
  }
};


} // detail


/*! \addtogroup random_number_engine_templates
 *  \{
 */

/*! \class threefry4x64_engine
 *  \brief A \p threefry4x64_engine pseudorandom number generator produces unsigned integer
 *         pseudorandom numbers using a cryptographic hash generation algorithm based on the
 *         Threefish encryption function. The algorithm is described in
 *         "Parallel Random Numbers: As Easy as 1, 2, 3" by Salmon et al., 2011.
 *
 *  \tparam UIntType The type of unsigned integer to produce.
 *  \tparam ReturnBits The number of bits of randomness in the result. Must be in {8, 16, 32, 64}.
 *  \tparam Rounds The number of rounds to use during generation.
 *  \tparam KeySize The size of key to use. Must be in [0, 4].
 *  \tparam CounterSize The size of counter to use. Must be in [1, 4].
 *
 *  \note Refer to Salmon et al., 2011 for detailed descriptions of these algorithm parameters.
 *
 *  \note Inexperienced users should not use this class template directly. Instead, use
 *  \p threefry4x64_13, \p threefry4x64_13_64, \p threefry4x64_20, or \p threefry4x64_20_64.
 *
 *  The following code snippet shows examples of use of a \p threefry4x64_engine instance:
 *
 *  \code
 *  #include <thrust/random/threefry4x64_engine.h>
 *  #include <iostream>
 *
 *  int main(void)
 *  {
 *    // create a threefry4x64_13 object, which is an instance of threefry4x64_engine
 *    thrust::threefry4x64_13 rng1;
 *
 *    // output some random values to cout
 *    std::cout << rng1() << std::endl;
 *
 *    // a random value is printed
 *
 *    // create a new minstd_rand from a seed
 *    thrust::threefry4x64_13 rng2(13);
 *
 *    // discard some random values
 *    rng2.discard(13);
 *
 *    // stream the object to an iostream
 *    std::cout << rng2 << std::endl;
 *
 *    // rng2's current state is printed
 *
 *    // print the minimum and maximum values that threefry4x64_13 can produce
 *    std::cout << thrust::threefry4x64_13.min() << std::endl;
 *    std::cout << thrust::threefry4x64_13.max() << std::endl;
 *
 *    // the range of minstd_rand is printed
 *
 *    // save the state of rng2 to a different object
 *    thrust::threefry4x64_13 rng3 = rng2;
 *
 *    // compare rng2 and rng3
 *    std::cout << (rng2 == rng3) << std::endl;
 *
 *    // 1 is printed
 *
 *    // re-seed rng2 with a different seed
 *    rng2.seed(7);
 *
 *    // compare rng2 and rng3
 *    std::cout << (rng2 == rng3) << std::endl;
 *
 *    // 0 is printed
 *
 *    return 0;
 *  }
 *
 *  \endcode
 *
 *  \see thrust::random::threefry4x64_13
 *  \see thrust::random::threefry4x64_13_64
 *  \see thrust::random::threefry4x64_20
 *  \see thrust::random::threefry4x64_20_64
 */
template<class UIntType,
         std::size_t ReturnBits,
         std::size_t Rounds = 20,
         std::size_t KeySize = 4,
         std::size_t CounterSize = 4
        >
class threefry4x64_engine
{
    /*! \cond
     */
  private:
    static constexpr std::size_t samples_per_block = 256/ReturnBits;
    /*! \endcond
     */

  public:
    static_assert(ReturnBits==8 or ReturnBits==16 or ReturnBits==32 or ReturnBits==64, "ReturnBits must be in {8, 16, 32, 64}.");
    static_assert(KeySize>=0 and KeySize<=4, "KeySize must be in [0, 4].");
    static_assert(CounterSize>=1 and CounterSize<=4, "CounterSize must be in [1, 4]");
 
    /*! \typedef result_type
     *  \brief The type of the unsigned integer produced by this \p threefry4x64_engine.
     */
    using result_type = UIntType;

    /*! The default seed of this \p threefry4x64_engine.
     */
    static constexpr result_type default_seed = 0;
    
    /*! Constructs a new \p threefry4x64_engine with the given seed value.
     *  \param value The seed used to initialize this \p threefry4x64_engine's state.
     */
    __host__ __device__
    explicit threefry4x64_engine(result_type value)
    {
      seed(value);
    }

    /*! Constructs a new \p threefry4x64_engine with the \p default_seed.
     */
    __host__ __device__
    threefry4x64_engine()
      : threefry4x64_engine(default_seed)
    {}

    /*! Initializes this \p threefry4x64_engine's state, and optionally accepts a seed value.
     *
     *  \param value The seed used to initialize this \p threefry4x64_engine's state.
     */
    __host__ __device__
    void seed(result_type value = default_seed)
    {
      if(KeySize>=1) key_[1] = value; // UIntType is max 64 bits, key[0] has the xor-tweak
      if(KeySize>=2) key_[2] = 0;
      if(KeySize>=3) key_[3] = 0;
      if(KeySize>=4) key_[4] = 0;
      reset_after_key_change();
    }
    
    // XXX this needs to be a constexpr function, but Thrust's existing
    //     random distributions check for a static member variable
    //     how would we like to address this mismatch?
    // Returns the smallest possible value in the output range.
    //__host__ __device__
    //constexpr static result_type min()
    //{
    //  return 0;
    //}
    constexpr static result_type min = 0;

    // XXX this needs to be a constexpr function, but Thrust's existing
    //     random distributions check for a static member variable
    //     how would we like to address this mismatch?
    // Returns the largest possible value in the output range.
    //__host__ __device__
    //constexpr static result_type max()
    //{
    //  return detail::extract4x64_impl<UIntType,ReturnBits>::w_max();
    //}
    constexpr static result_type max = detail::extract4x64_impl<UIntType,ReturnBits>::w_max();

    /*! Produces a new random value and updates this \p threefry4x64_engine's state.
     *  \return A new random number.
     */
    __host__ __device__
    result_type operator()()
    {
      // can we return a value from the current block?
      if(o_counter_ < samples_per_block)
      {
        return detail::extract4x64_impl<UIntType,ReturnBits>::nth(output_, o_counter_++);
      }
      
      // generate a new block and return the first result_type 
      increment_counter();
      encrypt_counter();
      o_counter_ = 1; // the next call
      return detail::extract4x64_impl<UIntType,ReturnBits>::zth(output_);
    }

    /*! Advances this \p threefry4x64_engine's state a given number of times
     *  and discards the results.
     *
     *  \param z The number of random values to discard.
     *  \note This function is provided because an implementation may be able to accelerate it.
     */
    __host__ __device__
    void discard(unsigned long long z)
    {
      // check if we stay in the current block
      if(z <= samples_per_block - o_counter_)
      {
        o_counter_ += static_cast<unsigned short>(z);
        return;
      }

      o_counter_ += (z % samples_per_block);
      z /= samples_per_block;
       
      if(o_counter_ > samples_per_block)
      {
        o_counter_ -= samples_per_block;
        ++z;
      }
       
      increment_counter(z);
      
      if(o_counter_ != samples_per_block)
      {
        encrypt_counter();
      }
    }

    /*! Streams a \p threefry4x64_engine to a \p std::basic_ostream.
     *  \param os The \p basic_ostream to stream out to.
     *  \param eng The \p threefry4x64_engine to stream out.
     *  \return \p os
     */
    template<class CharT, class Traits>
    friend std::basic_ostream<CharT, Traits>& operator<<(std::basic_ostream<CharT, Traits>& os, const threefry4x64_engine& eng)
    {
      for(int i=0; i < int(KeySize); ++i)
      {
        os << eng.key_[i+1] << ' ';
      }
          
      for(int i=0; i < int(CounterSize); ++i)
      {
        os << eng.counter_[i] << ' ';
      }
          
      os << eng.o_counter_;

      return os;
    }
   
    /*! Streams a \p threefry4x64_engine in from a \p std::basic_istream.
     *  \param is The \p basic_istream to stream from.
     *  \param eng The \p threefry4x64_engine to stream in.
     *  \return \p is
     */
    template<class CharT, class Traits>
    friend std::basic_istream<CharT, Traits>& operator>>(std::basic_istream<CharT, Traits>& is, threefry4x64_engine& eng)
    {
      for(int i=0; i < int(KeySize); ++i) 
      {
        is >> eng.key_[i+1] >> std::ws;
      }
          
      for(int i=0; i < int(CounterSize); ++i) 
      {
        is >> eng.counter_[i] >> std::ws;
      }
          
      is >> eng.o_counter_;
      eng.initialize_key();
      
      eng.encrypt_counter();

      return is;
    } 

    /*! Compares two \p threefry4x64_engines for equality.
     *  \param lhs The first \p threefry4x64_engines to compare.
     *  \param rhs The second \p threefry4x64_engines to compare.
     *  \return \c true if \p lhs is equal to \p rhs; \c false, otherwise.
     */
    __host__ __device__
    friend bool operator==(const threefry4x64_engine& lhs, const threefry4x64_engine& rhs) 
    {
      if(lhs.o_counter_ != rhs.o_counter_) return false;
      
      for(unsigned short i=0; i<KeySize; ++i)
      {
        if(lhs.key_[i] != rhs.key_[i]) return false;
      }

      for(unsigned short i=0; i<CounterSize; ++i)
      {
        if(lhs.counter_[i] != rhs.counter_[i]) return false;
      }

      return true;
    }
    
    /*! Compares two \p threefry4x64_engines for inequality.
     *  \param lhs The first \p threefry4x64_engines to compare.
     *  \param rhs The second \p threefry4x64_engines to compare.
     *  \return \c true if \p lhs is not equal to \p rhs; \c false, otherwise.
     */
    __host__ __device__
    friend bool operator!=(const threefry4x64_engine& lhs, const threefry4x64_engine& rhs) 
    { 
      return !(lhs == rhs);
    }


    /*! \cond
     */
  private:
    __host__ __device__
    inline static void rotl64(std::uint_least64_t& v, std::uint8_t bits)
    { 
      v = (v << bits) | (v >> (64 - bits));
    }
    
    __host__ __device__
    inline static void mix64(std::uint_least64_t& x0, std::uint64_t& x1, const std::uint8_t bits)
    {
      x0 += x1;
      rotl64(x1, bits);
      x1 ^= x0;
    }

    __host__ __device__
    inline static void double_mix64(std::uint_least64_t& x0, std::uint_least64_t& x1, const std::uint8_t rx,
                                    std::uint_least64_t& z0, std::uint_least64_t& z1, const std::uint8_t rz)
    {
      mix64(x0,x1,rx);
      mix64(z0,z1,rz);
    }

    template<std::size_t offset>
    __host__ __device__
    inline static void add_key64_t(std::uint_least64_t (&output)[4], std::uint_least64_t (&key)[KeySize+1], const std::size_t c)
    {
      if(((offset+1)%5) <= KeySize) output[0] += key[(offset+1)%5];
      if(((offset+2)%5) <= KeySize) output[1] += key[(offset+2)%5];
      if(((offset+3)%5) <= KeySize) output[2] += key[(offset+3)%5];
      if(((offset+4)%5) <= KeySize) output[3] += key[(offset+4)%5];
      output[3] += c;
    }

    template<std::size_t R>
    __host__ __device__
    inline void encrypt_counter_t(std::size_t& four_cycles)
    {
      double_mix64(output_[0], output_[1], 14, output_[2], output_[3], 16); if (R<2) return;
      double_mix64(output_[0], output_[3], 52, output_[2], output_[1], 57); if (R<3) return;
      double_mix64(output_[0], output_[1], 23, output_[2], output_[3], 40); if (R<4) return;
      double_mix64(output_[0], output_[3],  5, output_[2], output_[1], 37);
      add_key64_t<1>(output_, key_, ++four_cycles); if (R<5) return;
      
      double_mix64(output_[0], output_[1], 25, output_[2], output_[3], 33); if (R<6) return;
      double_mix64(output_[0], output_[3], 46, output_[2], output_[1], 12); if (R<7) return;
      double_mix64(output_[0], output_[1], 58, output_[2], output_[3], 22); if (R<8) return;
      double_mix64(output_[0], output_[3], 32, output_[2], output_[1], 32);
      add_key64_t<2>(output_, key_, ++four_cycles); if (R<9) return;
      
      double_mix64(output_[0], output_[1], 14, output_[2], output_[3], 16); if (R<10) return;
      double_mix64(output_[0], output_[3], 52, output_[2], output_[1], 57); if (R<11) return;
      double_mix64(output_[0], output_[1], 23, output_[2], output_[3], 40); if (R<12) return;
      double_mix64(output_[0], output_[3],  5, output_[2], output_[1], 37);
      add_key64_t<3>(output_, key_, ++four_cycles); if (R<13) return;

      double_mix64(output_[0], output_[1], 25, output_[2], output_[3], 33); if (R<14) return;
      double_mix64(output_[0], output_[3], 46, output_[2], output_[1], 12); if (R<15) return;
      double_mix64(output_[0], output_[1], 58, output_[2], output_[3], 22); if (R<16) return;
      double_mix64(output_[0], output_[3], 32, output_[2], output_[1], 32); 
      add_key64_t<4>(output_, key_, ++four_cycles); if (R<17) return;
      
      double_mix64(output_[0], output_[1], 14, output_[2], output_[3], 16); if (R<18) return;
      double_mix64(output_[0], output_[3], 52, output_[2], output_[1], 57); if (R<19) return;
      double_mix64(output_[0], output_[1], 23, output_[2], output_[3], 40); if (R<20) return;
      double_mix64(output_[0], output_[3],  5, output_[2], output_[1], 37);
      add_key64_t<0>(output_, key_, ++four_cycles); if (R<21) return;

      double_mix64(output_[0], output_[1], 25, output_[2], output_[3], 33); if (R<22) return;
      double_mix64(output_[0], output_[3], 46, output_[2], output_[1], 12); if (R<23) return;
      double_mix64(output_[0], output_[1], 58, output_[2], output_[3], 22); if (R<24) return;
      double_mix64(output_[0], output_[3], 32, output_[2], output_[1], 32);
      add_key64_t<1>(output_, key_, ++four_cycles); if (R<25) return;

      double_mix64(output_[0], output_[1], 14, output_[2], output_[3], 16); if (R<26) return;
      double_mix64(output_[0], output_[3], 52, output_[2], output_[1], 57); if (R<27) return;
      double_mix64(output_[0], output_[1], 23, output_[2], output_[3], 40); if (R<28) return;
      double_mix64(output_[0], output_[3],  5, output_[2], output_[1], 37); 
      add_key64_t<2>(output_, key_, ++four_cycles); if (R<29) return;

      double_mix64(output_[0], output_[1], 25, output_[2], output_[3], 33); if (R<30) return;
      double_mix64(output_[0], output_[3], 46, output_[2], output_[1], 12); if (R<31) return;
      double_mix64(output_[0], output_[1], 58, output_[2], output_[3], 22); if (R<32) return;
      double_mix64(output_[0], output_[3], 32, output_[2], output_[1], 32); 
      add_key64_t<3>(output_, key_, ++four_cycles); if (R<33) return;

      double_mix64(output_[0], output_[1], 14, output_[2], output_[3], 16); if (R<34) return;
      double_mix64(output_[0], output_[3], 52, output_[2], output_[1], 57); if (R<35) return;
      double_mix64(output_[0], output_[1], 23, output_[2], output_[3], 40); if (R<36) return;

      double_mix64(output_[0], output_[3],  5, output_[2], output_[1], 37);
      add_key64_t<4>(output_, key_, ++four_cycles); if (R<37) return;

      double_mix64(output_[0], output_[1], 25, output_[2], output_[3], 33); if (R<38) return;
      double_mix64(output_[0], output_[3], 46, output_[2], output_[1], 12); if (R<39) return;
      double_mix64(output_[0], output_[1], 58, output_[2], output_[3], 22); if (R<40) return;

      double_mix64(output_[0], output_[3], 32, output_[2], output_[1], 32);
      add_key64_t<0>(output_, key_, ++four_cycles);
    }
    
    __host__ __device__
    void encrypt_counter()
    {
      for(int i = 0; i < int(CounterSize); ++i)
      {
        output_[i] = counter_[i];
      }
      
      for(int i = CounterSize; i<4; ++i)
      {
        output_[i] = 0;
      }
      
      for(int i = 0; i < int(KeySize); ++i)
      {
        output_[i] += key_[(i+1)%5];
      }
      
      std::size_t four_cycles = 0;

      // do chunks of 40 rounds
      for(int big_rounds = 0; big_rounds < int(Rounds/40); ++big_rounds)
      {
        encrypt_counter_t<40>(four_cycles);
      }
      
      // the remaining rounds
      encrypt_counter_t<Rounds - 40*(Rounds/40)>(four_cycles);
    }
    
    // increment the counter by 1
    __host__ __device__
    void increment_counter()
    {
      ++counter_[0]; 
      
      if(CounterSize > 1)
      {
        if(counter_[0] != 0) return; // test for overflow, exit when not

        ++counter_[1];

        if(CounterSize > 2)
        {
          if(counter_[1] != 0) return;

          ++counter_[2];

          if(CounterSize > 3)
          {
            if(counter_[2] != 0) return;

            ++counter_[3];
          }
        }
      }
    }
    
    // increment the counter by z
    __host__ __device__
    void increment_counter(std::uintmax_t z)
    {
      if(CounterSize == 1)
      {
        counter_[0] += z;
        return;
      }
      
      bool overflow = (z > 0xFFFFFFFFFFFFFFFF - counter_[0]);
      counter_[0] += z;

      if(!overflow) return;
      
      ++counter_[1];
      
      if(CounterSize <= 2) return;

      if(counter_[1]!=0) return;

      ++counter_[2];
          
      if(CounterSize <= 3) return;

      if(counter_[2]!=0) return;

      ++counter_[3];
    }

    __host__ __device__
    void initialize_key()
    {
      key_[0] = detail::threefry4x64_tweak;
      if(KeySize>0) key_[0] ^= key_[1];
      if(KeySize>1) key_[0] ^= key_[2];
      if(KeySize>2) key_[0] ^= key_[3];
      if(KeySize>3) key_[0] ^= key_[4];
    }
    
    __host__ __device__
    void reset_counter()
    {
      counter_[0] = 0xFFFFFFFFFFFFFFFF;
      if(CounterSize>=2) counter_[1] = 0xFFFFFFFFFFFFFFFF;
      if(CounterSize>=3) counter_[2] = 0xFFFFFFFFFFFFFFFF;
      if(CounterSize>=4) counter_[3] = 0xFFFFFFFFFFFFFFFF;
      o_counter_ = samples_per_block;
    }

    // reset the counter to zero, and reset the key
    __host__ __device__
    void reset_after_key_change()
    {
      initialize_key();
      reset_counter();
    }
    
    std::uint_least64_t counter_[CounterSize];    // the 256 bit counter (message) that gets encrypted
    std::uint_least64_t output_[4];               // the 256 bit cipher output 4 * 64 bit = 256 bit output
    std::uint_least64_t key_[KeySize+1];          // the 256 bit encryption key
    std::uint_least16_t o_counter_;               // output chunk counter, e.g. for a 64 bit random engine
                                                  // the 256 bit output buffer gets split in 4x64bit chunks or 8x32bit chunks chunks.

    /*! \endcond
     */
};


/*! \} // end random_number_engine_templates
 */

/*! \addtogroup predefined_random
 *  \{
 */

/*! \typedef threefry4x64_13
 *  \brief A random number engine with predefined parameters which implements a 
 *         version of the Threefry-4x64 random number generation algorithm.
 *         This engine produces 32b numbers with a 2^67 cycle length from 13 rounds of the threefry4x64_engine using a 64b seed.
 *  \note The 10000th consecutive invocation of a default-constructed object of type \p threefry4x64_13
 *        shall produce the value \c 2373911387 .
 */
using threefry4x64_13 = threefry4x64_engine<std::uint32_t, 32, 13, 1, 1>;

/*! \typedef threefry4x64_13_64
 *  \brief A random number engine with predefined parameters which implements a 
 *         version of the Threefry-4x64 random number generation algorithm.
 *         This engine produces 64b numbers with a 2^66 cycle length from 13 rounds of the threefry4x64_engine using a 64b seed.
 *  \note The 10000th consecutive invocation of a default-constructed object of type \p threefry4x64_13_64
 *        shall produce the value \c 10433005109212818813 .
 */
using threefry4x64_13_64 = threefry4x64_engine<std::uint64_t, 64, 13, 1, 1>;

/*! \typedef threefry4x64_20
 *  \brief A random number engine with predefined parameters which implements a 
 *         version of the Threefry-4x64 random number generation algorithm.
 *         This engine produces 32b numbers with a 2^67 cycle length from 20 rounds of the threefry4x64_engine using a 64b seed.
 *  \note The 10000th consecutive invocation of a default-constructed object of type \p threefry4x64_20
 *        shall produce the value \c 4062489272 .
 */
using threefry4x64_20 = threefry4x64_engine<std::uint32_t, 32, 20, 1, 1>;

/*! \typedef threefry4x64_20_64
 *  \brief A random number engine with predefined parameters which implements a 
 *         version of the Threefry-4x64 random number generation algorithm.
 *         This engine produces 64b numbers with a 2^66 cycle length from 20 rounds of the threefry4x64_engine using a 64b seed.
 *  \note The 10000th consecutive invocation of a default-constructed object of type \p threefry4x64_20_64
 *        shall produce the value \c 3110068036081305158 .
 */
using threefry4x64_20_64 = threefry4x64_engine<std::uint64_t, 64, 20, 1, 1>;

/*! \} // predefined
 */

} // end random

// import names into thrust::
using random::threefry4x64_engine;
using random::threefry4x64_13;
using random::threefry4x64_13_64;
using random::threefry4x64_20;
using random::threefry4x64_20_64;

} // end thrust

