/*
 *  Copyright 2008-20120 NVIDIA Corporation
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

#include <thrust/detail/config.h>
#include <thrust/detail/cpp11_required.h>

#if THRUST_CPP_DIALECT >= 2011

#include <thrust/detail/temporary_array.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/transform_output_iterator.h>
#include <thrust/random.h>
#include <thrust/scan.h>
#include <thrust/system/detail/generic/shuffle.h>

namespace thrust {
namespace system {
namespace detail {
namespace generic {

// An implementation of RC5
template <uint32_t num_rounds = 12> class rc5_bijection {
private:
  struct round_state {
    uint32_t A;
    uint32_t B;
  };

public:
  template <class URBG>
  __host__ __device__ rc5_bijection(uint64_t m, URBG &&g)
   : w(get_cipher_bits(m))
  {
    init_state(std::forward<URBG>(g));
  }

  __host__ __device__ uint64_t bijection_width() const {
    return 1ull << (2*w);
  }

  __host__ __device__ uint64_t operator()(const uint64_t val) const {
    if(w == 0)
      return val;
    round_state state = { (uint32_t)val & get_mask(), (uint32_t)(val >> w) };
    state.A = (state.A + S[0]) & get_mask();
    state.B = (state.B + S[1]) & get_mask();
    for(uint32_t i = 0; i < num_rounds; i++)
      state = do_round(state, i);
    uint64_t res = state.B << w | state.A;
    return res;
  }

private:
  template <class URBG>
  __host__ __device__ void init_state(URBG&& g)
  {
    thrust::uniform_int_distribution<uint32_t> dist(0, get_mask());
    for( uint32_t i = 0; i < state_size; i++ )
      S[i] = dist(g);
  }

  // Find the nearest power of two
  __host__ __device__ uint64_t get_cipher_bits(uint64_t m) {
    if(m == 0)
      return 0;
    uint64_t i = 0;
    m--;
    while (m != 0) {
      i++;
      m >>= 2u;
    }
    return i;
  }

  __host__ __device__ uint32_t get_mask() const
  {
    return (uint32_t)((1ull << (uint64_t)w) - 1ull);
  }

  __host__ __device__ uint32_t rotl( uint32_t val, uint32_t amount ) const
  {
    const uint32_t amount_mod = amount % w;
    return val << amount_mod | val >> (w-amount_mod);
  }

  __host__ __device__ round_state do_round(const round_state state,
                                           const uint64_t round) const {
    uint32_t A = state.A;
    uint32_t B = state.B;

    A = ( rotl( A ^ B, B ) + S[ 2 * round + 2 ] ) & get_mask();
    B = ( rotl( A ^ B, A ) + S[ 2 * round + 3 ] ) & get_mask();

    return { A, B };
  }

  static constexpr uint64_t state_size = 2 * num_rounds + 3;
  const uint32_t w = 0;
  uint32_t S[state_size];
};

struct key_flag_tuple {
  uint64_t key;
  uint64_t flag;
};

// scan only flags
struct key_flag_scan_op {
  __host__ __device__ key_flag_tuple operator()(const key_flag_tuple &a,
                                                const key_flag_tuple &b) {
    return {b.key, a.flag + b.flag};
  }
};

template <class bijection_op> struct construct_key_flag_op {
  uint64_t m;
  bijection_op bijection;
  __host__ __device__ construct_key_flag_op(uint64_t m, bijection_op bijection)
      : m(m), bijection(bijection) {}
  __host__ __device__ key_flag_tuple operator()(uint64_t idx) {
    auto gather_key = bijection(idx);
    return key_flag_tuple{gather_key, (gather_key < m) ? 1ull : 0ull};
  }
};

template <typename InputIterT, typename OutputIterT> struct write_output_op {
  uint64_t m;
  InputIterT in;
  OutputIterT out;
  // flag contains inclusive scan of valid keys
  // perform gather using valid keys
  __thrust_exec_check_disable__ __host__ __device__ size_t
  operator()(key_flag_tuple x) {
    if (x.key < m) {
      // -1 because inclusive scan
      out[x.flag - 1] = in[x.key];
    }
    return 0; // Discarded
  }
};

template <typename ExecutionPolicy, typename RandomIterator, typename URBG>
__host__ __device__ void
shuffle(thrust::execution_policy<ExecutionPolicy> &exec, RandomIterator first,
        RandomIterator last, URBG &&g) {
  typedef
      typename thrust::iterator_traits<RandomIterator>::value_type InputType;

  // copy input to temp buffer
  thrust::detail::temporary_array<InputType, ExecutionPolicy> temp(exec, first,
                                                                   last);
  thrust::shuffle_copy(exec, temp.begin(), temp.end(), first, g);
}

template <typename ExecutionPolicy, typename RandomIterator,
          typename OutputIterator, typename URBG>
__host__ __device__ void
shuffle_copy(thrust::execution_policy<ExecutionPolicy> &exec,
             RandomIterator first, RandomIterator last, OutputIterator result,
             URBG &&g) {
  // m is the length of the input
  // we have an available bijection of length n via a feistel cipher
  size_t m = last - first;
  using bijection_op = rc5_bijection<>;
  bijection_op bijection(m, g);
  uint64_t n = bijection.bijection_width();

  // perform stream compaction over length n bijection to get length m
  // pseudorandom bijection over the original input
  thrust::counting_iterator<uint64_t> indices(0);
  thrust::transform_iterator<construct_key_flag_op<bijection_op>,
                             decltype(indices), key_flag_tuple>
      key_flag_it(indices, construct_key_flag_op<bijection_op>(m, bijection));
  write_output_op<RandomIterator, decltype(result)> write_functor{m, first,
                                                                  result};
  auto gather_output_it = thrust::make_transform_output_iterator(
      thrust::discard_iterator<size_t>(), write_functor);
  // the feistel_bijection outputs a stream of permuted indices in range [0,n)
  // flag each value < m and compact it, so we have a set of permuted indices in
  // range [0,m) each thread gathers an input element according to its
  // pseudorandom permuted index
  thrust::inclusive_scan(exec, key_flag_it, key_flag_it + n, gather_output_it,
                         key_flag_scan_op());
}

} // end namespace generic
} // end namespace detail
} // end namespace system
} // end namespace thrust
#endif
