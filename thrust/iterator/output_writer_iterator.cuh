/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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
 */

// Output writer iterator
#pragma once

#include <thrust/iterator/iterator_adaptor.h>

namespace thrust {
namespace detail {

// Proxy reference that calls BinaryFunction with Iterator value and the rhs of assignment operator
template <typename BinaryFunction, typename Iterator>
class output_writer_iterator_proxy {
 public:
  __host__ __device__ output_writer_iterator_proxy(const Iterator& index_iter, BinaryFunction fun)
    : index_iter(index_iter), fun(fun)
  {
  }
  template <typename T>
  __host__ __device__ output_writer_iterator_proxy operator=(const T& x)
  {
    fun(*index_iter, x);
    return *this;
  }

 private:
  Iterator index_iter;
  BinaryFunction fun;
};

// Register output_writer_iterator_proxy with 'is_proxy_reference' from
// type_traits to enable its use with algorithms.
template <class BinaryFunction, class Iterator>
struct is_proxy_reference<output_writer_iterator_proxy<BinaryFunction, Iterator>>
  : public thrust::detail::true_type {
};

}  // namespace detail
}  // namespace thrust

namespace cudf {
/**
 * @brief Transform output iterator with custom writer binary function which takes index and value.
 *
 * @code {.cpp}
 * #include <cudf/utilities/output_writer_iterator.cuh>
 * #include <thrust/device_vector.h>
 * #include <thrust/iterator/counting_iterator.h>
 * #include <thrust/iterator/transform_iterator.h>
 *
 * struct set_bits_field {
 *   int* bitfield;
 *   __device__ inline void set_bit(size_t bit_index)
 *   {
 *     atomicOr(&bitfield[bit_index/32], (int{1} << (bit_index % 32)));
 *   }
 *   __device__ inline void clear_bit(size_t bit_index)
 *   {
 *     atomicAnd(&bitfield[bit_index / 32], ~(int{1} << (bit_index % 32)));
 *   }
 *   // Index, value
 *   __device__ void operator()(size_t i, bool x)
 *   {
 *     if (x)
 *       set_bit(i);
 *     else
 *       clear_bit(i);
 *   }
 * };
 *
 * thrust::device_vector<int> v(1, 0x0000);
 * auto result_begin = cudf::make_output_writer_iterator(thrust::make_counting_iterator(0),
 *                                                 set_bits_field{v.data().get()});
 * auto value = thrust::make_transform_iterator(thrust::make_counting_iterator(0), [] __device__
 * (int x) { return x%2;
 * });
 * thrust::copy(thrust::device, value, value+32, result_begin);
 *
 * #include <cudf/utilities/output_writer_iterator.cuh>
 * #include <thrust/device_vector.h>
 * #include <thrust/iterator/counting_iterator.h>
 * #include <thrust/iterator/transform_iterator.h>
 *
 * struct set_bits_field {
 *   int* bitfield;
 *   __device__ inline void set_bit(size_t bit_index)
 *   {
 *     atomicOr(&bitfield[bit_index/32], (int{1} << (bit_index % 32)));
 *   }
 *   __device__ inline void clear_bit(size_t bit_index)
 *   {
 *     atomicAnd(&bitfield[bit_index / 32], ~(int{1} << (bit_index % 32)));
 *   }
 *   // Index, value
 *   __device__ void operator()(size_t i, bool x)
 *   {
 *     if (x)
 *       set_bit(i);
 *     else
 *       clear_bit(i);
 *   }
 * };
 *
 * thrust::device_vector<int> v(1, 0x0000);
 * auto result_begin = cudf::make_output_writer_iterator(thrust::make_counting_iterator(0),
 *                                                 set_bits_field{v.data().get()});
 * auto value = thrust::make_transform_iterator(thrust::make_counting_iterator(0),
 *   [] __device__ (int x) {   return x%2; });
 * thrust::copy(thrust::device, value, value+32, result_begin);
 * int(v[0]); // returns 0xaaaaaaaa;
 * @endcode
 *
 *
 * @tparam BinaryFunction Binary function to be called with the Iterator value and the rhs of
 * assignment operator.
 * @tparam Iterator iterator type that acts as index of the output.
 */
template <typename BinaryFunction, typename Iterator>
class output_writer_iterator
  : public thrust::iterator_adaptor<
      output_writer_iterator<BinaryFunction, Iterator>,
      Iterator,
      thrust::use_default,
      thrust::use_default,
      thrust::use_default,
      thrust::detail::output_writer_iterator_proxy<BinaryFunction, Iterator>> {
 public:
  // parent class.
  typedef thrust::iterator_adaptor<
    output_writer_iterator<BinaryFunction, Iterator>,
    Iterator,
    thrust::use_default,
    thrust::use_default,
    thrust::use_default,
    thrust::detail::output_writer_iterator_proxy<BinaryFunction, Iterator>>
    super_t;
  // friend thrust::iterator_core_access to allow it access to the private interface dereference()
  friend class thrust::iterator_core_access;
  __host__ __device__ output_writer_iterator(Iterator const& x, BinaryFunction fun)
    : super_t(x), fun(fun)
  {
  }

 private:
  BinaryFunction fun;

  // thrust::iterator_core_access accesses this function
  __host__ __device__ typename super_t::reference dereference() const
  {
    return thrust::detail::output_writer_iterator_proxy<BinaryFunction, Iterator>(
      this->base_reference(), fun);
  }
};

template <typename BinaryFunction, typename Iterator>
output_writer_iterator<BinaryFunction, Iterator> __host__ __device__
make_output_writer_iterator(Iterator out, BinaryFunction fun)
{
  return output_writer_iterator<BinaryFunction, Iterator>(out, fun);
}  // end make_output_writer_iterator
}  // namespace cudf
