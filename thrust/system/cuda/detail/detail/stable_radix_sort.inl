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

#include <thrust/detail/config.h>

// do not attempt to compile this file with any other compiler
#if THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_NVCC

#include <thrust/detail/copy.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/detail/temporary_array.h>
#include <thrust/detail/raw_pointer_cast.h>
#include <thrust/system/cuda/detail/execute_on_stream.h>
#include <thrust/system/cuda/detail/bulk.h>
#include <thrust/system/cuda/detail/cub.h>
#include <thrust/detail/util/blocking.h>
#include <thrust/tuple.h>

namespace thrust
{
namespace system
{
namespace cuda
{
namespace detail
{
namespace detail
{
namespace stable_radix_sort_detail
{


// sort ascending
template<typename Key>
__host__ __device__
cudaError_t cub_sort_keys_wrapper(void *d_temp_storage,
                                  size_t &temp_storage_bytes,
                                  cub_::DoubleBuffer<Key> &d_keys,
                                  int num_items,
                                  thrust::less<Key> comp,
                                  int begin_bit = 0,
                                  int end_bit = sizeof(Key) * 8,
                                  cudaStream_t stream = 0,
                                  bool debug_synchronous = false)
{
  struct workaround
  {
    __host__ 
    static cudaError_t host_path(void *d_temp_storage,
                                 size_t &temp_storage_bytes,
                                 cub_::DoubleBuffer<Key> &d_keys,
                                 int num_items,
                                 thrust::less<Key>,
                                 int begin_bit,
                                 int end_bit,
                                 cudaStream_t stream,
                                 bool debug_synchronous)
    {
      return cub_::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_keys, num_items, begin_bit, end_bit, stream, debug_synchronous);
    }

    __device__
    static cudaError_t device_path(void *d_temp_storage,
                                   size_t &temp_storage_bytes,
                                   cub_::DoubleBuffer<Key> &d_keys,
                                   int num_items,
                                   thrust::less<Key>,
                                   int begin_bit,
                                   int end_bit,
                                   cudaStream_t stream,
                                   bool debug_synchronous)
    {
#if __BULK_HAS_CUDART__
      return cub_::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_keys, num_items, begin_bit, end_bit, stream, debug_synchronous);
#else
      return cudaErrorNotSupported;
#endif
    }
  };

#ifndef __CUDA_ARCH__
  return workaround::host_path(d_temp_storage, temp_storage_bytes, d_keys, num_items, comp, begin_bit, end_bit, stream, debug_synchronous);
#else
  return workaround::device_path(d_temp_storage, temp_storage_bytes, d_keys, num_items, comp, begin_bit, end_bit, stream, debug_synchronous);
#endif
}


// sort descending
template<typename Key>
__host__ __device__
cudaError_t cub_sort_keys_wrapper(void *d_temp_storage,
                                  size_t &temp_storage_bytes,
                                  cub_::DoubleBuffer<Key> &d_keys,
                                  int num_items,
                                  thrust::greater<Key> comp,
                                  int begin_bit = 0,
                                  int end_bit = sizeof(Key) * 8,
                                  cudaStream_t stream = 0,
                                  bool debug_synchronous = false)
{
  struct workaround
  {
    __host__ 
    static cudaError_t host_path(void *d_temp_storage,
                                 size_t &temp_storage_bytes,
                                 cub_::DoubleBuffer<Key> &d_keys,
                                 int num_items,
                                 thrust::greater<Key>,
                                 int begin_bit,
                                 int end_bit,
                                 cudaStream_t stream,
                                 bool debug_synchronous)
    {
      return cub_::DeviceRadixSort::SortKeysDescending(d_temp_storage, temp_storage_bytes, d_keys, num_items, begin_bit, end_bit, stream, debug_synchronous);
    }

    __device__
    static cudaError_t device_path(void *d_temp_storage,
                                   size_t &temp_storage_bytes,
                                   cub_::DoubleBuffer<Key> &d_keys,
                                   int num_items,
                                   thrust::greater<Key>,
                                   int begin_bit,
                                   int end_bit,
                                   cudaStream_t stream,
                                   bool debug_synchronous)
    {
#if __BULK_HAS_CUDART__
      return cub_::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_keys, num_items, begin_bit, end_bit, stream, debug_synchronous);
#else
      return cudaErrorNotSupported;
#endif
    }
  };

#ifndef __CUDA_ARCH__
  return workaround::host_path(d_temp_storage, temp_storage_bytes, d_keys, num_items, comp, begin_bit, end_bit, stream, debug_synchronous);
#else
  return workaround::device_path(d_temp_storage, temp_storage_bytes, d_keys, num_items, comp, begin_bit, end_bit, stream, debug_synchronous);
#endif
}


// returns 1. the total size of temporary storage required for a key sort
//         2. an offset to the "d_temp_storage" parameter for CUB's sort
//         3. the value of the "temp_storage_bytes" parameter for CUB's sort
template<typename T, typename Compare>
__host__ __device__
thrust::tuple<size_t, size_t, size_t> compute_temporary_storage_requirements_for_radix_sort_n(size_t n, Compare comp, cudaStream_t stream)
{
  cub_::DoubleBuffer<T> dummy;

  // measure the number of additional temporary storage bytes required
  size_t num_additional_temp_storage_bytes = 0;
  thrust::system::cuda::detail::throw_on_error(cub_sort_keys_wrapper(0, num_additional_temp_storage_bytes, dummy, static_cast<int>(n), comp, 0, sizeof(T)*8, stream),
                                               "after cub_::DeviceRadixSort::SortKeys(0)");

  // XXX the additional temporary storage bytes
  //     must be allocated on a 16b aligned address
  typedef typename bulk_::detail::aligned_type<16>::type aligned_type;

  size_t num_double_buffer_bytes = n * sizeof(T);
  size_t num_aligned_double_buffer_bytes = thrust::detail::util::round_i(num_double_buffer_bytes, sizeof(aligned_type));
  size_t num_aligned_total_temporary_storage_bytes = num_aligned_double_buffer_bytes + num_additional_temp_storage_bytes;

  return thrust::make_tuple(num_aligned_total_temporary_storage_bytes, num_aligned_double_buffer_bytes, num_additional_temp_storage_bytes);
}


template<typename DerivedPolicy, typename T, typename Compare>
__host__ __device__
void stable_radix_sort_n(execution_policy<DerivedPolicy> &exec, T* first, size_t n, Compare comp)
{
  if(n > 1)
  {
    cudaStream_t s = stream(thrust::detail::derived_cast<DerivedPolicy>(exec));

    // compute temporary storage requirements
    size_t num_temporary_storage_bytes = 0;
    size_t offset_to_additional_temp_storage = 0;
    size_t num_additional_temp_storage_bytes = 0;
    thrust::tie(num_temporary_storage_bytes, offset_to_additional_temp_storage, num_additional_temp_storage_bytes) =
      compute_temporary_storage_requirements_for_radix_sort_n<T>(n, comp, s);

    // allocate storage
    thrust::detail::temporary_array<char,DerivedPolicy> temporary_storage(exec, num_temporary_storage_bytes);

    // set up double buffer
    cub_::DoubleBuffer<T> double_buffer;
    double_buffer.d_buffers[0] = thrust::raw_pointer_cast(&*first);
    double_buffer.d_buffers[1] = reinterpret_cast<T*>(reinterpret_cast<void*>(thrust::raw_pointer_cast(&temporary_storage[0])));

    thrust::system::cuda::detail::throw_on_error(cub_sort_keys_wrapper(thrust::raw_pointer_cast(&temporary_storage[offset_to_additional_temp_storage]),
                                                                       num_additional_temp_storage_bytes,
                                                                       double_buffer,
                                                                       static_cast<int>(n),
                                                                       comp,
                                                                       0,
                                                                       sizeof(T)*8,
                                                                       s),
                                                 "after cub_::DeviceRadixSort::SortKeys(1)");

    thrust::system::cuda::detail::synchronize_if_enabled("stable_radix_sort_n(): after cub_::DeviceRadixSort::SortKeys(1)");

    if(double_buffer.selector != 0)
    {
      T* temp_ptr = reinterpret_cast<T*>(double_buffer.d_buffers[1]);
      thrust::copy(exec, temp_ptr, temp_ptr + n, first);
    }
  }
}


} // end namespace stable_radix_sort_detail


template<typename DerivedPolicy,
         typename RandomAccessIterator>
__host__ __device__
void stable_radix_sort(execution_policy<DerivedPolicy> &exec,
                       RandomAccessIterator first,
                       RandomAccessIterator last,
                       thrust::less<typename thrust::iterator_value<RandomAccessIterator>::type> comp)
{
  stable_radix_sort_detail::stable_radix_sort_n(exec, thrust::raw_pointer_cast(&*first), last - first, comp);
}


template<typename DerivedPolicy,
         typename RandomAccessIterator>
__host__ __device__
void stable_radix_sort(execution_policy<DerivedPolicy> &exec,
                       RandomAccessIterator first,
                       RandomAccessIterator last,
                       thrust::greater<typename thrust::iterator_value<RandomAccessIterator>::type> comp)
{
  stable_radix_sort_detail::stable_radix_sort_n(exec, thrust::raw_pointer_cast(&*first), last - first, comp);
}


///////////////////////
// Key-Value Sorting //
///////////////////////


namespace stable_radix_sort_detail
{


// sort ascending
template<typename Key, typename Value>
__host__ __device__
cudaError_t cub_sort_pairs_wrapper(void *d_temp_storage,
                                   size_t &temp_storage_bytes,
                                   cub_::DoubleBuffer<Key> &d_keys,
                                   cub_::DoubleBuffer<Value> &d_values,
                                   int num_items,
                                   thrust::less<Key> comp,
                                   int begin_bit = 0,
                                   int end_bit = sizeof(Key) * 8,
                                   cudaStream_t stream = 0,
                                   bool debug_synchronous = false)
{
  struct workaround
  {
    __host__ 
    static cudaError_t host_path(void *d_temp_storage,
                                 size_t &temp_storage_bytes,
                                 cub_::DoubleBuffer<Key> &d_keys,
                                 cub_::DoubleBuffer<Value> &d_values,
                                 int num_items,
                                 thrust::less<Key>,
                                 int begin_bit,
                                 int end_bit,
                                 cudaStream_t stream,
                                 bool debug_synchronous)
    {
      return cub_::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_keys, d_values, num_items, begin_bit, end_bit, stream, debug_synchronous);
    }

    __device__
    static cudaError_t device_path(void *d_temp_storage,
                                   size_t &temp_storage_bytes,
                                   cub_::DoubleBuffer<Key> &d_keys,
                                   cub_::DoubleBuffer<Value> &d_values,
                                   int num_items,
                                   thrust::less<Key>,
                                   int begin_bit,
                                   int end_bit,
                                   cudaStream_t stream,
                                   bool debug_synchronous)
    {
#if __BULK_HAS_CUDART__
      return cub_::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_keys, d_values, num_items, begin_bit, end_bit, stream, debug_synchronous);
#else
      return cudaErrorNotSupported;
#endif
    }
  };

#ifndef __CUDA_ARCH__
  return workaround::host_path(d_temp_storage, temp_storage_bytes, d_keys, d_values, num_items, comp, begin_bit, end_bit, stream, debug_synchronous);
#else
  return workaround::device_path(d_temp_storage, temp_storage_bytes, d_keys, d_values, num_items, comp, begin_bit, end_bit, stream, debug_synchronous);
#endif
}


// sort descending
template<typename Key, typename Value>
__host__ __device__
cudaError_t cub_sort_pairs_wrapper(void *d_temp_storage,
                                   size_t &temp_storage_bytes,
                                   cub_::DoubleBuffer<Key> &d_keys,
                                   cub_::DoubleBuffer<Value> &d_values,
                                   int num_items,
                                   thrust::greater<Key> comp,
                                   int begin_bit = 0,
                                   int end_bit = sizeof(Key) * 8,
                                   cudaStream_t stream = 0,
                                   bool debug_synchronous = false)
{
  struct workaround
  {
    __host__ 
    static cudaError_t host_path(void *d_temp_storage,
                                 size_t &temp_storage_bytes,
                                 cub_::DoubleBuffer<Key> &d_keys,
                                 cub_::DoubleBuffer<Value> &d_values,
                                 int num_items,
                                 thrust::greater<Key>,
                                 int begin_bit,
                                 int end_bit,
                                 cudaStream_t stream,
                                 bool debug_synchronous)
    {
      return cub_::DeviceRadixSort::SortPairsDescending(d_temp_storage, temp_storage_bytes, d_keys, d_values, num_items, begin_bit, end_bit, stream, debug_synchronous);
    }

    __device__
    static cudaError_t device_path(void *d_temp_storage,
                                   size_t &temp_storage_bytes,
                                   cub_::DoubleBuffer<Key> &d_keys,
                                   cub_::DoubleBuffer<Value> &d_values,
                                   int num_items,
                                   thrust::greater<Key>,
                                   int begin_bit,
                                   int end_bit,
                                   cudaStream_t stream,
                                   bool debug_synchronous)
    {
#if __BULK_HAS_CUDART__
      return cub_::DeviceRadixSort::SortPairsDescending(d_temp_storage, temp_storage_bytes, d_keys, d_values, num_items, begin_bit, end_bit, stream, debug_synchronous);
#else
      return cudaErrorNotSupported;
#endif
    }
  };

#ifndef __CUDA_ARCH__
  return workaround::host_path(d_temp_storage, temp_storage_bytes, d_keys, d_values, num_items, comp, begin_bit, end_bit, stream, debug_synchronous);
#else
  return workaround::device_path(d_temp_storage, temp_storage_bytes, d_keys, d_values, num_items, comp, begin_bit, end_bit, stream, debug_synchronous);
#endif
}


// returns 1. the total size of temporary storage required for a key sort
//         2. an offset to the double buffer for values
//         3. an offset to the "d_temp_storage" parameter for CUB's sort
//         4. the value of the "temp_storage_bytes" parameter for CUB's sort
template<typename Key, typename Value, typename Compare>
__host__ __device__
thrust::tuple<size_t, size_t, size_t, size_t> compute_temporary_storage_requirements_for_radix_sort_by_key_n(size_t n, Compare comp, cudaStream_t stream)
{
  cub_::DoubleBuffer<Key> dummy_keys;
  cub_::DoubleBuffer<Value> dummy_values;

  // measure the number of additional temporary storage bytes required
  size_t num_additional_temp_storage_bytes = 0;
  thrust::system::cuda::detail::throw_on_error(cub_sort_pairs_wrapper(0, num_additional_temp_storage_bytes, dummy_keys, dummy_values, static_cast<int>(n), comp, 0, sizeof(Key)*8, stream),
                                               "after cub_::DeviceRadixSort::SortPairs(0)");

  // XXX the additional temporary storage bytes
  //     must be allocated on a 16b aligned address
  typedef typename bulk_::detail::aligned_type<16>::type aligned_type;

  size_t num_keys_double_buffer_bytes = n * sizeof(Key);

  // align up the allocation for the keys double buffer
  size_t num_aligned_keys_double_buffer_bytes = thrust::detail::util::round_i(num_keys_double_buffer_bytes, sizeof(aligned_type));

  size_t num_values_double_buffer_bytes = n * sizeof(Value);

  // align up the allocation for both double buffers
  size_t num_aligned_double_buffer_bytes = thrust::detail::util::round_i(num_aligned_keys_double_buffer_bytes + num_values_double_buffer_bytes, sizeof(aligned_type));

  size_t num_aligned_total_temporary_storage_bytes = num_aligned_double_buffer_bytes + num_additional_temp_storage_bytes;

  return thrust::make_tuple(num_aligned_total_temporary_storage_bytes, num_aligned_keys_double_buffer_bytes, num_aligned_double_buffer_bytes, num_additional_temp_storage_bytes);
}


// sort values directly
template<typename DerivedPolicy,
         typename Key,
         typename Value,
         typename Compare>
__host__ __device__
void stable_radix_sort_by_key_n(execution_policy<DerivedPolicy> &exec,
                                Key* first1,
                                size_t n,
                                Value* first2,
                                Compare comp)
{
  if(n > 1)
  {
    cudaStream_t s = stream(thrust::detail::derived_cast<DerivedPolicy>(exec));

    // compute temporary storage requirements
    size_t num_temporary_storage_bytes = 0;
    size_t offset_to_values_buffer = 0;
    size_t offset_to_additional_temp_storage = 0;
    size_t num_additional_temp_storage_bytes = 0;
    thrust::tie(num_temporary_storage_bytes, offset_to_values_buffer, offset_to_additional_temp_storage, num_additional_temp_storage_bytes) =
      compute_temporary_storage_requirements_for_radix_sort_by_key_n<Key,Value>(n, comp, s);

    // allocate storage
    thrust::detail::temporary_array<char,DerivedPolicy> temporary_storage(exec, num_temporary_storage_bytes);

    // set up double buffers
    cub_::DoubleBuffer<Key> double_buffer_keys;
    double_buffer_keys.d_buffers[0] = thrust::raw_pointer_cast(&*first1);
    double_buffer_keys.d_buffers[1] = reinterpret_cast<Key*>(reinterpret_cast<void*>(thrust::raw_pointer_cast(&temporary_storage[0])));

    cub_::DoubleBuffer<Value> double_buffer_values;
    double_buffer_values.d_buffers[0] = thrust::raw_pointer_cast(&*first2);
    double_buffer_values.d_buffers[1] = reinterpret_cast<Value*>(reinterpret_cast<void*>(thrust::raw_pointer_cast(&temporary_storage[offset_to_values_buffer])));

    thrust::system::cuda::detail::throw_on_error(cub_sort_pairs_wrapper(thrust::raw_pointer_cast(&temporary_storage[offset_to_additional_temp_storage]),
                                                                        num_additional_temp_storage_bytes,
                                                                        double_buffer_keys,
                                                                        double_buffer_values,
                                                                        static_cast<int>(n),
                                                                        comp,
                                                                        0,
                                                                        sizeof(Key)*8,
                                                                        s),
                                                 "after cub_::DeviceRadixSort::SortPairs(1)");

    thrust::system::cuda::detail::synchronize_if_enabled("stable_radix_sort_by_key_n(): after cub_::DeviceRadixSort::SortPairs(1)");

    if(double_buffer_keys.selector != 0)
    {
      Key* temp_ptr = reinterpret_cast<Key*>(double_buffer_keys.d_buffers[1]);
      thrust::copy(exec, temp_ptr, temp_ptr + n, first1);
    }

    if(double_buffer_values.selector != 0)
    {
      Value* temp_ptr = reinterpret_cast<Value*>(double_buffer_values.d_buffers[1]);
      thrust::copy(exec, temp_ptr, temp_ptr + n, first2);
    }
  }
}


} // end stable_radix_sort_detail


template<typename DerivedPolicy,
         typename RandomAccessIterator1,
         typename RandomAccessIterator2>
__host__ __device__
void stable_radix_sort_by_key(execution_policy<DerivedPolicy> &exec,
                              RandomAccessIterator1 first1,
                              RandomAccessIterator1 last1,
                              RandomAccessIterator2 first2,
                              thrust::less<typename thrust::iterator_value<RandomAccessIterator1>::type> comp)
{
  stable_radix_sort_detail::stable_radix_sort_by_key_n(exec,
                                                       thrust::raw_pointer_cast(&*first1),
                                                       last1 - first1,
                                                       thrust::raw_pointer_cast(&*first2),
                                                       comp);
}


template<typename DerivedPolicy,
         typename RandomAccessIterator1,
         typename RandomAccessIterator2>
__host__ __device__
void stable_radix_sort_by_key(execution_policy<DerivedPolicy> &exec,
                              RandomAccessIterator1 first1,
                              RandomAccessIterator1 last1,
                              RandomAccessIterator2 first2,
                              thrust::greater<typename thrust::iterator_value<RandomAccessIterator1>::type> comp)
{
  stable_radix_sort_detail::stable_radix_sort_by_key_n(exec,
                                                       thrust::raw_pointer_cast(&*first1),
                                                       last1 - first1,
                                                       thrust::raw_pointer_cast(&*first2),
                                                       comp);
}


} // end namespace detail
} // end namespace detail
} // end namespace cuda
} // end namespace system
} // end namespace thrust


#endif // THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_NVCC

