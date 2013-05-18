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

#include <thrust/detail/config.h>

// do not attempt to compile this file with any other compiler
#if THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_NVCC

#include <thrust/detail/copy.h>
#include <thrust/gather.h>
#include <thrust/sequence.h>
#include <thrust/iterator/iterator_traits.h>

#include <thrust/detail/temporary_array.h>
#include <thrust/detail/type_traits.h>
#include <thrust/detail/util/align.h>
#include <thrust/detail/raw_pointer_cast.h>


__THRUST_DISABLE_MSVC_POSSIBLE_LOSS_OF_DATA_WARNING_BEGIN


#include <thrust/system/cuda/detail/detail/b40c/radixsort_api.h>

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


template<typename DerivedPolicy, typename ContiguousIterator>
struct ensure_aligned_range
{
  typedef typename thrust::iterator_value<ContiguousIterator>::type value_type;
  typedef value_type * iterator;

  __host__ __device__
  ensure_aligned_range(thrust::execution_policy<DerivedPolicy> &exec,
                       ContiguousIterator first,
                       ContiguousIterator last)
    : m_do_copy_back(true),
      m_exec(exec),
      m_orig_first(thrust::raw_pointer_cast(&*first)),
      m_first(0),
      m_last(0),
      m_storage(exec)
  {
    // if the range isn't aligned, copy it into aligned temporary storage
    if(thrust::detail::util::is_aligned(m_orig_first, 2 * sizeof(value_type)))
    {
      m_first = thrust::raw_pointer_cast(m_orig_first);
      m_last  = thrust::raw_pointer_cast(&*last);
    }
    else
    {
      // XXX temporary_array can't resize, so we need to placement construct this
      ::new(static_cast<void*>(&m_storage)) thrust::detail::temporary_array<value_type, DerivedPolicy>(exec, first, last);

      m_first = thrust::raw_pointer_cast(&*m_storage.begin());
      m_last  = thrust::raw_pointer_cast(&*m_storage.end());
    }
  }

  __host__ __device__
  ~ensure_aligned_range()
  {
    // copy back to the original range, if the temporary storage exists
    // and the client requires it
    if(m_storage.size() && m_do_copy_back)
    {
      thrust::copy(m_exec, m_storage.begin(), m_storage.end(), m_orig_first);
    }
  }

  __host__ __device__
  value_type *begin() const
  {
    return m_first;
  }

  __host__ __device__
  value_type *end() const
  {
    return m_last;
  }

  __host__ __device__
  void do_copy_back(bool b)
  {
    m_do_copy_back = b;
  }

  bool m_do_copy_back;
  thrust::execution_policy<DerivedPolicy> &m_exec;
  value_type *m_orig_first;
  value_type *m_first;
  value_type *m_last;
  thrust::detail::temporary_array<value_type, DerivedPolicy> m_storage;
};


} // end namespace stable_radix_sort_detail


template<typename DerivedPolicy,
         typename RandomAccessIterator>
__host__ __device__
void stable_radix_sort(execution_policy<DerivedPolicy> &exec,
                       RandomAccessIterator first,
                       RandomAccessIterator last)
{
  typedef typename thrust::iterator_value<RandomAccessIterator>::type K;

  // ensure data is properly aligned
  stable_radix_sort_detail::ensure_aligned_range<DerivedPolicy,RandomAccessIterator> aligned_data(exec, first, last);
  
  unsigned int num_elements = aligned_data.end() - aligned_data.begin();
  
  thrust::system::cuda::detail::detail::b40c_thrust::RadixSortingEnactor<K> sorter(num_elements);
  thrust::system::cuda::detail::detail::b40c_thrust::RadixSortStorage<K>    storage;
  
  // allocate temporary buffers
  thrust::detail::temporary_array<K,    DerivedPolicy> temp_keys(exec, num_elements);
  thrust::detail::temporary_array<int,  DerivedPolicy> temp_spine(exec, sorter.SpineElements());
  thrust::detail::temporary_array<bool, DerivedPolicy> temp_from_alt(exec, 2);
  
  // define storage
  storage.d_keys             = thrust::raw_pointer_cast(aligned_data.begin());
  storage.d_alt_keys         = thrust::raw_pointer_cast(&temp_keys[0]);
  storage.d_spine            = thrust::raw_pointer_cast(&temp_spine[0]);
  storage.d_from_alt_storage = thrust::raw_pointer_cast(&temp_from_alt[0]);
  
  // perform the sort
  sorter.EnactSort(storage);
  
  // radix sort sometimes leaves results in the alternate buffers
  if(storage.using_alternate_storage)
  {
    thrust::copy(exec, temp_keys.begin(), temp_keys.end(), first);

    // since we've updated the data in first, aligned_data doesn't need to do it
    aligned_data.do_copy_back(false);
  }
}


///////////////////////
// Key-Value Sorting //
///////////////////////


namespace stable_radix_sort_detail
{


// sort values directly
template<typename DerivedPolicy,
         typename RandomAccessIterator1,
         typename RandomAccessIterator2>
__host__ __device__
void stable_radix_sort_by_key(execution_policy<DerivedPolicy> &exec,
                              RandomAccessIterator1 first1,
                              RandomAccessIterator1 last1,
                              RandomAccessIterator2 first2,
                              thrust::detail::true_type)
{
  typedef typename thrust::iterator_value<RandomAccessIterator1>::type K;
  typedef typename thrust::iterator_value<RandomAccessIterator2>::type V;
  
  unsigned int num_elements = last1 - first1;

  // ensure data is properly aligned
  ensure_aligned_range<DerivedPolicy,RandomAccessIterator1> aligned_keys(exec, first1, last1);
  ensure_aligned_range<DerivedPolicy,RandomAccessIterator2> aligned_values(exec, first2, first2 + num_elements);
  
  thrust::system::cuda::detail::detail::b40c_thrust::RadixSortingEnactor<K,V> sorter(num_elements);
  thrust::system::cuda::detail::detail::b40c_thrust::RadixSortStorage<K,V>    storage;
  
  // allocate temporary buffers
  thrust::detail::temporary_array<K,    DerivedPolicy> temp_keys(exec, num_elements);
  thrust::detail::temporary_array<V,    DerivedPolicy> temp_values(exec, num_elements);
  thrust::detail::temporary_array<int,  DerivedPolicy> temp_spine(exec, sorter.SpineElements());
  thrust::detail::temporary_array<bool, DerivedPolicy> temp_from_alt(exec, 2);
  
  // define storage
  storage.d_keys             = thrust::raw_pointer_cast(aligned_keys.begin());
  storage.d_values           = thrust::raw_pointer_cast(aligned_values.begin());
  storage.d_alt_keys         = thrust::raw_pointer_cast(&temp_keys[0]);
  storage.d_alt_values       = thrust::raw_pointer_cast(&temp_values[0]);
  storage.d_spine            = thrust::raw_pointer_cast(&temp_spine[0]);
  storage.d_from_alt_storage = thrust::raw_pointer_cast(&temp_from_alt[0]);
  
  // perform the sort
  sorter.EnactSort(storage);
  
  // radix sort sometimes leaves results in the alternate buffers
  if(storage.using_alternate_storage)
  {
    thrust::copy(exec, temp_keys.begin(),   temp_keys.end(),   first1);
    thrust::copy(exec, temp_values.begin(), temp_values.end(), first2);

    // since we've updated the originals, the aligned data doesn't need to do it
    aligned_keys.do_copy_back(false);
    aligned_values.do_copy_back(false);
  }
}


// sort values indirectly
template<typename DerivedPolicy, 
         typename RandomAccessIterator1,
         typename RandomAccessIterator2>
__host__ __device__
void stable_radix_sort_by_key(execution_policy<DerivedPolicy> &exec,
                              RandomAccessIterator1 first1,
                              RandomAccessIterator1 last1,
                              RandomAccessIterator2 first2,
                              thrust::detail::false_type)
{
  typedef typename thrust::iterator_value<RandomAccessIterator2>::type V;
  
  unsigned int num_elements = last1 - first1;
  
  // sort with integer values and then permute the real values accordingly
  thrust::detail::temporary_array<unsigned int,DerivedPolicy> permutation(exec, num_elements);
  thrust::sequence(exec, permutation.begin(), permutation.end());
  
  stable_radix_sort_by_key(exec, first1, last1, permutation.begin());
  
  // copy values into temp vector and then permute
  thrust::detail::temporary_array<V,DerivedPolicy> temp_values(exec, first2, first2 + num_elements);
  
  // permute values
  thrust::gather(exec,
                 permutation.begin(), permutation.end(),
                 temp_values.begin(),
                 first2);
}


} // end stable_radix_sort_detail


template<typename DerivedPolicy,
         typename RandomAccessIterator1,
         typename RandomAccessIterator2>
__host__ __device__
void stable_radix_sort_by_key(execution_policy<DerivedPolicy> &exec,
                              RandomAccessIterator1 first1,
                              RandomAccessIterator1 last1,
                              RandomAccessIterator2 first2)
{
  typedef typename thrust::iterator_value<RandomAccessIterator2>::type V;
  
  // decide how to handle values
  const bool sort_values_directly = thrust::detail::is_trivial_iterator<RandomAccessIterator2>::value &&
                                    thrust::detail::is_arithmetic<V>::value &&
                                    sizeof(V) <= 8;    // TODO profile this
  
  // XXX WAR unused variable warning
  (void) sort_values_directly;
  
  stable_radix_sort_detail::stable_radix_sort_by_key(exec, first1, last1, first2, 
    thrust::detail::integral_constant<bool, sort_values_directly>());
}


} // end namespace detail
} // end namespace detail
} // end namespace cuda
} // end namespace system
} // end namespace thrust


__THRUST_DISABLE_MSVC_POSSIBLE_LOSS_OF_DATA_WARNING_END


#endif // THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_NVCC

