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


/*! \file scan.inl
 *  \brief Inline file for scan.h.
 */

#include <thrust/detail/config.h>
#include <thrust/scan.h>
#include <thrust/detail/seq.h>
#include <thrust/detail/temporary_array.h>
#include <thrust/detail/type_traits/function_traits.h>
#include <thrust/system/cuda/detail/decomposition.h>
#include <thrust/system/cuda/detail/bulk.h>


namespace thrust
{
namespace system
{
namespace cuda
{
namespace detail
{
namespace scan_detail
{


struct inclusive_scan_n
{
  template<typename ConcurrentGroup, typename InputIterator, typename Size, typename OutputIterator, typename T, typename BinaryFunction>
  __device__ void operator()(ConcurrentGroup &this_group, InputIterator first, Size n, OutputIterator result, T init, BinaryFunction binary_op)
  {
    bulk_::inclusive_scan(this_group, first, first + n, result, init, binary_op);
  }


  template<typename ConcurrentGroup, typename InputIterator, typename Size, typename OutputIterator, typename BinaryFunction>
  __device__ void operator()(ConcurrentGroup &this_group, InputIterator first, Size n, OutputIterator result, BinaryFunction binary_op)
  {
    bulk_::inclusive_scan(this_group, first, first + n, result, binary_op);
  }
};


struct exclusive_scan_n
{
  template<typename ConcurrentGroup, typename InputIterator, typename Size, typename OutputIterator, typename T, typename BinaryFunction>
  __device__ void operator()(ConcurrentGroup &this_group, InputIterator first, Size n, OutputIterator result, T init, BinaryFunction binary_op)
  {
    bulk_::exclusive_scan(this_group, first, first + n, result, init, binary_op);
  }
};


struct inclusive_downsweep
{
  template<typename ConcurrentGroup, typename RandomAccessIterator1, typename Decomposition, typename RandomAccessIterator2, typename RandomAccessIterator3, typename BinaryFunction>
  __device__ void operator()(ConcurrentGroup &this_group,
                             RandomAccessIterator1 first,
                             Decomposition decomp,
                             RandomAccessIterator2 carries_first,
                             RandomAccessIterator3 result,
                             BinaryFunction binary_op)
  {
    typename Decomposition::range range = decomp[this_group.index()];
  
    RandomAccessIterator1 last = first + range.second;
    first += range.first;
    result += range.first;
  
    if(this_group.index() == 0)
    {
      bulk_::inclusive_scan(this_group, first, last, result, binary_op);
    }
    else
    {
      typename thrust::iterator_value<RandomAccessIterator2>::type carry = carries_first[this_group.index() - 1];

      bulk_::inclusive_scan(this_group, first, last, result, carry, binary_op);
    }
  }
};


struct exclusive_downsweep
{
  template<typename ConcurrentGroup, typename RandomAccessIterator1, typename Decomposition, typename RandomAccessIterator2, typename RandomAccessIterator3, typename BinaryFunction>
  __device__ void operator()(ConcurrentGroup &this_group,
                             RandomAccessIterator1 first,
                             Decomposition decomp,
                             RandomAccessIterator2 carries_first,
                             RandomAccessIterator3 result,
                             BinaryFunction binary_op)
  {
    typename Decomposition::range range = decomp[this_group.index()];
  
    RandomAccessIterator1 last = first + range.second;
    first += range.first;
    result += range.first;
  
    typename thrust::iterator_value<RandomAccessIterator2>::type carry = carries_first[this_group.index()];

    bulk_::exclusive_scan(this_group, first, last, result, carry, binary_op);
  }
};


template<typename T> struct accumulate_tiles_tuning_impl;


template<> struct accumulate_tiles_tuning_impl<int>
{
  // determined from empirical testing on k20c & nvcc 6.5 RC
  static const int groupsize = 128;
  static const int grainsize = 9;
};


template<> struct accumulate_tiles_tuning_impl<double>
{
  // determined from empirical testing on k20c & nvcc 6.5 RC
  static const int groupsize = 128;
  static const int grainsize = 9;
};


// determined from empirical testing on k20c
template<typename T>
  struct accumulate_tiles_tuning
{
  static const int groupsize =
    sizeof(T) <=     sizeof(int) ? accumulate_tiles_tuning_impl<int>::groupsize :
    sizeof(T) <= 2 * sizeof(int) ? accumulate_tiles_tuning_impl<double>::groupsize :
    128;
  
  static const int grainsize =
    sizeof(T) <=     sizeof(int) ? accumulate_tiles_tuning_impl<int>::grainsize :
    sizeof(T) <= 2 * sizeof(int) ? accumulate_tiles_tuning_impl<double>::grainsize :
    3;
};

// this specialization accomodates scan_by_key,
// whose intermediate type is a tuple
template<typename T1, typename T2>
  struct accumulate_tiles_tuning<thrust::tuple<T1,T2> >
{
  // determined from empirical testing on k20c
  static const int groupsize = 128;
  static const int grainsize = ((sizeof(T1) + sizeof(T2)) <= (2 * sizeof(double))) ? 5 : 3;
};





struct accumulate_tiles
{
  template<typename ConcurrentGroup, typename RandomAccessIterator1, typename Decomposition, typename RandomAccessIterator2, typename BinaryFunction>
  __device__ void operator()(ConcurrentGroup &this_group,
                             RandomAccessIterator1 first,
                             Decomposition decomp,
                             RandomAccessIterator2 result,
                             BinaryFunction binary_op)
  {
    typedef typename thrust::iterator_value<RandomAccessIterator1>::type value_type;
    
    typename Decomposition::range range = decomp[this_group.index()];

    const bool commutative = thrust::detail::is_commutative<BinaryFunction>::value;

    // for a commutative accumulate, it's much faster to pass the last value as the init for some reason
    value_type init = commutative ? first[range.second-1] : first[range.first];

    value_type sum = commutative ?
      bulk_::accumulate(this_group, first + range.first, first + range.second - 1, init, binary_op) :
      bulk_::accumulate(this_group, first + range.first + 1, first + range.second, init, binary_op);

    if(this_group.this_exec.index() == 0)
    {
      result[this_group.index()] = sum;
    } // end if
  } // end operator()
}; // end accumulate_tiles


template<typename DerivedPolicy,
         typename InputIterator,
         typename OutputIterator,
         typename AssociativeOperator>
__host__ __device__
OutputIterator inclusive_scan(execution_policy<DerivedPolicy> &exec,
                              InputIterator first,
                              InputIterator last,
                              OutputIterator result,
                              AssociativeOperator binary_op)
{
  typedef typename bulk_::detail::scan_detail::scan_intermediate<
    InputIterator,
    OutputIterator,
    AssociativeOperator
  >::type intermediate_type;

  typedef typename thrust::iterator_difference<InputIterator>::type Size;

  Size n = last - first;

  cudaStream_t s = stream(thrust::detail::derived_cast(exec));
  
  const Size threshold_of_parallelism = 20000;

  if(n < threshold_of_parallelism)
  {
    const Size groupsize =
      sizeof(intermediate_type) <= 2 * sizeof(int) ? 512 :
      sizeof(intermediate_type) <= 4 * sizeof(int) ? 256 :
      128;

    typedef bulk_::detail::scan_detail::scan_buffer<groupsize,3,InputIterator,OutputIterator,AssociativeOperator> heap_type;
    Size heap_size = sizeof(heap_type);
    bulk_::async(bulk_::grid<groupsize,3>(1, heap_size, s), scan_detail::inclusive_scan_n(), bulk_::root.this_exec, first, n, result, binary_op);

    // XXX WAR unused variable warning
    (void) groupsize;
  } // end if
  else
  {
    const Size groupsize = scan_detail::accumulate_tiles_tuning<intermediate_type>::groupsize;
    const Size grainsize = scan_detail::accumulate_tiles_tuning<intermediate_type>::grainsize;

    const Size tile_size = groupsize * grainsize;
    Size num_tiles = (n + tile_size - 1) / tile_size;

    // 20 determined from empirical testing on k20c & GTX 480
    Size subscription = 20;
    Size num_groups = thrust::min<Size>(subscription * bulk_::concurrent_group<>::hardware_concurrency(), num_tiles);

    aligned_decomposition<Size> decomp(n, num_groups, tile_size);

    thrust::detail::temporary_array<intermediate_type,DerivedPolicy> carries(exec, num_groups);
    	
    // Run the parallel raking reduce as an upsweep.
    // n loads + num_groups stores
    Size heap_size = groupsize * sizeof(intermediate_type);
    bulk_::async(bulk_::grid<groupsize,grainsize>(num_groups,heap_size,s), scan_detail::accumulate_tiles(), bulk_::root.this_exec, first, decomp, carries.begin(), binary_op);

    // scan the sums to get the carries
    // num_groups loads + num_groups stores
    const Size groupsize2 = sizeof(intermediate_type) <= 2 * sizeof(int) ? 256 : 128;
    const Size grainsize2 = 3;
    typedef bulk_::detail::scan_detail::scan_buffer<groupsize2,grainsize2,InputIterator,OutputIterator,AssociativeOperator> heap_type2;
    heap_size = sizeof(heap_type2);
    bulk_::async(bulk_::grid<groupsize2,grainsize2>(1,heap_size,s), scan_detail::inclusive_scan_n(), bulk_::root.this_exec, carries.begin(), num_groups, carries.begin(), binary_op);

    // do the downsweep - n loads, n stores
    typedef bulk_::detail::scan_detail::scan_buffer<
      groupsize,
      grainsize,
      InputIterator,OutputIterator,AssociativeOperator
    > heap_type3;
    heap_size = sizeof(heap_type3);
    bulk_::async(bulk_::grid<groupsize,grainsize>(num_groups,heap_size,s), scan_detail::inclusive_downsweep(), bulk_::root.this_exec, first, decomp, carries.begin(), result, binary_op);

    // XXX WAR unused variable warnings
    (void) groupsize2;
    (void) grainsize2;
  } // end else

  return result + n;
} // end inclusive_scan()


template<typename DerivedPolicy,
         typename InputIterator,
         typename OutputIterator,
         typename T,
         typename AssociativeOperator>
__host__ __device__
OutputIterator exclusive_scan(execution_policy<DerivedPolicy> &exec,
                              InputIterator first,
                              InputIterator last,
                              OutputIterator result,
                              T init,
                              AssociativeOperator binary_op)
{
  typedef typename bulk_::detail::scan_detail::scan_intermediate<
    InputIterator,
    OutputIterator,
    AssociativeOperator
  >::type intermediate_type;

  typedef typename thrust::iterator_difference<InputIterator>::type Size;

  Size n = last - first;

  cudaStream_t s = stream(thrust::detail::derived_cast(exec));
  
  const Size threshold_of_parallelism = 20000;

  if(n < threshold_of_parallelism)
  {
    const Size groupsize =
      sizeof(intermediate_type) <= 2 * sizeof(int) ? 512 :
      sizeof(intermediate_type) <= 4 * sizeof(int) ? 256 :
      128;

    typedef bulk_::detail::scan_detail::scan_buffer<groupsize,3,InputIterator,OutputIterator,AssociativeOperator> heap_type;
    Size heap_size = sizeof(heap_type);
    bulk_::async(bulk_::grid<groupsize,3>(1, heap_size, s), scan_detail::exclusive_scan_n(), bulk_::root.this_exec, first, n, result, init, binary_op);

    // XXX WAR unused variable warning
    (void) groupsize;
  } // end if
  else
  {
    const Size groupsize = scan_detail::accumulate_tiles_tuning<intermediate_type>::groupsize;
    const Size grainsize = scan_detail::accumulate_tiles_tuning<intermediate_type>::grainsize;

    const Size tile_size = groupsize * grainsize;
    Size num_tiles = (n + tile_size - 1) / tile_size;

    // 20 determined from empirical testing on k20c & GTX 480
    Size subscription = 20;
    Size num_groups = thrust::min<Size>(subscription * bulk_::concurrent_group<>::hardware_concurrency(), num_tiles);

    aligned_decomposition<Size> decomp(n, num_groups, tile_size);

    thrust::detail::temporary_array<intermediate_type,DerivedPolicy> carries(exec, num_groups);
    	
    // Run the parallel raking reduce as an upsweep.
    // n loads + num_groups stores
    Size heap_size = groupsize * sizeof(intermediate_type);
    bulk_::async(bulk_::grid<groupsize,grainsize>(num_groups,heap_size,s), scan_detail::accumulate_tiles(), bulk_::root.this_exec, first, decomp, carries.begin(), binary_op);
    
    // scan the sums to get the carries
    // num_groups loads + num_groups stores
    const Size groupsize2 = sizeof(intermediate_type) <= 2 * sizeof(int) ? 256 : 128;
    const Size grainsize2 = 3;

    typedef bulk_::detail::scan_detail::scan_buffer<groupsize2,grainsize2,InputIterator,OutputIterator,AssociativeOperator> heap_type2;
    heap_size = sizeof(heap_type2);
    bulk_::async(bulk_::grid<groupsize2,grainsize2>(1,heap_size,s), scan_detail::exclusive_scan_n(), bulk_::root.this_exec, carries.begin(), num_groups, carries.begin(), init, binary_op);

    // do the downsweep - n loads, n stores
    typedef bulk_::detail::scan_detail::scan_buffer<
      groupsize,
      grainsize,
      InputIterator,OutputIterator,AssociativeOperator
    > heap_type3;
    heap_size = sizeof(heap_type3);
    bulk_::async(bulk_::grid<groupsize,grainsize>(num_groups,heap_size,s), scan_detail::exclusive_downsweep(), bulk_::root.this_exec, first, decomp, carries.begin(), result, binary_op);

    // XXX WAR unused variable warnings
    (void) groupsize2;
    (void) grainsize2;
  } // end else

  return result + n;
} // end exclusive_scan()


} // end scan_detail


template<typename DerivedPolicy,
         typename InputIterator,
         typename OutputIterator,
         typename AssociativeOperator>
__host__ __device__
OutputIterator inclusive_scan(execution_policy<DerivedPolicy> &exec,
                              InputIterator first,
                              InputIterator last,
                              OutputIterator result,
                              AssociativeOperator binary_op)
{
  // we're attempting to launch a kernel, assert we're compiling with nvcc
  // ========================================================================
  // X Note to the user: If you've found this line due to a compiler error, X
  // X you need to compile your code using nvcc, rather than g++ or cl.exe  X
  // ========================================================================
  THRUST_STATIC_ASSERT( (thrust::detail::depend_on_instantiation<InputIterator, THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_NVCC>::value) );

  struct workaround
  {
    __host__ __device__
    static OutputIterator parallel_path(execution_policy<DerivedPolicy> &exec,
                                        InputIterator first,
                                        InputIterator last,
                                        OutputIterator result,
                                        AssociativeOperator binary_op)
    {
      return thrust::system::cuda::detail::scan_detail::inclusive_scan(exec, first, last, result, binary_op);
    }

    __host__ __device__
    static OutputIterator sequential_path(execution_policy<DerivedPolicy> &,
                                          InputIterator first,
                                          InputIterator last,
                                          OutputIterator result,
                                          AssociativeOperator binary_op)
    {
      return thrust::inclusive_scan(thrust::seq, first, last, result, binary_op);
    }
  };

#if __BULK_HAS_CUDART__
  return workaround::parallel_path(exec, first, last, result, binary_op);
#else
  return workaround::sequential_path(exec, first, last, result, binary_op);
#endif
} // end inclusive_scan()


template<typename DerivedPolicy,
         typename InputIterator,
         typename OutputIterator,
         typename T,
         typename AssociativeOperator>
__host__ __device__
OutputIterator exclusive_scan(execution_policy<DerivedPolicy> &exec,
                              InputIterator first,
                              InputIterator last,
                              OutputIterator result,
                              T init,
                              AssociativeOperator binary_op)
{
  // we're attempting to launch a kernel, assert we're compiling with nvcc
  // ========================================================================
  // X Note to the user: If you've found this line due to a compiler error, X
  // X you need to compile your code using nvcc, rather than g++ or cl.exe  X
  // ========================================================================
  THRUST_STATIC_ASSERT( (thrust::detail::depend_on_instantiation<InputIterator, THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_NVCC>::value) );

  struct workaround
  {
    __host__ __device__
    static OutputIterator parallel_path(execution_policy<DerivedPolicy> &exec,
                                        InputIterator first,
                                        InputIterator last,
                                        OutputIterator result,
                                        T init,
                                        AssociativeOperator binary_op)
    {
      return thrust::system::cuda::detail::scan_detail::exclusive_scan(exec, first, last, result, init, binary_op);
    }

    __host__ __device__
    static OutputIterator sequential_path(execution_policy<DerivedPolicy> &,
                                          InputIterator first,
                                          InputIterator last,
                                          OutputIterator result,
                                          T init,
                                          AssociativeOperator binary_op)
    {
      return thrust::exclusive_scan(thrust::seq, first, last, result, init, binary_op);
    }
  };

#if __BULK_HAS_CUDART__
  return workaround::parallel_path(exec, first, last, result, init, binary_op);
#else
  return workaround::sequential_path(exec, first, last, result, init, binary_op);
#endif
} // end exclusive_scan()


} // end namespace detail
} // end namespace cuda
} // end namespace system
} // end namespace thrust

