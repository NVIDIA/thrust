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


/*! \file scan.inl
 *  \brief Inline file for scan.h.
 */

#include <thrust/detail/config.h>
#include <thrust/detail/static_assert.h>
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


// avoid accidentally picking up some other installation of bulk that
// may be floating around
namespace bulk_ = thrust::system::cuda::detail::bulk;


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

    bulk::exclusive_scan(this_group, first, last, result, carry, binary_op);
  }
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


} // end scan_detail


template<typename DerivedPolicy,
         typename InputIterator,
         typename OutputIterator,
         typename AssociativeOperator>
  OutputIterator inclusive_scan(execution_policy<DerivedPolicy> &exec,
                                InputIterator first,
                                InputIterator last,
                                OutputIterator result,
                                AssociativeOperator binary_op)
{
  namespace bulk_ = thrust::system::cuda::detail::bulk;

  typedef typename bulk_::detail::scan_detail::scan_intermediate<
    InputIterator,
    OutputIterator,
    AssociativeOperator
  >::type intermediate_type;

  typedef typename thrust::iterator_difference<InputIterator>::type Size;

  Size n = last - first;
  
  const Size threshold_of_parallelism = 20000;

  if(n < threshold_of_parallelism)
  {
    const Size groupsize =
      sizeof(intermediate_type) <= 2 * sizeof(int) ? 512 :
      sizeof(intermediate_type) <= 4 * sizeof(int) ? 256 :
      128;

    typedef bulk_::detail::scan_detail::scan_buffer<groupsize,3,InputIterator,OutputIterator,AssociativeOperator> heap_type;
    Size heap_size = sizeof(heap_type);
    bulk_::async(bulk_::con<groupsize,3>(heap_size), scan_detail::inclusive_scan_n(), bulk::root, first, n, result, binary_op);

    // XXX WAR unused variable warning
    (void) groupsize;
  } // end if
  else
  {
    // determined from empirical testing on k20c
    const Size groupsize =
      sizeof(intermediate_type) <=     sizeof(int) ? 128 :
      sizeof(intermediate_type) <= 2 * sizeof(int) ? 256 :
      128;

    const Size grainsize =
      sizeof(intermediate_type) <=     sizeof(int) ? 9 :
      sizeof(intermediate_type) <= 2 * sizeof(int) ? 5 :
      3;

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
    bulk_::async(bulk_::grid<groupsize,grainsize>(num_groups,heap_size), scan_detail::accumulate_tiles(), bulk_::root.this_exec, first, decomp, carries.begin(), binary_op);

    // scan the sums to get the carries
    // num_groups loads + num_groups stores
    const Size groupsize2 = sizeof(intermediate_type) <= 2 * sizeof(int) ? 256 : 128;
    const Size grainsize2 = 3;
    typedef bulk_::detail::scan_detail::scan_buffer<groupsize2,grainsize2,InputIterator,OutputIterator,AssociativeOperator> heap_type2;
    heap_size = sizeof(heap_type2);
    bulk_::async(bulk_::con<groupsize2,grainsize2>(heap_size), scan_detail::inclusive_scan_n(), bulk_::root, carries.begin(), num_groups, carries.begin(), binary_op);

    // do the downsweep - n loads, n stores
    typedef bulk_::detail::scan_detail::scan_buffer<
      groupsize,
      grainsize,
      InputIterator,OutputIterator,AssociativeOperator
    > heap_type3;
    heap_size = sizeof(heap_type3);
    bulk_::async(bulk_::grid<groupsize,grainsize>(num_groups,heap_size), scan_detail::inclusive_downsweep(), bulk_::root.this_exec, first, decomp, carries.begin(), result, binary_op);

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
  OutputIterator exclusive_scan(execution_policy<DerivedPolicy> &exec,
                                InputIterator first,
                                InputIterator last,
                                OutputIterator result,
                                T init,
                                AssociativeOperator binary_op)
{
  namespace bulk_ = thrust::system::cuda::detail::bulk;

  typedef typename bulk_::detail::scan_detail::scan_intermediate<
    InputIterator,
    OutputIterator,
    AssociativeOperator
  >::type intermediate_type;

  typedef typename thrust::iterator_difference<InputIterator>::type Size;

  Size n = last - first;
  
  const Size threshold_of_parallelism = 20000;

  if(n < threshold_of_parallelism)
  {
    const Size groupsize =
      sizeof(intermediate_type) <= 2 * sizeof(int) ? 512 :
      sizeof(intermediate_type) <= 4 * sizeof(int) ? 256 :
      128;

    typedef bulk_::detail::scan_detail::scan_buffer<groupsize,3,InputIterator,OutputIterator,AssociativeOperator> heap_type;
    Size heap_size = sizeof(heap_type);
    bulk_::async(bulk_::con<groupsize,3>(heap_size), scan_detail::exclusive_scan_n(), bulk::root, first, n, result, init, binary_op);

    // XXX WAR unused variable warning
    (void) groupsize;
  } // end if
  else
  {
    // determined from empirical testing on k20c
    const Size groupsize =
      sizeof(intermediate_type) <=     sizeof(int) ? 128 :
      sizeof(intermediate_type) <= 2 * sizeof(int) ? 256 :
      128;

    const Size grainsize =
      sizeof(intermediate_type) <=     sizeof(int) ? 9 :
      sizeof(intermediate_type) <= 2 * sizeof(int) ? 5 :
      3;

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
    bulk_::async(bulk_::grid<groupsize,grainsize>(num_groups,heap_size), scan_detail::accumulate_tiles(), bulk::root.this_exec, first, decomp, carries.begin(), binary_op);
    
    // scan the sums to get the carries
    // num_groups loads + num_groups stores
    const Size groupsize2 = sizeof(intermediate_type) <= 2 * sizeof(int) ? 256 : 128;
    const Size grainsize2 = 3;

    typedef bulk_::detail::scan_detail::scan_buffer<groupsize2,grainsize2,InputIterator,OutputIterator,AssociativeOperator> heap_type2;
    heap_size = sizeof(heap_type2);
    bulk_::async(bulk_::con<groupsize2,grainsize2>(heap_size), scan_detail::exclusive_scan_n(), bulk::root, carries.begin(), num_groups, carries.begin(), init, binary_op);

    // do the downsweep - n loads, n stores
    typedef bulk_::detail::scan_detail::scan_buffer<
      groupsize,
      grainsize,
      InputIterator,OutputIterator,AssociativeOperator
    > heap_type3;
    heap_size = sizeof(heap_type3);
    bulk_::async(bulk_::grid<groupsize,grainsize>(num_groups,heap_size), scan_detail::exclusive_downsweep(), bulk::root.this_exec, first, decomp, carries.begin(), result, binary_op);

    // XXX WAR unused variable warnings
    (void) groupsize2;
    (void) grainsize2;
  } // end else

  return result + n;
} // end exclusive_scan()


} // end namespace detail
} // end namespace cuda
} // end namespace system
} // end namespace thrust

