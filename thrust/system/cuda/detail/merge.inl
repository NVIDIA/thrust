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
#include <thrust/merge.h>
#include <thrust/detail/seq.h>
#include <thrust/system/cuda/detail/merge.h>
#include <thrust/system/cuda/detail/bulk.h>
#include <thrust/detail/temporary_array.h>
#include <thrust/tabulate.h>
#include <thrust/iterator/detail/join_iterator.h>
#include <thrust/detail/minmax.h>
#include <thrust/system/cuda/detail/execute_on_stream.h>

namespace thrust
{
namespace system
{
namespace cuda
{
namespace detail
{
namespace merge_detail
{


template<std::size_t groupsize, std::size_t grainsize, typename RandomAccessIterator1, typename Size,typename RandomAccessIterator2, typename RandomAccessIterator3, typename RandomAccessIterator4, typename Compare>
__device__
RandomAccessIterator4
  staged_merge(bulk_::concurrent_group<bulk_::agent<grainsize>,groupsize> &exec,
               RandomAccessIterator1 first1, Size n1,
               RandomAccessIterator2 first2, Size n2,
               RandomAccessIterator3 stage,
               RandomAccessIterator4 result,
               Compare comp)
{
  // copy into the stage
  bulk_::copy_n(bulk_::bound<groupsize * grainsize>(exec),
                thrust::detail::make_join_iterator(first1, n1, first2),
                n1 + n2,
                stage);

  // inplace merge in the stage
  bulk_::inplace_merge(bulk_::bound<groupsize * grainsize>(exec),
                       stage, stage + n1, stage + n1 + n2,
                       comp);
  
  // copy to the result
  // XXX this might be slightly faster with a bounded copy_n
  return bulk_::copy_n(exec, stage, n1 + n2, result);
} // end staged_merge()


struct merge_kernel
{
  template<std::size_t groupsize, std::size_t grainsize, typename RandomAccessIterator1, typename Size, typename RandomAccessIterator2, typename RandomAccessIterator3, typename RandomAccessIterator4, typename Compare>
  __device__
  void operator()(bulk_::concurrent_group<bulk_::agent<grainsize>,groupsize> &g,
                  RandomAccessIterator1 first1, Size n1,
                  RandomAccessIterator2 first2, Size n2,
                  RandomAccessIterator3 merge_paths_first,
                  RandomAccessIterator4 result,
                  Compare comp)
  {
    typedef int size_type;

    size_type elements_per_group = g.size() * g.this_exec.grainsize();

    // determine the ranges to merge
    size_type mp0  = merge_paths_first[g.index()];
    size_type mp1  = merge_paths_first[g.index()+1];
    size_type diag = elements_per_group * g.index();

    size_type local_size1 = mp1 - mp0;
    size_type local_size2 = thrust::min<size_type>(n1 + n2, diag + elements_per_group) - mp1 - diag + mp0;

    first1 += mp0;
    first2 += diag - mp0;
    result += elements_per_group * g.index();

    // XXX this assumes that RandomAccessIterator2's value_type converts to RandomAccessIterator1's value_type
    typedef typename thrust::iterator_value<RandomAccessIterator1>::type value_type;

#if __CUDA_ARCH__ >= 200
    // merge through a stage
    value_type *stage = reinterpret_cast<value_type*>(bulk_::malloc(g, elements_per_group * sizeof(value_type)));

    if(bulk_::is_on_chip(stage))
    {
      staged_merge(g,
                   first1, local_size1,
                   first2, local_size2,
                   bulk_::on_chip_cast(stage),
                   result,
                   comp);
    } // end if
    else
    {
      staged_merge(g,
                   first1, local_size1,
                   first2, local_size2,
                   stage,
                   result,
                   comp);
    } // end else

    bulk_::free(g, stage);
#else
    __shared__ bulk_::uninitialized_array<value_type, groupsize * grainsize> stage;
    staged_merge(g, first1, local_size1, first2, local_size2, stage.data(), result, comp);
#endif
  } // end operator()
}; // end merge_kernel


template<typename Size, typename RandomAccessIterator1,typename RandomAccessIterator2, typename Compare>
struct locate_merge_path
{
  Size partition_size;
  RandomAccessIterator1 first1, last1;
  RandomAccessIterator2 first2, last2;
  Compare comp;

  __host__ __device__
  locate_merge_path(Size partition_size, RandomAccessIterator1 first1, RandomAccessIterator1 last1, RandomAccessIterator2 first2, RandomAccessIterator2 last2, Compare comp)
    : partition_size(partition_size),
      first1(first1), last1(last1),
      first2(first2), last2(last2),
      comp(comp)
  {}

  template<typename Index>
  __device__
  Size operator()(Index i)
  {
    Size n1 = last1 - first1;
    Size n2 = last2 - first2;
    Size diag = thrust::min<Size>(partition_size * i, n1 + n2);
    return bulk_::merge_path(first1, n1, first2, n2, diag, comp);
  }
};


template<typename DerivedPolicy,
         typename RandomAccessIterator1,
         typename RandomAccessIterator2, 
	 typename RandomAccessIterator3,
         typename Compare>
__host__ __device__
RandomAccessIterator3 merge(execution_policy<DerivedPolicy> &exec,
                            RandomAccessIterator1 first1,
                            RandomAccessIterator1 last1,
                            RandomAccessIterator2 first2,
                            RandomAccessIterator2 last2,
                            RandomAccessIterator3 result,
                            Compare comp)
{
  typedef typename thrust::iterator_value<RandomAccessIterator1>::type value_type;
  typedef typename thrust::iterator_difference<RandomAccessIterator1>::type difference_type;
  typedef int size_type;

  // determined through empirical testing on K20c
  const size_type groupsize = (sizeof(value_type) == sizeof(int)) ? 256 : 256 + 32;
  const size_type grainsize = (sizeof(value_type) == sizeof(int)) ? 9   : 5;
  
  const size_type tile_size = groupsize * grainsize;

  difference_type n = (last1 - first1) + (last2 - first2);
  difference_type num_groups = (n + tile_size - 1) / tile_size;

  thrust::detail::temporary_array<size_type,DerivedPolicy> merge_paths(exec, num_groups + 1);

  thrust::tabulate(exec, merge_paths.begin(), merge_paths.end(), merge_detail::locate_merge_path<size_type,RandomAccessIterator1,RandomAccessIterator2,Compare>(tile_size,first1,last1,first2,last2,comp));

  // merge partitions
  size_type heap_size = tile_size * sizeof(value_type);
  bulk_::concurrent_group<bulk_::agent<grainsize>,groupsize> g(heap_size);
  bulk_::async(bulk_::par(stream(thrust::detail::derived_cast(exec)), g, num_groups), merge_detail::merge_kernel(), bulk_::root.this_exec, first1, last1 - first1, first2, last2 - first2, merge_paths.begin(), result, comp);

  return result + n;
} // end merge()


} // end merge_detail


template<typename DerivedPolicy,
         typename RandomAccessIterator1,
         typename RandomAccessIterator2, 
	 typename RandomAccessIterator3,
         typename Compare>
__host__ __device__
RandomAccessIterator3 merge(execution_policy<DerivedPolicy> &exec,
                            RandomAccessIterator1 first1,
                            RandomAccessIterator1 last1,
                            RandomAccessIterator2 first2,
                            RandomAccessIterator2 last2,
                            RandomAccessIterator3 result,
                            Compare comp)
{
  // we're attempting to launch a kernel, assert we're compiling with nvcc
  // ========================================================================
  // X Note to the user: If you've found this line due to a compiler error, X
  // X you need to compile your code using nvcc, rather than g++ or cl.exe  X
  // ========================================================================
  THRUST_STATIC_ASSERT( (thrust::detail::depend_on_instantiation<RandomAccessIterator1, THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_NVCC>::value) );

  struct workaround
  {
    __host__ __device__
    static RandomAccessIterator3 parallel_path(execution_policy<DerivedPolicy> &exec,
                                               RandomAccessIterator1 first1,
                                               RandomAccessIterator1 last1,
                                               RandomAccessIterator2 first2,
                                               RandomAccessIterator2 last2,
                                               RandomAccessIterator3 result,
                                               Compare comp)
    {
      return thrust::system::cuda::detail::merge_detail::merge(exec, first1, last1, first2, last2, result, comp);
    }

    __host__ __device__
    static RandomAccessIterator3 sequential_path(execution_policy<DerivedPolicy> &,
                                                 RandomAccessIterator1 first1,
                                                 RandomAccessIterator1 last1,
                                                 RandomAccessIterator2 first2,
                                                 RandomAccessIterator2 last2,
                                                 RandomAccessIterator3 result,
                                                 Compare comp)
    {
      return thrust::merge(thrust::seq, first1, last1, first2, last2, result, comp);
    }
  };

#if __BULK_HAS_CUDART__
  return workaround::parallel_path(exec, first1, last1, first2, last2, result, comp);
#else
  return workaround::sequential_path(exec, first1, last1, first2, last2, result, comp);
#endif
} // end merge()


} // end namespace detail
} // end namespace cuda
} // end namespace system
} // end namespace thrust

