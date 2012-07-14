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

// don't attempt to #include this file without omp support
#if (THRUST_DEVICE_COMPILER_IS_OMP_CAPABLE == THRUST_TRUE)
#include <omp.h>
#endif // omp support

#include <thrust/iterator/iterator_traits.h>
#include <thrust/system/detail/generic/select_system.h>
#include <thrust/system/cpp/detail/sort.h>
#include <thrust/system/cpp/detail/merge.h>
#include <thrust/system/cpp/detail/tag.h>
#include <thrust/detail/temporary_array.h>

namespace thrust
{
namespace system
{
namespace omp
{
namespace detail
{
namespace sort_detail
{


template <typename System,
          typename RandomAccessIterator,
          typename StrictWeakOrdering>
void inplace_merge(dispatchable<System> &system,
                   RandomAccessIterator first,
                   RandomAccessIterator middle,
                   RandomAccessIterator last,
                   StrictWeakOrdering comp)
{
  typedef typename thrust::iterator_value<RandomAccessIterator>::type value_type;

  thrust::detail::temporary_array<value_type,System> a(system, first, middle);
  thrust::detail::temporary_array<value_type,System> b(system, middle, last);

  thrust::system::cpp::detail::merge(system, a.begin(), a.end(), b.begin(), b.end(), first, comp);
}


template <typename System,
          typename RandomAccessIterator1,
          typename RandomAccessIterator2,
          typename StrictWeakOrdering>
void inplace_merge_by_key(dispatchable<System> &system,
                          RandomAccessIterator1 first1,
                          RandomAccessIterator1 middle1,
                          RandomAccessIterator1 last1,
                          RandomAccessIterator2 first2,
                          StrictWeakOrdering comp)
{
  typedef typename thrust::iterator_value<RandomAccessIterator1>::type value_type1;
  typedef typename thrust::iterator_value<RandomAccessIterator2>::type value_type2;

  RandomAccessIterator2 middle2 = first2 + (middle1 - first1);
  RandomAccessIterator2 last2   = first2 + (last1   - first1);

  thrust::detail::temporary_array<value_type1,System> lhs1(system, first1, middle1);
  thrust::detail::temporary_array<value_type1,System> rhs1(system, middle1, last1);
  thrust::detail::temporary_array<value_type2,System> lhs2(system, first2, middle2);
  thrust::detail::temporary_array<value_type2,System> rhs2(system, middle2, last2);

  thrust::system::cpp::detail::merge_by_key
    (system,
     lhs1.begin(), lhs1.end(), rhs1.begin(), rhs1.end(),
     lhs2.begin(), rhs2.begin(),
     first1, first2, comp);
}


} // end sort_detail


template<typename System,
         typename RandomAccessIterator,
         typename StrictWeakOrdering>
void stable_sort(dispatchable<System> &system,
                 RandomAccessIterator first,
                 RandomAccessIterator last,
                 StrictWeakOrdering comp)
{
  // we're attempting to launch an omp kernel, assert we're compiling with omp support
  // ========================================================================
  // X Note to the user: If you've found this line due to a compiler error, X
  // X you need to enable OpenMP support in your compiler.                  X
  // ========================================================================
  THRUST_STATIC_ASSERT( (thrust::detail::depend_on_instantiation<RandomAccessIterator,
                        (THRUST_DEVICE_COMPILER_IS_OMP_CAPABLE == THRUST_TRUE)>::value) );

#if (THRUST_DEVICE_COMPILER_IS_OMP_CAPABLE == THRUST_TRUE)
  typedef typename thrust::iterator_difference<RandomAccessIterator>::type IndexType;
  
  if (first == last)
    return;

  #pragma omp parallel
  {
    thrust::system::detail::internal::uniform_decomposition<IndexType> decomp(last - first, 1, omp_get_num_threads());

    // process id
    IndexType p_i = omp_get_thread_num();

    // every thread sorts its own tile
    if (p_i < decomp.size())
    {
      thrust::system::cpp::detail::stable_sort(system,
                                               first + decomp[p_i].begin(),
                                               first + decomp[p_i].end(),
                                               comp);
    }

    #pragma omp barrier

    IndexType nseg = decomp.size();
    IndexType h = 2;

    // keep track of which sub-range we're processing
    IndexType a=p_i, b=p_i, c=p_i+1;

    while( nseg>1 )
    {
        if(c >= decomp.size())
          c = decomp.size() - 1;

        if((p_i % h) == 0 && c > b)
        {
          thrust::system::omp::detail::sort_detail::inplace_merge
            (system,
             first + decomp[a].begin(),
             first + decomp[b].end(),
             first + decomp[c].end(),
             comp);
            b = c;
            c += h;
        }

        nseg = (nseg + 1) / 2;
        h *= 2;

        #pragma omp barrier
    }
  }
#endif // THRUST_DEVICE_COMPILER_IS_OMP_CAPABLE
}


template<typename System,
         typename RandomAccessIterator1,
         typename RandomAccessIterator2,
         typename StrictWeakOrdering>
void stable_sort_by_key(dispatchable<System> &system,
                        RandomAccessIterator1 keys_first,
                        RandomAccessIterator1 keys_last,
                        RandomAccessIterator2 values_first,
                        StrictWeakOrdering comp)
{
  // we're attempting to launch an omp kernel, assert we're compiling with omp support
  // ========================================================================
  // X Note to the user: If you've found this line due to a compiler error, X
  // X you need to enable OpenMP support in your compiler.                  X
  // ========================================================================
  THRUST_STATIC_ASSERT( (thrust::detail::depend_on_instantiation<RandomAccessIterator1,
                        (THRUST_DEVICE_COMPILER_IS_OMP_CAPABLE == THRUST_TRUE)>::value) );

#if (THRUST_DEVICE_COMPILER_IS_OMP_CAPABLE == THRUST_TRUE)
  typedef typename thrust::iterator_difference<RandomAccessIterator1>::type IndexType;
  
  if (keys_first == keys_last)
    return;

  #pragma omp parallel
  {
    thrust::system::detail::internal::uniform_decomposition<IndexType> decomp(keys_last - keys_first, 1, omp_get_num_threads());

    // process id
    IndexType p_i = omp_get_thread_num();

    // every thread sorts its own tile
    if (p_i < decomp.size())
    {
      thrust::system::cpp::detail::stable_sort_by_key(system,
                                                      keys_first + decomp[p_i].begin(),
                                                      keys_first + decomp[p_i].end(),
                                                      values_first + decomp[p_i].begin(),
                                                      comp);
    }

    #pragma omp barrier

    IndexType nseg = decomp.size();
    IndexType h = 2;

    // keep track of which sub-range we're processing
    IndexType a=p_i, b=p_i, c=p_i+1;

    while( nseg>1 )
    {
        if(c >= decomp.size())
          c = decomp.size() - 1;

        if((p_i % h) == 0 && c > b)
        {
          thrust::system::omp::detail::sort_detail::inplace_merge_by_key
            (system,
             keys_first + decomp[a].begin(),
             keys_first + decomp[b].end(),
             keys_first + decomp[c].end(),
             values_first + decomp[a].begin(),
             comp);
            b = c;
            c += h;
        }

        nseg = (nseg + 1) / 2;
        h *= 2;

        #pragma omp barrier
    }
  }
#endif // THRUST_DEVICE_COMPILER_IS_OMP_CAPABLE
}


} // end namespace detail
} // end namespace omp
} // end namespace system
} // end namespace thrust

