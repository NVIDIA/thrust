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

#pragma once

#include <thrust/system/cuda/detail/bulk/detail/config.hpp>
#include <thrust/system/cuda/detail/bulk/execution_policy.hpp>
#include <thrust/system/cuda/detail/bulk/malloc.hpp>
#include <thrust/system/cuda/detail/bulk/algorithm/copy.hpp>
#include <thrust/system/cuda/detail/bulk/algorithm/gather.hpp>
#include <thrust/system/cuda/detail/bulk/uninitialized.hpp>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/detail/join_iterator.h>
#include <thrust/detail/minmax.h>


BULK_NAMESPACE_PREFIX
namespace bulk
{


template<typename RandomAccessIterator1, typename Size, typename RandomAccessIterator2, typename Compare>
__device__
Size merge_path(RandomAccessIterator1 first1, Size n1,
                RandomAccessIterator2 first2, Size n2,
                Size diag,
                Compare comp)
{
  Size begin = thrust::max<Size>(Size(0), diag - n2);
  Size end = thrust::min<Size>(diag, n1);
  
  while(begin < end)
  {
    Size mid = (begin + end) >> 1;

    if(comp(first2[diag - 1 - mid], first1[mid]))
    {
      end = mid;
    } // end if
    else
    {
      begin = mid + 1;
    } // end else
  } // end while

  return begin;
} // end merge_path()


template<std::size_t bound,
         std::size_t grainsize,
         typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator,
         typename Compare>
__device__
OutputIterator merge(const bulk::bounded<bound,agent<grainsize> > &e,
                     InputIterator1 first1, InputIterator1 last1,
                     InputIterator2 first2, InputIterator2 last2,
                     OutputIterator result,
                     Compare comp)
{
  typedef typename bulk::bounded<bound,bulk::agent<grainsize> >::size_type size_type;

  typedef typename thrust::iterator_value<InputIterator1>::type value_type1;
  typedef typename thrust::iterator_value<InputIterator2>::type value_type2;

  size_type n = (last1 - first1) + (last2 - first2);

  // XXX uninitialized is a speed-down in this instance
  //bulk::uninitialized<value_type1>   key_a;
  value_type1   key_a;
  size_type     n1 = last1 - first1;
  size_type     idx1 = 0;

  if(n1 > 0)
  {
    //key_a.construct(first1[idx1]);
    key_a = first1[idx1];
  } // end if

  //bulk::uninitialized<value_type2>   key_b;
  value_type2   key_b;
  size_type     n2 = last2 - first2;
  size_type     idx2 = 0;

  if(n2 > 0)
  {
    //key_b.construct(first2[idx2]);
    key_b = first2[idx2];
  } // end if
  
  // avoid branching when possible
  if(bound <= n)
  {
    for(size_type i = 0; i < grainsize; ++i)
    {
      bool p = (idx2 >= n2) || ((idx1 < n1) && !comp(key_b, key_a));
      
      result[i] = p ? key_a : key_b;

      if(p)
      {
        ++idx1;
        
        // use of min avoids conditional load
        key_a = first1[min(idx1, n1 - 1)];
      } // end if
      else
      {
        ++idx2;

        // use of min avoids conditional load
        key_b = first2[min(idx2, n2 - 1)];
      } // end else
    } // end for
  } // end if
  else
  {
    for(size_type i = 0; i < grainsize; ++i)
    {
      if(i < n)
      {
        bool p = (idx2 >= n2) || ((idx1 < n1) && !comp(key_b, key_a));
        
        result[i] = p ? key_a : key_b;

        if(p)
        {
          ++idx1;

          // use of min avoids conditional load
          key_a = first1[min(idx1, n1 - 1)];
        } // end if
        else
        {
          ++idx2;

          // use of min avoids conditional load
          key_b = first2[min(idx2, n2 - 1)];
        } // end else
      } // end if
    } // end for
  } // end else

//  if(n1 > 0)
//  {
//    key_a.destroy();
//  } // end if
//
//  if(n2 > 0)
//  {
//    key_b.destroy();
//  } // end if

  return result + n;
} // end merge


template<std::size_t bound, std::size_t grainsize,
         typename RandomAccessIterator1,
         typename RandomAccessIterator2,
         typename RandomAccessIterator3,
         typename RandomAccessIterator4,
         typename RandomAccessIterator5,
         typename RandomAccessIterator6,
         typename Compare>
__device__
thrust::pair<RandomAccessIterator5,RandomAccessIterator6>
  merge_by_key(const bulk::bounded<bound,bulk::agent<grainsize> > &,
               RandomAccessIterator1 keys_first1, RandomAccessIterator1 keys_last1,
               RandomAccessIterator2 keys_first2, RandomAccessIterator2 keys_last2,
               RandomAccessIterator3 values_first1,
               RandomAccessIterator4 values_first2,
               RandomAccessIterator5 keys_result,
               RandomAccessIterator6 values_result,
               Compare comp)
{
  typedef typename bulk::bounded<bound,bulk::agent<grainsize> >::size_type size_type;

  typedef typename thrust::iterator_value<RandomAccessIterator1>::type key_type1;
  typedef typename thrust::iterator_value<RandomAccessIterator2>::type key_type2;

  typedef typename thrust::iterator_value<RandomAccessIterator3>::type value_type1;
  typedef typename thrust::iterator_value<RandomAccessIterator4>::type value_type2;

  size_type n = (keys_last1 - keys_first1) + (keys_last2 - keys_first2);

  // XXX uninitialized is a speed-down in this instance
  //bulk::uninitialized<key_type1>   key_a;
  //bulk::uninitialized<value_type1> val_a;
  key_type1   key_a;
  value_type1 val_a;
  size_type   n1 = keys_last1 - keys_first1;
  size_type   idx1 = 0;

  if(n1 > 0)
  {
    //key_a.construct(keys_first1[idx1]);
    //val_a.construct(values_first1[idx1]);
    key_a = keys_first1[idx1];
    val_a = values_first1[idx1];
  } // end if

  //bulk::uninitialized<key_type2>   key_b;
  //bulk::uninitialized<value_type2> val_b;
  key_type2   key_b;
  value_type2 val_b;
  size_type   n2 = keys_last2 - keys_first2;
  size_type   idx2 = 0;

  if(n2 > 0)
  {
    //key_b.construct(keys_first2[idx2]);
    //val_b.construct(values_first2[idx2]);
    key_b = keys_first2[idx2];
    val_b = values_first2[idx2];
  } // end if
  
  // avoid branching when possible
  if(bound <= n)
  {
    for(size_type i = 0; i < grainsize; ++i)
    {
      bool p = (idx2 >= n2) || ((idx1 < n1) && !comp(key_b, key_a));
      
      keys_result[i]   = p ? key_a : key_b;
      values_result[i] = p ? val_a : val_b;

      if(p)
      {
        ++idx1;

        // use of min avoids conditional loads
        key_a = keys_first1[min(idx1, n1 - 1)];
        val_a = values_first1[min(idx1, n1 - 1)];
      } // end if
      else
      {
        ++idx2;

        // use of min avoids conditional loads
        key_b = keys_first2[min(idx2, n2 - 1)];
        val_b = values_first2[min(idx2, n2 - 1)];
      } // end else
    } // end for
  } // end if
  else
  {
    for(size_type i = 0; i < grainsize; ++i)
    {
      if(i < n)
      {
        bool p = (idx2 >= n2) || ((idx1 < n1) && !comp(key_b, key_a));
        
        keys_result[i]   = p ? key_a : key_b;
        values_result[i] = p ? val_a : val_b;

        if(p)
        {
          ++idx1;

          // use of min avoids conditional loads
          key_a = keys_first1[min(idx1, n1 - 1)];
          val_a = values_first1[min(idx1, n1 - 1)];
        } // end if
        else
        {
          ++idx2;

          // use of min avoids conditional loads
          key_b = keys_first2[min(idx2, n2 - 1)];
          val_b = values_first2[min(idx2, n2 - 1)];
        } // end else
      } // end if
    } // end for
  } // end else

//  if(n1 > 0)
//  {
//    key_a.destroy();
//    val_a.destroy();
//  } // end if
//
//  if(n2 > 0)
//  {
//    key_b.destroy();
//    val_b.destroy();
//  } // end if

  return thrust::make_pair(keys_result + n, values_result + n);
} // end merge_by_key()


template<std::size_t bound, std::size_t groupsize, std::size_t grainsize, typename RandomAccessIterator, typename Compare>
__device__
typename thrust::detail::enable_if<
  (bound <= groupsize * grainsize)
>::type
inplace_merge(bulk::bounded<
                bound,
                bulk::concurrent_group<
                  bulk::agent<grainsize>,
                  groupsize
                >
              > &g,
              RandomAccessIterator first, RandomAccessIterator middle, RandomAccessIterator last,
              Compare comp)
{
  typedef typename bulk::concurrent_group<bulk::agent<grainsize>,groupsize>::size_type size_type;

  size_type n1 = middle - first;
  size_type n2 = last - middle;

  // find the start of each local merge
  size_type local_offset = grainsize * g.this_exec.index();

  size_type mp = bulk::merge_path(first, n1, middle, n2, local_offset, comp);
  
  // do a local sequential merge
  size_type local_offset1 = mp;
  size_type local_offset2 = n1 + local_offset - mp;

  typedef typename thrust::iterator_value<RandomAccessIterator>::type value_type;
  value_type local_result[grainsize];
  bulk::merge(bulk::bound<grainsize>(g.this_exec),
              first + local_offset1, middle,
              first + local_offset2, last,
              local_result,
              comp);

  g.wait();

  // copy local result back to source
  // this is faster than getting the size from merge's result
  size_type local_size = thrust::max<size_type>(0, thrust::min<size_type>(grainsize, n1 + n2 - local_offset));
  bulk::copy_n(bulk::bound<grainsize>(g.this_exec), local_result, local_size, first + local_offset); 

  g.wait();
} // end inplace_merge()


template<std::size_t bound, std::size_t groupsize, std::size_t grainsize,
         typename RandomAccessIterator1,
         typename RandomAccessIterator2,
         typename RandomAccessIterator3,
         typename Compare>
__device__
typename thrust::detail::enable_if<
  (bound <= groupsize * grainsize),
  RandomAccessIterator3
>::type
merge(bulk::bounded<
        bound,
        bulk::concurrent_group<
          bulk::agent<grainsize>,
          groupsize
        >
      > &g,
      RandomAccessIterator1 first1, RandomAccessIterator1 last1,
      RandomAccessIterator2 first2, RandomAccessIterator2 last2,
      RandomAccessIterator3 result,
      Compare comp)
{
  typedef typename bulk::concurrent_group<bulk::agent<grainsize>,groupsize>::size_type size_type;

  size_type n1 = last1 - first1;
  size_type n2 = last2 - first2;

  // find the start of each local merge
  size_type local_offset = grainsize * g.this_exec.index();

  size_type mp = bulk::merge_path(first1, n1, first2, n2, local_offset, comp);
  
  // do a local sequential merge
  size_type local_offset1 = mp;
  size_type local_offset2 = local_offset - mp;
  
  typedef typename thrust::iterator_value<RandomAccessIterator3>::type value_type;
  value_type local_result[grainsize];
  bulk::merge(bulk::bound<grainsize>(g.this_exec),
              first1 + local_offset1, last1,
              first2 + local_offset2, last2,
              local_result,
              comp);

  // store local result
  // this is faster than getting the size from merge's result
  size_type local_size = thrust::max<size_type>(0, thrust::min<size_type>(grainsize, n1 + n2 - local_offset));
  bulk::copy_n(bulk::bound<grainsize>(g.this_exec), local_result, local_size, result + local_offset); 

  g.wait();

  return result + thrust::min<size_type>(groupsize * grainsize, n1 + n2);
} // end merge()


namespace detail
{
namespace merge_detail
{


// XXX this should take a bounded
template<std::size_t groupsize, std::size_t grainsize, typename RandomAccessIterator1, typename RandomAccessIterator2, typename RandomAccessIterator3, typename RandomAccessIterator4, typename Compare>
__device__
RandomAccessIterator4
  bounded_merge_with_buffer(bulk::concurrent_group<bulk::agent<grainsize>,groupsize> &exec,
                            RandomAccessIterator1 first1, RandomAccessIterator1 last1,
                            RandomAccessIterator2 first2, RandomAccessIterator2 last2,
                            RandomAccessIterator3 buffer,
                            RandomAccessIterator4 result,
                            Compare comp)
{
  typedef typename bulk::concurrent_group<bulk::agent<grainsize>,groupsize>::size_type size_type;

  size_type n1 = last1 - first1;
  size_type n2 = last2 - first2;

  // copy into the buffer
  bulk::copy_n(bulk::bound<groupsize * grainsize>(exec),
               thrust::detail::make_join_iterator(first1, n1, first2),
               n1 + n2,
               buffer);

  // inplace merge in the buffer
  bulk::inplace_merge(bulk::bound<groupsize * grainsize>(exec),
                      buffer, buffer + n1, buffer + n1 + n2,
                      comp);
  
  // copy to the result
  // XXX this might be slightly faster with a bounded copy_n
  return bulk::copy_n(exec, buffer, n1 + n2, result);
} // end bounded_merge_with_buffer()


} // end merge_detail
} // end detail


template<std::size_t groupsize, std::size_t grainsize, typename RandomAccessIterator1, typename RandomAccessIterator2, typename RandomAccessIterator3, typename Compare>
__device__
RandomAccessIterator3 merge(bulk::concurrent_group<bulk::agent<grainsize>,groupsize> &exec,
                            RandomAccessIterator1 first1, RandomAccessIterator1 last1,
                            RandomAccessIterator2 first2, RandomAccessIterator2 last2,
                            RandomAccessIterator3 result,
                            Compare comp)
{
  typedef typename bulk::concurrent_group<bulk::agent<grainsize>,groupsize>::size_type size_type;

  typedef typename thrust::iterator_value<RandomAccessIterator3>::type value_type;

  value_type *buffer = reinterpret_cast<value_type*>(bulk::malloc(exec, exec.size() * exec.grainsize() * sizeof(value_type)));

  size_type chunk_size = exec.size() * exec.this_exec.grainsize();

  size_type n1 = last1 - first1;
  size_type n2 = last2 - first2;

  // avoid the search & loop when possible
  if(n1 + n2 <= chunk_size)
  {
    result = detail::merge_detail::bounded_merge_with_buffer(exec, first1, last1, first2, last2, buffer, result, comp);
  } // end if
  else
  {
    while((first1 < last1) || (first2 < last2))
    {
      size_type n1 = last1 - first1;
      size_type n2 = last2 - first2;

      size_type diag = thrust::min<size_type>(chunk_size, n1 + n2);

      size_type mp = bulk::merge_path(first1, n1, first2, n2, diag, comp);

      result = detail::merge_detail::bounded_merge_with_buffer(exec,
                                                               first1, first1 + mp,
                                                               first2, first2 + diag - mp,
                                                               buffer,
                                                               result,
                                                               comp);

      first1 += mp;
      first2 += diag - mp;
    } // end while
  } // end else

  bulk::free(exec, buffer);

  return result;
} // end merge()


template<std::size_t groupsize, std::size_t grainsize,
         typename RandomAccessIterator1,
         typename RandomAccessIterator2,
         typename RandomAccessIterator3,
         typename RandomAccessIterator4,
         typename RandomAccessIterator5,
         typename RandomAccessIterator6,
         typename Compare>
__device__
thrust::pair<RandomAccessIterator5,RandomAccessIterator6>
merge_by_key(bulk::bounded<
               groupsize*grainsize,
               bulk::concurrent_group<bulk::agent<grainsize>, groupsize>
             > &g,
             RandomAccessIterator1 keys_first1, RandomAccessIterator1 keys_last1,
             RandomAccessIterator2 keys_first2, RandomAccessIterator2 keys_last2,
             RandomAccessIterator3 values_first1,
             RandomAccessIterator4 values_first2,
             RandomAccessIterator5 keys_result,
             RandomAccessIterator6 values_result,
             Compare comp)
{
  typedef typename bulk::concurrent_group<bulk::agent<grainsize>,groupsize>::size_type size_type;

  typedef typename thrust::iterator_value<RandomAccessIterator5>::type key_type;

#if __CUDA_ARCH__ >= 200
  union
  {
    key_type  *keys;
    size_type *indices;
  } stage;

  stage.keys = static_cast<key_type*>(bulk::malloc(g, groupsize * grainsize * thrust::max(sizeof(key_type), sizeof(size_type))));
#else
  __shared__ union
  {
    key_type  keys[groupsize * grainsize];
    size_type indices[groupsize * grainsize];
  } stage;
#endif

  size_type n1 = keys_last1 - keys_first1;
  size_type n2 = keys_last2 - keys_first2;
  size_type  n = n1 + n2;
  
  // copy keys into stage
  bulk::copy_n(g,
               thrust::detail::make_join_iterator(keys_first1, n1, keys_first2),
               n,
               stage.keys);

  // find the start of each agent's sequential merge
  size_type diag = thrust::min<size_type>(n1 + n2, grainsize * g.this_exec.index());
  size_type mp = bulk::merge_path(stage.keys, n1, stage.keys + n1, n2, diag, comp);
  
  // compute the ranges of the sources in the stage.
  size_type start1 = mp;
  size_type start2 = n1 + diag - mp;

  size_type end1 = n1;
  size_type end2 = n1 + n2;
  
  // each agent merges sequentially
  key_type  results[grainsize];
  size_type indices[grainsize];
  bulk::merge_by_key(bulk::bound<grainsize>(g.this_exec),
                     stage.keys + start1, stage.keys + end1,
                     stage.keys + start2, stage.keys + end2,
                     thrust::make_counting_iterator<size_type>(start1),
                     thrust::make_counting_iterator<size_type>(start2),
                     results,
                     indices,
                     comp);
  g.wait();
  
  // each agent stores merged keys back to the stage
  size_type local_offset = grainsize * g.this_exec.index();
  size_type local_size = thrust::max<size_type>(0, thrust::min<size_type>(grainsize, n - local_offset));
  bulk::copy_n(bulk::bound<grainsize>(g.this_exec), results, local_size, stage.keys + local_offset);
  g.wait();
  
  // store merged keys to the result
  keys_result = bulk::copy_n(g, stage.keys, n, keys_result);
  
  // each agent copies the indices into the stage
  bulk::copy_n(bulk::bound<grainsize>(g.this_exec), indices, local_size, stage.indices + local_offset);
  g.wait();
  
  // gather values into merged order
  values_result = bulk::gather(g,
                               stage.indices, stage.indices + n,
                               thrust::detail::make_join_iterator(values_first1, n1, values_first2),
                               values_result);

#if __CUDA_ARCH__ >= 200
  bulk::free(g, stage.keys);
#endif

  return thrust::make_pair(keys_result, values_result);
} // end merge_by_key()


} // end bulk
BULK_NAMESPACE_SUFFIX

