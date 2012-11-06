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

#pragma once

#include <thrust/detail/config.h>
#include <thrust/iterator/iterator_traits.h>

namespace thrust
{
namespace system
{
namespace cuda
{
namespace detail
{
namespace block
{

template<typename Context,
         typename InputIterator,
         typename BinaryFunction>
__device__ __thrust_forceinline__
void inclusive_scan(Context context,
                    InputIterator first,
                    BinaryFunction binary_op)
{
  // TODO generalize to arbitrary n
  // TODO support dynamic block_size
  const unsigned int block_size = Context::ThreadsPerBlock::value;

  typename thrust::iterator_value<InputIterator>::type val = first[context.thread_index()];

  if(block_size >    1) { if (context.thread_index() >=    1) { val = binary_op(first[context.thread_index() -    1], val); } context.barrier(); first[context.thread_index()] = val; context.barrier(); }
  if(block_size >    2) { if (context.thread_index() >=    2) { val = binary_op(first[context.thread_index() -    2], val); } context.barrier(); first[context.thread_index()] = val; context.barrier(); } 
  if(block_size >    4) { if (context.thread_index() >=    4) { val = binary_op(first[context.thread_index() -    4], val); } context.barrier(); first[context.thread_index()] = val; context.barrier(); }
  if(block_size >    8) { if (context.thread_index() >=    8) { val = binary_op(first[context.thread_index() -    8], val); } context.barrier(); first[context.thread_index()] = val; context.barrier(); }
  if(block_size >   16) { if (context.thread_index() >=   16) { val = binary_op(first[context.thread_index() -   16], val); } context.barrier(); first[context.thread_index()] = val; context.barrier(); }
  if(block_size >   32) { if (context.thread_index() >=   32) { val = binary_op(first[context.thread_index() -   32], val); } context.barrier(); first[context.thread_index()] = val; context.barrier(); }
  if(block_size >   64) { if (context.thread_index() >=   64) { val = binary_op(first[context.thread_index() -   64], val); } context.barrier(); first[context.thread_index()] = val; context.barrier(); }
  if(block_size >  128) { if (context.thread_index() >=  128) { val = binary_op(first[context.thread_index() -  128], val); } context.barrier(); first[context.thread_index()] = val; context.barrier(); }
  if(block_size >  256) { if (context.thread_index() >=  256) { val = binary_op(first[context.thread_index() -  256], val); } context.barrier(); first[context.thread_index()] = val; context.barrier(); }
  if(block_size >  512) { if (context.thread_index() >=  512) { val = binary_op(first[context.thread_index() -  512], val); } context.barrier(); first[context.thread_index()] = val; context.barrier(); }
  if(block_size > 1024) { if (context.thread_index() >= 1024) { val = binary_op(first[context.thread_index() - 1024], val); } context.barrier(); first[context.thread_index()] = val; context.barrier(); }
} // end inclusive_scan()


template<typename Context,
         typename InputIterator,
         typename Size,
         typename BinaryFunction>
__device__ __thrust_forceinline__
void inclusive_scan_n(Context context,
                      InputIterator first,
                      Size n,
                      BinaryFunction binary_op)
{
  // TODO support n > context.block_dimension()
  typename thrust::iterator_value<InputIterator>::type val = first[context.thread_index()];

  for (unsigned int i = 1; i < n; i <<= 1)
  {
    if (context.thread_index() < n && context.thread_index() >= i)
      val = binary_op(first[context.thread_index() - i], val);

    context.barrier();
    
    first[context.thread_index()] = val;
    
    context.barrier();
  }
} // end inclusive_scan()


template<typename Context,
         typename InputIterator1,
         typename InputIterator2,
         typename BinaryFunction>
__device__ __thrust_forceinline__
void inclusive_scan_by_flag(Context context,
                            InputIterator1 first1,
                            InputIterator2 first2,
                            BinaryFunction binary_op)
{
  // TODO generalize to arbitrary n
  // TODO support dynamic block_size
  const unsigned int block_size = Context::ThreadsPerBlock::value;

  typename thrust::iterator_value<InputIterator1>::type flg = first1[context.thread_index()];
  typename thrust::iterator_value<InputIterator2>::type val = first2[context.thread_index()];

  if(block_size >    1) { if (context.thread_index() >=    1) { if (!flg) { flg |= first1[context.thread_index() -    1]; val = binary_op(first2[context.thread_index() -    1], val); } } context.barrier(); first1[context.thread_index()] = flg; first2[context.thread_index()] = val; context.barrier(); }
  if(block_size >    2) { if (context.thread_index() >=    2) { if (!flg) { flg |= first1[context.thread_index() -    2]; val = binary_op(first2[context.thread_index() -    2], val); } } context.barrier(); first1[context.thread_index()] = flg; first2[context.thread_index()] = val; context.barrier(); } 
  if(block_size >    4) { if (context.thread_index() >=    4) { if (!flg) { flg |= first1[context.thread_index() -    4]; val = binary_op(first2[context.thread_index() -    4], val); } } context.barrier(); first1[context.thread_index()] = flg; first2[context.thread_index()] = val; context.barrier(); }
  if(block_size >    8) { if (context.thread_index() >=    8) { if (!flg) { flg |= first1[context.thread_index() -    8]; val = binary_op(first2[context.thread_index() -    8], val); } } context.barrier(); first1[context.thread_index()] = flg; first2[context.thread_index()] = val; context.barrier(); }
  if(block_size >   16) { if (context.thread_index() >=   16) { if (!flg) { flg |= first1[context.thread_index() -   16]; val = binary_op(first2[context.thread_index() -   16], val); } } context.barrier(); first1[context.thread_index()] = flg; first2[context.thread_index()] = val; context.barrier(); }
  if(block_size >   32) { if (context.thread_index() >=   32) { if (!flg) { flg |= first1[context.thread_index() -   32]; val = binary_op(first2[context.thread_index() -   32], val); } } context.barrier(); first1[context.thread_index()] = flg; first2[context.thread_index()] = val; context.barrier(); }
  if(block_size >   64) { if (context.thread_index() >=   64) { if (!flg) { flg |= first1[context.thread_index() -   64]; val = binary_op(first2[context.thread_index() -   64], val); } } context.barrier(); first1[context.thread_index()] = flg; first2[context.thread_index()] = val; context.barrier(); }
  if(block_size >  128) { if (context.thread_index() >=  128) { if (!flg) { flg |= first1[context.thread_index() -  128]; val = binary_op(first2[context.thread_index() -  128], val); } } context.barrier(); first1[context.thread_index()] = flg; first2[context.thread_index()] = val; context.barrier(); }
  if(block_size >  256) { if (context.thread_index() >=  256) { if (!flg) { flg |= first1[context.thread_index() -  256]; val = binary_op(first2[context.thread_index() -  256], val); } } context.barrier(); first1[context.thread_index()] = flg; first2[context.thread_index()] = val; context.barrier(); }
  if(block_size >  512) { if (context.thread_index() >=  512) { if (!flg) { flg |= first1[context.thread_index() -  512]; val = binary_op(first2[context.thread_index() -  512], val); } } context.barrier(); first1[context.thread_index()] = flg; first2[context.thread_index()] = val; context.barrier(); }
  if(block_size > 1024) { if (context.thread_index() >= 1024) { if (!flg) { flg |= first1[context.thread_index() - 1024]; val = binary_op(first2[context.thread_index() - 1024], val); } } context.barrier(); first1[context.thread_index()] = flg; first2[context.thread_index()] = val; context.barrier(); }
} // end inclusive_scan_by_flag()


template<typename Context,
         typename InputIterator1,
         typename InputIterator2,
         typename Size,
         typename BinaryFunction>
__device__ __thrust_forceinline__
void inclusive_scan_by_flag_n(Context context,
                              InputIterator1 first1,
                              InputIterator2 first2,
                              Size n,
                              BinaryFunction binary_op)
{
  // TODO support n > context.block_dimension()
  typename thrust::iterator_value<InputIterator1>::type flg = first1[context.thread_index()];
  typename thrust::iterator_value<InputIterator2>::type val = first2[context.thread_index()];
  
  for (unsigned int i = 1; i < n; i <<= 1)
  {
    if (context.thread_index() < n && context.thread_index() >= i) 
    {
      if (!flg)
      { 
        flg |= first1[context.thread_index() - i];
        val  = binary_op(first2[context.thread_index() - i], val);
      }
    }

    context.barrier();
    
    first1[context.thread_index()] = flg;
    first2[context.thread_index()] = val;
    
    context.barrier();
  }
} // end inclusive_scan_by_flag()


template<typename Context, typename RandomAccessIterator, typename BinaryFunction>
__device__ __thrust_forceinline__
void inplace_inclusive_scan(Context &ctx, RandomAccessIterator first, BinaryFunction op)
{
  typename thrust::iterator_value<RandomAccessIterator>::type x = first[ctx.thread_index()];

  for(unsigned int offset = 1; offset < ctx.block_dimension(); offset *= 2)
  {
    if(ctx.thread_index() >= offset)
    {
      x = op(first[ctx.thread_index() - offset], x);
    }

    ctx.barrier();

    first[ctx.thread_index()] = x;

    ctx.barrier();
  }
}


template<typename Context, typename RandomAccessIterator>
__device__ __thrust_forceinline__
void inplace_inclusive_scan(Context &ctx, RandomAccessIterator first)
{
  block::inplace_inclusive_scan(ctx, first, thrust::plus<typename thrust::iterator_value<RandomAccessIterator>::type>());
}


} // end namespace block
} // end namespace detail
} // end namespace cuda
} // end namespace system
} // end namespace thrust

