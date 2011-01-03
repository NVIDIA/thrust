/*
 *  Copyright 2008-2011 NVIDIA Corporation
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


/*! \file safe_scan.h
 *  \brief A robust scan for general types.
 */

#pragma once

#include <thrust/detail/config.h>

// do not attempt to compile this file with any other compiler
#if THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_NVCC

#include <thrust/iterator/iterator_traits.h>

#include <thrust/detail/util/blocking.h>

#include <thrust/detail/raw_buffer.h>
#include <thrust/detail/device/dereference.h>

#include <thrust/detail/device/cuda/extern_shared_ptr.h>
#include <thrust/detail/device/cuda/synchronize.h>

// to configure launch parameters
#include <thrust/detail/device/cuda/arch.h>


__THRUST_DISABLE_MSVC_POSSIBLE_LOSS_OF_DATA_WARNING_BEGIN


namespace thrust
{
namespace detail
{

// forward declaration of raw_cuda_device_buffer
template<typename> class raw_cuda_device_buffer;

namespace device
{
namespace cuda
{
namespace detail
{
namespace safe_scan
{


template <typename SharedArray,
          typename T,
          typename BinaryFunction>
          __device__
T scan_block(SharedArray array, T val, BinaryFunction binary_op)
{
    array[threadIdx.x] = val;

    __syncthreads();

    // copy to temporary so val and tmp have the same memory space
    if (blockDim.x >   1) { if(threadIdx.x >=   1) { T tmp = array[threadIdx.x -   1]; val = binary_op(tmp, val); } __syncthreads(); array[threadIdx.x] = val; __syncthreads(); }
    if (blockDim.x >   2) { if(threadIdx.x >=   2) { T tmp = array[threadIdx.x -   2]; val = binary_op(tmp, val); } __syncthreads(); array[threadIdx.x] = val; __syncthreads(); }
    if (blockDim.x >   4) { if(threadIdx.x >=   4) { T tmp = array[threadIdx.x -   4]; val = binary_op(tmp, val); } __syncthreads(); array[threadIdx.x] = val; __syncthreads(); }
    if (blockDim.x >   8) { if(threadIdx.x >=   8) { T tmp = array[threadIdx.x -   8]; val = binary_op(tmp, val); } __syncthreads(); array[threadIdx.x] = val; __syncthreads(); }
    if (blockDim.x >  16) { if(threadIdx.x >=  16) { T tmp = array[threadIdx.x -  16]; val = binary_op(tmp, val); } __syncthreads(); array[threadIdx.x] = val; __syncthreads(); }
    if (blockDim.x >  32) { if(threadIdx.x >=  32) { T tmp = array[threadIdx.x -  32]; val = binary_op(tmp, val); } __syncthreads(); array[threadIdx.x] = val; __syncthreads(); }
    if (blockDim.x >  64) { if(threadIdx.x >=  64) { T tmp = array[threadIdx.x -  64]; val = binary_op(tmp, val); } __syncthreads(); array[threadIdx.x] = val; __syncthreads(); }
    if (blockDim.x > 128) { if(threadIdx.x >= 128) { T tmp = array[threadIdx.x - 128]; val = binary_op(tmp, val); } __syncthreads(); array[threadIdx.x] = val; __syncthreads(); }
    if (blockDim.x > 256) { if(threadIdx.x >= 256) { T tmp = array[threadIdx.x - 256]; val = binary_op(tmp, val); } __syncthreads(); array[threadIdx.x] = val; __syncthreads(); }  
    if (blockDim.x > 512) { if(threadIdx.x >= 512) { T tmp = array[threadIdx.x - 512]; val = binary_op(tmp, val); } __syncthreads(); array[threadIdx.x] = val; __syncthreads(); }  

    return val;
}

template <typename SharedArray,
          typename T,
          typename BinaryFunction>
          __device__
T scan_block_n(SharedArray array, const unsigned int n, T val, BinaryFunction binary_op)
{
    array[threadIdx.x] = val;

    __syncthreads();

    if (blockDim.x >   1) { if(threadIdx.x < n && threadIdx.x >=   1) { T tmp = array[threadIdx.x -   1]; val = binary_op(tmp, val); } __syncthreads(); array[threadIdx.x] = val; __syncthreads(); }
    if (blockDim.x >   2) { if(threadIdx.x < n && threadIdx.x >=   2) { T tmp = array[threadIdx.x -   2]; val = binary_op(tmp, val); } __syncthreads(); array[threadIdx.x] = val; __syncthreads(); }
    if (blockDim.x >   4) { if(threadIdx.x < n && threadIdx.x >=   4) { T tmp = array[threadIdx.x -   4]; val = binary_op(tmp, val); } __syncthreads(); array[threadIdx.x] = val; __syncthreads(); }
    if (blockDim.x >   8) { if(threadIdx.x < n && threadIdx.x >=   8) { T tmp = array[threadIdx.x -   8]; val = binary_op(tmp, val); } __syncthreads(); array[threadIdx.x] = val; __syncthreads(); }
    if (blockDim.x >  16) { if(threadIdx.x < n && threadIdx.x >=  16) { T tmp = array[threadIdx.x -  16]; val = binary_op(tmp, val); } __syncthreads(); array[threadIdx.x] = val; __syncthreads(); }
    if (blockDim.x >  32) { if(threadIdx.x < n && threadIdx.x >=  32) { T tmp = array[threadIdx.x -  32]; val = binary_op(tmp, val); } __syncthreads(); array[threadIdx.x] = val; __syncthreads(); }
    if (blockDim.x >  64) { if(threadIdx.x < n && threadIdx.x >=  64) { T tmp = array[threadIdx.x -  64]; val = binary_op(tmp, val); } __syncthreads(); array[threadIdx.x] = val; __syncthreads(); }
    if (blockDim.x > 128) { if(threadIdx.x < n && threadIdx.x >= 128) { T tmp = array[threadIdx.x - 128]; val = binary_op(tmp, val); } __syncthreads(); array[threadIdx.x] = val; __syncthreads(); }
    if (blockDim.x > 256) { if(threadIdx.x < n && threadIdx.x >= 256) { T tmp = array[threadIdx.x - 256]; val = binary_op(tmp, val); } __syncthreads(); array[threadIdx.x] = val; __syncthreads(); }
    if (blockDim.x > 512) { if(threadIdx.x < n && threadIdx.x >= 512) { T tmp = array[threadIdx.x - 512]; val = binary_op(tmp, val); } __syncthreads(); array[threadIdx.x] = val; __syncthreads(); }

    return val;
}

template <typename InputIterator,
          typename OutputIterator,
          typename BinaryFunction>
__global__
void scan_intervals(InputIterator input,
                    const unsigned int N,
                    const unsigned int interval_size,
                    OutputIterator output,
                    typename thrust::iterator_value<OutputIterator>::type * block_results,
                    BinaryFunction binary_op)
{
    typedef typename thrust::iterator_value<OutputIterator>::type OutputType;

    thrust::detail::device::cuda::extern_shared_ptr<OutputType> sdata;
    
    const unsigned int interval_begin = interval_size * blockIdx.x;
    const unsigned int interval_end   = min(interval_begin + interval_size, N);

    unsigned int base = interval_begin;

    OutputType val;

    // process full blocks
    for(; base + blockDim.x <= interval_end; base += blockDim.x)
    {
        // read data
        {
            InputIterator temp = input + (base + threadIdx.x);
            val = thrust::detail::device::dereference(temp);
        }
       
        // carry in
        if (threadIdx.x == 0 && base != interval_begin)
        {
            OutputType tmp = sdata[blockDim.x - 1];
            val = binary_op(tmp, val);
        }

        __syncthreads();

        // scan block
        val = scan_block(sdata, val, binary_op);
       
        // write data
        {
            OutputIterator temp = output + (base + threadIdx.x);
            thrust::detail::device::dereference(temp) = val;
        }   
    }

    // process partially full block at end of input (if necessary)
    if (base < interval_end)
    {
        // read data
        if (base + threadIdx.x < interval_end)
        {
            InputIterator temp = input + (base + threadIdx.x);
            val = thrust::detail::device::dereference(temp);
        }
       
        // carry in
        if (threadIdx.x == 0 && base != interval_begin)
        {
            OutputType tmp = sdata[blockDim.x - 1];
            val = binary_op(tmp, val);
        }
        __syncthreads();

        // scan block
        val = scan_block_n(sdata, interval_end - base, val, binary_op);
       
        // write data
        if (base + threadIdx.x < interval_end)
        {
            OutputIterator temp = output + (base + threadIdx.x);
            thrust::detail::device::dereference(temp) = val;
        }   
    }

    __syncthreads();
    
    // write interval sum
    if (threadIdx.x == 0)
    {
        OutputIterator temp = output + (interval_end - 1);
        block_results[blockIdx.x] = thrust::detail::device::dereference(temp);
    }
}


template <typename OutputIterator,
          typename OutputType,
          typename BinaryFunction>
__global__
void inclusive_update(OutputIterator output,
                      const unsigned int N,
                      const unsigned int interval_size,
                      OutputType *   block_results,
                      BinaryFunction binary_op)
{
    const unsigned int interval_begin = interval_size * blockIdx.x;
    const unsigned int interval_end   = min(interval_begin + interval_size, N);

    if (blockIdx.x == 0)
        return;

    // value to add to this segment 
    OutputType sum = block_results[blockIdx.x - 1];
    
    // advance result iterator
    output += interval_begin + threadIdx.x;
    
    for(unsigned int base = interval_begin; base < interval_end; base += blockDim.x, output += blockDim.x)
    {
        const unsigned int i = base + threadIdx.x;

        if(i < interval_end)
        {
            OutputType tmp = thrust::detail::device::dereference(output);
            thrust::detail::device::dereference(output) = binary_op(sum, tmp);
        }

        __syncthreads();
    }
}

template <typename OutputIterator,
          typename OutputType,
          typename BinaryFunction>
__global__
void exclusive_update(OutputIterator output,
                      const unsigned int N,
                      const unsigned int interval_size,
                      OutputType * block_results,
                      BinaryFunction binary_op)
{
    thrust::detail::device::cuda::extern_shared_ptr<OutputType> sdata;

    const unsigned int interval_begin = interval_size * blockIdx.x;
    const unsigned int interval_end   = min(interval_begin + interval_size, N);

    // value to add to this segment 
    OutputType carry = block_results[gridDim.x]; // init
    if (blockIdx.x != 0)
    {
        OutputType tmp = block_results[blockIdx.x - 1];
        carry = binary_op(carry, tmp);
    }

    OutputType val = carry;

    // advance result iterator
    output += interval_begin + threadIdx.x;

    for(unsigned int base = interval_begin; base < interval_end; base += blockDim.x, output += blockDim.x)
    {
        const unsigned int i = base + threadIdx.x;

        if(i < interval_end)
        {
            OutputType tmp = thrust::detail::device::dereference(output);
            sdata[threadIdx.x] = binary_op(carry, tmp);
        }
        __syncthreads();

        if (threadIdx.x != 0)
            val = sdata[threadIdx.x - 1];

        if (i < interval_end)
            thrust::detail::device::dereference(output) = val;

        if(threadIdx.x == 0)
            val = sdata[blockDim.x - 1];
        
        __syncthreads();
    }
}


template <typename InputIterator,
          typename OutputIterator,
          typename BinaryFunction>
OutputIterator inclusive_scan(InputIterator first,
                              InputIterator last,
                              OutputIterator output,
                              BinaryFunction binary_op)
{
    if (first == last)
        return output;

    typedef typename thrust::iterator_value<OutputIterator>::type OutputType;

    const unsigned int N = last - first;
    
    // determine maximal launch parameters
    const unsigned int smem_per_thread = sizeof(OutputType);
    const unsigned int block_size = thrust::detail::device::cuda::arch::max_blocksize_with_highest_occupancy(scan_intervals<InputIterator,OutputIterator,BinaryFunction>, smem_per_thread);
    const unsigned int smem_size  = block_size * smem_per_thread;
    const unsigned int max_blocks = thrust::detail::device::cuda::arch::max_active_blocks(scan_intervals<InputIterator,OutputIterator,BinaryFunction>, block_size, smem_size);

    // determine final launch parameters
    const unsigned int unit_size     = block_size;
    const unsigned int num_units     = thrust::detail::util::divide_ri(N, unit_size);
    const unsigned int num_blocks    = (std::min)(max_blocks, num_units);
    const unsigned int num_iters     = thrust::detail::util::divide_ri(num_units, num_blocks);
    const unsigned int interval_size = unit_size * num_iters;
    
    //std::cout << "N             " << N << std::endl;
    //std::cout << "max_blocks    " << max_blocks    << std::endl;
    //std::cout << "unit_size     " << unit_size     << std::endl;
    //std::cout << "num_blocks    " << num_blocks    << std::endl;
    //std::cout << "num_iters     " << num_iters     << std::endl;
    //std::cout << "interval_size " << interval_size << std::endl;

    thrust::detail::raw_cuda_device_buffer<OutputType> block_results(num_blocks + 1);
                
    // first level scan of interval (one interval per block)
    {
        scan_intervals<<<num_blocks, block_size, smem_size>>>
            (first,
             N,
             interval_size,
             output,
             thrust::raw_pointer_cast(&block_results[0]),
             binary_op);
        synchronize_if_enabled("scan_intervals");
    }
  
    // second level inclusive scan of per-block results
    {
        const unsigned int block_size_pass2 = thrust::detail::device::cuda::arch::max_blocksize(scan_intervals<OutputType *, OutputType *, BinaryFunction>, smem_per_thread);
        const unsigned int smem_size_pass2  = smem_per_thread * block_size_pass2;

        scan_intervals<<<         1, block_size_pass2, smem_size_pass2>>>
            (thrust::raw_pointer_cast(&block_results[0]),
             num_blocks,
             interval_size,
             thrust::raw_pointer_cast(&block_results[0]),
             thrust::raw_pointer_cast(&block_results[0]) + num_blocks,
             binary_op);
        synchronize_if_enabled("scan_intervals");
    }
   
    // update intervals with result of second level scan
    {
        const unsigned int block_size_pass3 = thrust::detail::device::cuda::arch::max_blocksize_with_highest_occupancy(inclusive_update<OutputIterator,OutputType,BinaryFunction>, 0);

        inclusive_update<<<num_blocks, block_size_pass3>>>
            (output,
             N,
             interval_size,
             thrust::raw_pointer_cast(&block_results[0]),
             binary_op);
        synchronize_if_enabled("inclusive_update");
    }

    return output + N;
}


template <typename InputIterator,
          typename OutputIterator,
          typename T,
          typename BinaryFunction>
OutputIterator exclusive_scan(InputIterator first,
                              InputIterator last,
                              OutputIterator output,
                              const T init,
                              BinaryFunction binary_op)
{
    if (first == last)
        return output;

    typedef typename thrust::iterator_value<OutputIterator>::type OutputType;

    const unsigned int N = last - first;
    
    // determine maximal launch parameters
    const unsigned int smem_per_thread = sizeof(OutputType);
    const unsigned int block_size = thrust::detail::device::cuda::arch::max_blocksize_with_highest_occupancy(scan_intervals<InputIterator,OutputIterator,BinaryFunction>, smem_per_thread);
    const unsigned int smem_size  = block_size * smem_per_thread;
    const unsigned int max_blocks = thrust::detail::device::cuda::arch::max_active_blocks(scan_intervals<InputIterator,OutputIterator,BinaryFunction>, block_size, smem_size);

    // determine final launch parameters
    const unsigned int unit_size     = block_size;
    const unsigned int num_units     = thrust::detail::util::divide_ri(N, unit_size);
    const unsigned int num_blocks    = (std::min)(max_blocks, num_units);
    const unsigned int num_iters     = thrust::detail::util::divide_ri(num_units, num_blocks);
    const unsigned int interval_size = unit_size * num_iters;
    
    //std::cout << "N             " << N << std::endl;
    //std::cout << "max_blocks    " << max_blocks    << std::endl;
    //std::cout << "unit_size     " << unit_size     << std::endl;
    //std::cout << "num_blocks    " << num_blocks    << std::endl;
    //std::cout << "num_iters     " << num_iters     << std::endl;
    //std::cout << "interval_size " << interval_size << std::endl;

    thrust::detail::raw_cuda_device_buffer<OutputType> block_results(num_blocks + 1);
                
    // first level scan of interval (one interval per block)
    {
        scan_intervals<<<num_blocks, block_size, smem_size>>>
            (first,
             N,
             interval_size,
             output,
             thrust::raw_pointer_cast(&block_results[0]),
             binary_op);
        synchronize_if_enabled("scan_intervals");
    }
        
    
    // second level inclusive scan of per-block results
    {
        const unsigned int block_size_pass2 = thrust::detail::device::cuda::arch::max_blocksize(scan_intervals<OutputType *, OutputType *, BinaryFunction>, smem_per_thread);
        const unsigned int smem_size_pass2  = smem_per_thread * block_size_pass2;

        scan_intervals<<<         1, block_size_pass2, smem_size_pass2>>>
            (thrust::raw_pointer_cast(&block_results[0]),
             num_blocks,
             interval_size,
             thrust::raw_pointer_cast(&block_results[0]),
             thrust::raw_pointer_cast(&block_results[0]) + num_blocks,
             binary_op);
        synchronize_if_enabled("scan_intervals");
    }

    // copy the initial value to the device
    block_results[num_blocks] = init;

    // update intervals with result of second level scan
    {
        const unsigned int block_size_pass3 = thrust::detail::device::cuda::arch::max_blocksize_with_highest_occupancy(exclusive_update<OutputIterator,OutputType,BinaryFunction>, smem_per_thread);
        const unsigned int smem_size_pass3  = smem_per_thread * block_size_pass3;

        exclusive_update<<<num_blocks, block_size_pass3, smem_size_pass3>>>
            (output,
             N,
             interval_size,
             thrust::raw_pointer_cast(&block_results[0]),
             binary_op);
        synchronize_if_enabled("exclusive_update");
    }

    return output + N;
}

} // end namespace safe_scan
} // end namespace detail
} // end namespace cuda
} // end namespace device
} // end namespace detail
} // end namespace thrust

__THRUST_DISABLE_MSVC_POSSIBLE_LOSS_OF_DATA_WARNING_END

#endif // THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_NVCC

