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


// do not attempt to compile this file with any other compiler
#if THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_NVCC

#include <thrust/iterator/iterator_traits.h>

#include <thrust/detail/util/blocking.h>

#include <thrust/detail/raw_buffer.h>
#include <thrust/detail/device/dereference.h>
#include <thrust/detail/device/cuda/synchronize.h>

// to configure launch parameters
#include <thrust/detail/device/cuda/arch.h>

#include <thrust/detail/device/cuda/partition.h>

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
namespace fast_scan
{

template <unsigned int CTA_SIZE,
          typename SharedArray,
          typename BinaryFunction>
          __device__
void scan_block(SharedArray array, BinaryFunction binary_op)
{
    typedef typename thrust::iterator_value<SharedArray>::type T;

    T val = array[threadIdx.x];

    if (CTA_SIZE >   1) { if(threadIdx.x >=   1) { T tmp = array[threadIdx.x -   1]; val = binary_op(tmp, val); } __syncthreads(); array[threadIdx.x] = val; __syncthreads(); }
    if (CTA_SIZE >   2) { if(threadIdx.x >=   2) { T tmp = array[threadIdx.x -   2]; val = binary_op(tmp, val); } __syncthreads(); array[threadIdx.x] = val; __syncthreads(); }
    if (CTA_SIZE >   4) { if(threadIdx.x >=   4) { T tmp = array[threadIdx.x -   4]; val = binary_op(tmp, val); } __syncthreads(); array[threadIdx.x] = val; __syncthreads(); }
    if (CTA_SIZE >   8) { if(threadIdx.x >=   8) { T tmp = array[threadIdx.x -   8]; val = binary_op(tmp, val); } __syncthreads(); array[threadIdx.x] = val; __syncthreads(); }
    if (CTA_SIZE >  16) { if(threadIdx.x >=  16) { T tmp = array[threadIdx.x -  16]; val = binary_op(tmp, val); } __syncthreads(); array[threadIdx.x] = val; __syncthreads(); }
    if (CTA_SIZE >  32) { if(threadIdx.x >=  32) { T tmp = array[threadIdx.x -  32]; val = binary_op(tmp, val); } __syncthreads(); array[threadIdx.x] = val; __syncthreads(); }
    if (CTA_SIZE >  64) { if(threadIdx.x >=  64) { T tmp = array[threadIdx.x -  64]; val = binary_op(tmp, val); } __syncthreads(); array[threadIdx.x] = val; __syncthreads(); }
    if (CTA_SIZE > 128) { if(threadIdx.x >= 128) { T tmp = array[threadIdx.x - 128]; val = binary_op(tmp, val); } __syncthreads(); array[threadIdx.x] = val; __syncthreads(); }
    if (CTA_SIZE > 256) { if(threadIdx.x >= 256) { T tmp = array[threadIdx.x - 256]; val = binary_op(tmp, val); } __syncthreads(); array[threadIdx.x] = val; __syncthreads(); }  
    if (CTA_SIZE > 512) { if(threadIdx.x >= 512) { T tmp = array[threadIdx.x - 512]; val = binary_op(tmp, val); } __syncthreads(); array[threadIdx.x] = val; __syncthreads(); }  
}

template <unsigned int CTA_SIZE,
          typename SharedArray,
          typename BinaryFunction>
          __device__
void scan_block_n(SharedArray array, const unsigned int n, BinaryFunction binary_op)
{
    typedef typename thrust::iterator_value<SharedArray>::type T;

    T val = array[threadIdx.x];

    if (CTA_SIZE >   1) { if(threadIdx.x < n && threadIdx.x >=   1) { T tmp = array[threadIdx.x -   1]; val = binary_op(tmp, val); } __syncthreads(); array[threadIdx.x] = val; __syncthreads(); }
    if (CTA_SIZE >   2) { if(threadIdx.x < n && threadIdx.x >=   2) { T tmp = array[threadIdx.x -   2]; val = binary_op(tmp, val); } __syncthreads(); array[threadIdx.x] = val; __syncthreads(); }
    if (CTA_SIZE >   4) { if(threadIdx.x < n && threadIdx.x >=   4) { T tmp = array[threadIdx.x -   4]; val = binary_op(tmp, val); } __syncthreads(); array[threadIdx.x] = val; __syncthreads(); }
    if (CTA_SIZE >   8) { if(threadIdx.x < n && threadIdx.x >=   8) { T tmp = array[threadIdx.x -   8]; val = binary_op(tmp, val); } __syncthreads(); array[threadIdx.x] = val; __syncthreads(); }
    if (CTA_SIZE >  16) { if(threadIdx.x < n && threadIdx.x >=  16) { T tmp = array[threadIdx.x -  16]; val = binary_op(tmp, val); } __syncthreads(); array[threadIdx.x] = val; __syncthreads(); }
    if (CTA_SIZE >  32) { if(threadIdx.x < n && threadIdx.x >=  32) { T tmp = array[threadIdx.x -  32]; val = binary_op(tmp, val); } __syncthreads(); array[threadIdx.x] = val; __syncthreads(); }
    if (CTA_SIZE >  64) { if(threadIdx.x < n && threadIdx.x >=  64) { T tmp = array[threadIdx.x -  64]; val = binary_op(tmp, val); } __syncthreads(); array[threadIdx.x] = val; __syncthreads(); }
    if (CTA_SIZE > 128) { if(threadIdx.x < n && threadIdx.x >= 128) { T tmp = array[threadIdx.x - 128]; val = binary_op(tmp, val); } __syncthreads(); array[threadIdx.x] = val; __syncthreads(); }
    if (CTA_SIZE > 256) { if(threadIdx.x < n && threadIdx.x >= 256) { T tmp = array[threadIdx.x - 256]; val = binary_op(tmp, val); } __syncthreads(); array[threadIdx.x] = val; __syncthreads(); }
    if (CTA_SIZE > 512) { if(threadIdx.x < n && threadIdx.x >= 512) { T tmp = array[threadIdx.x - 512]; val = binary_op(tmp, val); } __syncthreads(); array[threadIdx.x] = val; __syncthreads(); }
}

template <unsigned int CTA_SIZE,
          unsigned int K,
          typename InputIterator,
          typename OutputIterator,
          typename BinaryFunction>
__launch_bounds__(CTA_SIZE,1)          
__global__
void scan_intervals(InputIterator input,
                    const unsigned int N,
                    const unsigned int interval_size,
                    OutputIterator output,
                    typename thrust::iterator_value<OutputIterator>::type * block_results,
                    BinaryFunction binary_op)
{
    typedef typename thrust::iterator_value<OutputIterator>::type OutputType;

    __shared__ OutputType sdata[K + 1][CTA_SIZE + 1];  // padded to avoid bank conflicts
    
    __syncthreads(); // TODO figure out why this seems necessary now
    
    const unsigned int interval_begin = interval_size * blockIdx.x;
    const unsigned int interval_end   = min(interval_begin + interval_size, N);

    const unsigned int unit_size  = K * CTA_SIZE;

    unsigned int base = interval_begin;

    // process full units
    for(; base + unit_size <= interval_end; base += unit_size)
    {
        // read data
        for(unsigned int k = 0; k < K; k++)
        {
            const unsigned int offset = k*CTA_SIZE + threadIdx.x;
                
            InputIterator temp = input + (base + offset);
            sdata[offset % K][offset / K] = thrust::detail::device::dereference(temp);
        }
       
        // carry in
        if (threadIdx.x == 0 && base != interval_begin)
        {
            //sdata[0][0] = binary_op(sdata[K][CTA_SIZE - 1], sdata[0][0]);
            //// XXX WAR sm_10 issue
            OutputType tmp1 = sdata[K][CTA_SIZE - 1];
            OutputType tmp2 = sdata[0][0];
            sdata[0][0] = binary_op(tmp1, tmp2);
        }

        __syncthreads();

        // scan local values
        OutputType sum = sdata[0][threadIdx.x];

        for(unsigned int k = 1; k < K; k++)
        {
            OutputType tmp = sdata[k][threadIdx.x];
            sum = binary_op(sum, tmp);
            sdata[k][threadIdx.x] = sum;
        }

        // second level scan
        sdata[K][threadIdx.x] = sum;  __syncthreads();
        scan_block<CTA_SIZE>(&sdata[K][0], binary_op);
       
        // update local values
        if (threadIdx.x > 0)
        {
            sum = sdata[K][threadIdx.x - 1];

            for(unsigned int k = 0; k < K; k++)
            {
                OutputType tmp = sdata[k][threadIdx.x];
                sdata[k][threadIdx.x] = binary_op(sum, tmp);
            }
        }

        __syncthreads();

        // write data
        for(unsigned int k = 0; k < K; k++)
        {
            const unsigned int offset = k*CTA_SIZE + threadIdx.x;

            OutputIterator temp = output + (base + offset);
            thrust::detail::device::dereference(temp) = sdata[offset % K][offset / K];
        }   
        
        __syncthreads();
    }

    // process partially full unit at end of input (if necessary)
    if (base < interval_end)
    {
        // read data
        for(unsigned int k = 0; k < K; k++)
        {
            const unsigned int offset = k*CTA_SIZE + threadIdx.x;

            if (base + offset < interval_end)
            {
                InputIterator temp = input + (base + offset);
                sdata[offset % K][offset / K] = thrust::detail::device::dereference(temp);
            }
        }
       
        // carry in
        if (threadIdx.x == 0 && base != interval_begin)
        {
            //sdata[0][0] = binary_op(sdata[K][CTA_SIZE - 1], sdata[0][0]);
            //// XXX WAR sm_10 issue
            OutputType tmp1 = sdata[K][CTA_SIZE - 1];
            OutputType tmp2 = sdata[0][0];
            sdata[0][0] = binary_op(tmp1, tmp2);
        }

        __syncthreads();

        // scan local values
        OutputType sum = sdata[0][threadIdx.x];

        const unsigned int offset_end = interval_end - base;

        for(unsigned int k = 1; k < K; k++)
        {
            if (K * threadIdx.x + k < offset_end)
            {
                OutputType tmp = sdata[k][threadIdx.x];
                sum = binary_op(sum, tmp);
                sdata[k][threadIdx.x] = sum;
            }
        }

        // second level scan
        sdata[K][threadIdx.x] = sum;  __syncthreads();
        scan_block_n<CTA_SIZE>(&sdata[K][0], offset_end / K, binary_op);
       
        // update local values
        if (threadIdx.x > 0)
        {
            sum = sdata[K][threadIdx.x - 1];

            for(unsigned int k = 0; k < K; k++)
            {
                if (K * threadIdx.x + k < offset_end)
                {
                    OutputType tmp = sdata[k][threadIdx.x];
                    sdata[k][threadIdx.x] = binary_op(sum, tmp);
                }
            }
        }

        __syncthreads();

        // write data
        for(unsigned int k = 0; k < K; k++)
        {
            const unsigned int offset = k*CTA_SIZE + threadIdx.x;

            if (base + offset < interval_end)
            {
                OutputIterator temp = output + (base + offset);
                thrust::detail::device::dereference(temp) = sdata[offset % K][offset / K];
            }
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


template <unsigned int CTA_SIZE,
          typename OutputIterator,
          typename OutputType,
          typename BinaryFunction>
__launch_bounds__(CTA_SIZE,1)          
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
    
    for(unsigned int base = interval_begin; base < interval_end; base += CTA_SIZE, output += CTA_SIZE)
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

template <unsigned int CTA_SIZE,
          typename OutputIterator,
          typename OutputType,
          typename BinaryFunction>
__launch_bounds__(CTA_SIZE,1)          
__global__
void exclusive_update(OutputIterator output,
                      const unsigned int N,
                      const unsigned int interval_size,
                      OutputType * block_results,
                      OutputType init,
                      BinaryFunction binary_op)
{
    __shared__ OutputType sdata[CTA_SIZE];

    __syncthreads(); // TODO figure out why this seems necessary now

    const unsigned int interval_begin = interval_size * blockIdx.x;
    const unsigned int interval_end   = min(interval_begin + interval_size, N);

    // value to add to this segment 
    OutputType carry = init;
    if(blockIdx.x != 0)
    {
        OutputType tmp = block_results[blockIdx.x - 1];
        carry = binary_op(carry, tmp);
    }

    OutputType val   = carry;

    // advance result iterator
    output += interval_begin + threadIdx.x;

    for(unsigned int base = interval_begin; base < interval_end; base += CTA_SIZE, output += CTA_SIZE)
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
            val = sdata[CTA_SIZE - 1];
        
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

    // Good parameters for GT200
    const unsigned int CTA_SIZE = 128;
    const unsigned int K        = 6;
    
    typedef typename thrust::iterator_value<OutputIterator>::type OutputType;

    const unsigned int N = last - first;
    
    const unsigned int unit_size  = CTA_SIZE * K;
    const unsigned int max_blocks = thrust::detail::device::cuda::arch::max_active_blocks(scan_intervals<CTA_SIZE,K,InputIterator,OutputIterator,BinaryFunction>, CTA_SIZE, 0);
    
    thrust::pair<unsigned int, unsigned int> splitting = uniform_interval_splitting<unsigned int>(N, unit_size, max_blocks);
    const unsigned int interval_size = splitting.first;
    const unsigned int num_blocks    = splitting.second;

    //std::cout << "N             " << N << std::endl;
    //std::cout << "max_blocks    " << max_blocks    << std::endl;
    //std::cout << "unit_size     " << unit_size     << std::endl;
    //std::cout << "num_blocks    " << num_blocks    << std::endl;
    //std::cout << "num_iters     " << num_iters     << std::endl;
    //std::cout << "interval_size " << interval_size << std::endl;
    
    thrust::detail::raw_cuda_device_buffer<OutputType> block_results(num_blocks + 1);
    
    // first level scan of interval (one interval per block)
    scan_intervals<CTA_SIZE,K> <<<num_blocks, CTA_SIZE>>>
        (first,
         N,
         interval_size,
         output,
         thrust::raw_pointer_cast(&block_results[0]),
         binary_op);
    synchronize_if_enabled("scan_intervals");
    
    // second level inclusive scan of per-block results
    scan_intervals<CTA_SIZE,K> <<<         1, CTA_SIZE>>>
        (thrust::raw_pointer_cast(&block_results[0]),
         num_blocks,
         interval_size,
         thrust::raw_pointer_cast(&block_results[0]),
         thrust::raw_pointer_cast(&block_results[0]) + num_blocks,
         binary_op);
    synchronize_if_enabled("scan_intervals");
    
    // update intervals with result of second level scan
    inclusive_update<256> <<<num_blocks, 256>>>
        (output,
         N,
         interval_size,
         thrust::raw_pointer_cast(&block_results[0]),
         binary_op);
    synchronize_if_enabled("inclusive_update");
    
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

    // Good parameters for GT200
    const unsigned int CTA_SIZE = 128;
    const unsigned int K        = 6;
    
    typedef typename thrust::iterator_value<OutputIterator>::type OutputType;

    const unsigned int N = last - first;

    const unsigned int unit_size  = CTA_SIZE * K;
    const unsigned int max_blocks = thrust::detail::device::cuda::arch::max_active_blocks(scan_intervals<CTA_SIZE,K,InputIterator,OutputIterator,BinaryFunction>, CTA_SIZE, 0);
    
    thrust::pair<unsigned int, unsigned int> splitting = uniform_interval_splitting<unsigned int>(N, unit_size, max_blocks);
    const unsigned int interval_size = splitting.first;
    const unsigned int num_blocks    = splitting.second;

    //std::cout << "N             " << N << std::endl;
    //std::cout << "max_blocks    " << max_blocks    << std::endl;
    //std::cout << "unit_size     " << unit_size     << std::endl;
    //std::cout << "num_blocks    " << num_blocks    << std::endl;
    //std::cout << "num_iters     " << num_iters     << std::endl;
    //std::cout << "interval_size " << interval_size << std::endl;

    thrust::detail::raw_cuda_device_buffer<OutputType> block_results(num_blocks + 1);
                
    // first level scan of interval (one interval per block)
    scan_intervals<CTA_SIZE,K> <<<num_blocks, CTA_SIZE>>>
        (first,
         N,
         interval_size,
         output,
         thrust::raw_pointer_cast(&block_results[0]),
         binary_op);
    synchronize_if_enabled("scan_intervals");
    
    // second level inclusive scan of per-block results
    scan_intervals<CTA_SIZE,K> <<<         1, CTA_SIZE>>>
        (thrust::raw_pointer_cast(&block_results[0]),
         num_blocks,
         interval_size,
         thrust::raw_pointer_cast(&block_results[0]),
         thrust::raw_pointer_cast(&block_results[0]) + num_blocks,
         binary_op);
    synchronize_if_enabled("scan_intervals");

    // update intervals with result of second level scan
    exclusive_update<256> <<<num_blocks, 256>>>
        (output,
         N,
         interval_size,
         thrust::raw_pointer_cast(&block_results[0]),
         OutputType(init),
         binary_op);
    synchronize_if_enabled("exclusive_update");
    
    return output + N;
}

} // end namespace fast_scan
} // end namespace detail
} // end namespace cuda
} // end namespace device
} // end namespace detail
} // end namespace thrust


__THRUST_DISABLE_MSVC_POSSIBLE_LOSS_OF_DATA_WARNING_END

#endif // THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_NVCC

