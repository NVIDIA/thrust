/*
 *  Copyright 2008-2009 NVIDIA Corporation
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


/*! \file vectorize.inl
 *  \brief Inline file for vectorize.h.
 */

// do not attempt to compile this with any other compiler
#ifdef __CUDACC__

#include <thrust/experimental/arch.h>
#include <thrust/detail/device/cuda/malloc.h>
#include <thrust/detail/device/cuda/free.h>

namespace thrust
{

namespace detail
{

namespace device
{

namespace cuda
{


template<typename IndexType,
         typename UnaryFunction>
__global__             
void vectorize_from_shared_kernel(IndexType n, UnaryFunction f)
{
    const IndexType grid_size = blockDim.x * gridDim.x;
    
    for(IndexType i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += grid_size)
        f(i);
}

template<typename IndexType,
         typename UnaryFunction>
__global__
void vectorize_from_global_kernel(IndexType n, UnaryFunction *f_ptr)
{
    // load f into registers
    UnaryFunction f = *f_ptr;

    const IndexType grid_size = blockDim.x * gridDim.x;
    
    for(IndexType i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += grid_size)
        f(i);
}

template<bool> struct launch_vectorize {};

// this path loads f into parameters as normal
template<> struct launch_vectorize<true>
{
    template<typename IndexType, typename UnaryFunctor>
    void operator()(const size_t NUM_BLOCKS, const size_t BLOCK_SIZE, IndexType n, UnaryFunctor f)
    {
        vectorize_from_shared_kernel<<<NUM_BLOCKS,BLOCK_SIZE>>>(n, f);
    }
};

// this path loads f into gmem
template<> struct launch_vectorize<false>
{
    template<typename IndexType, typename UnaryFunctor>
    void operator()(const size_t NUM_BLOCKS, const size_t BLOCK_SIZE, IndexType n, UnaryFunctor f)
    {
        // allocate device memory for the functor
        thrust::device_ptr<void> temp_ptr = thrust::detail::device::cuda::malloc(sizeof(UnaryFunctor));

        // cast to UnaryFunctor *
        thrust::device_ptr<UnaryFunctor> f_ptr(reinterpret_cast<UnaryFunctor*>(temp_ptr.get()));

        // copy
        *f_ptr = f;

        // launch
        vectorize_from_global_kernel<<<NUM_BLOCKS,BLOCK_SIZE>>>(n, f_ptr.get());

        // free device memory
        thrust::detail::device::cuda::free(f_ptr);
    }
};

template<typename IndexType,
         typename UnaryFunction>
void vectorize(IndexType n, UnaryFunction f)
{
    if (n == 0)
        return;

    const size_t BLOCK_SIZE = 256;
    const size_t MAX_BLOCKS = 3 * thrust::experimental::arch::max_active_threads()/BLOCK_SIZE;
    const size_t NUM_BLOCKS = std::min<size_t>(MAX_BLOCKS, (n + (BLOCK_SIZE - 1) ) / BLOCK_SIZE);

    // B.1.4 in NVIDIA Programming Guide v2.2
    // XXX perhaps publish this somewhere
    const size_t MAX_GLOBAL_PARM_SIZE = 256;

    // can we fit the parameters into smem?
    const bool use_shared = (sizeof(IndexType) + sizeof(UnaryFunction)) <= MAX_GLOBAL_PARM_SIZE;

    launch_vectorize<use_shared> launch;
    launch(NUM_BLOCKS, BLOCK_SIZE, n, f);
}



template<typename RandomAccessDeviceIterator>
  __global__ void vectorize_kernel(RandomAccessDeviceIterator first, RandomAccessDeviceIterator last)
{
  // this kernel simply dereferences each iterator i in [first, last)
  
  typedef typename thrust::iterator_traits<RandomAccessDeviceIterator>::difference_type difference_type;

  difference_type grid_size = blockDim.x * gridDim.x;

  for(first += blockIdx.x * blockDim.x + threadIdx.x;
      first < last;
      first += grid_size)
  {
    *first;
  }
}

template<typename RandomAccessDeviceIterator>
  void vectorize(RandomAccessDeviceIterator first, RandomAccessDeviceIterator last)
{
  if(first >= last) return;

  typedef typename thrust::iterator_traits<RandomAccessDeviceIterator>::difference_type difference_type;
  difference_type n = last - first;

  const size_t BLOCK_SIZE = 256;
  const size_t MAX_BLOCKS = 3 * thrust::experimental::arch::max_active_threads()/BLOCK_SIZE;
  const size_t NUM_BLOCKS = std::min(MAX_BLOCKS, (n + (BLOCK_SIZE - 1) ) / BLOCK_SIZE);

  vectorize_kernel<<<NUM_BLOCKS, BLOCK_SIZE>>>(first,last);
}


} // end namespace cuda

} // end namespace device

} // end namespace detail

} // end namespace thrust

#endif // __CUDACC__

