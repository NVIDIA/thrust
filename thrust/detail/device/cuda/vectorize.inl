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
void vectorize_kernel(IndexType n, UnaryFunction f)
{
    const IndexType grid_size = blockDim.x * gridDim.x;
    
    for(IndexType i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += grid_size)
        f(i);
}

template<typename IndexType,
         typename UnaryFunction>
void vectorize(IndexType n, UnaryFunction f)
{
    if (n == 0)
        return;

    const size_t BLOCK_SIZE = 256;
    const size_t MAX_BLOCKS = 3 * thrust::experimental::arch::max_active_threads()/BLOCK_SIZE;
    const size_t NUM_BLOCKS = std::min(MAX_BLOCKS, (n + (BLOCK_SIZE - 1) ) / BLOCK_SIZE);

    vectorize_kernel<<<NUM_BLOCKS, BLOCK_SIZE>>>(n, f);
}


} // end namespace cuda

} // end namespace device

} // end namespace detail

} // end namespace thrust

#endif // __CUDACC__

