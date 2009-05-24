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


/*! \file for_each.inl
 *  \brief Inline file for for_each.h.
 */

// do not attempt to compile this file with any other compiler
#ifdef __CUDACC__

#include <algorithm>

#include <thrust/iterator/iterator_traits.h>
#include <thrust/experimental/arch.h>

namespace thrust
{

namespace detail
{

namespace device
{

namespace cuda
{

template<typename InputIterator,
         typename UnaryFunction>
__global__             
void for_each_kernel(InputIterator first,
                     InputIterator last,
                     UnaryFunction f)
{
    typedef typename thrust::iterator_traits<InputIterator>::difference_type IndexType;
    
    const IndexType grid_size = blockDim.x * gridDim.x;
    
    first += blockIdx.x * blockDim.x + threadIdx.x;

    while (first < last){
        f(*first);
        first += grid_size;
    }
}


template<typename InputIterator,
         typename UnaryFunction>
void for_each(InputIterator first,
              InputIterator last,
              UnaryFunction f)
{
    if (first >= last) return;  //empty range

    const size_t BLOCK_SIZE = 256;
    const size_t MAX_BLOCKS = thrust::experimental::arch::max_active_threads()/BLOCK_SIZE;
    const size_t NUM_BLOCKS = std::min(MAX_BLOCKS, ( (last - first) + (BLOCK_SIZE - 1) ) / BLOCK_SIZE);

    for_each_kernel<<<NUM_BLOCKS, BLOCK_SIZE>>>(first, last, f);
} 


} // end namespace cuda

} // end namespace device

} // end namespace detail

} // end namespace thrust

#endif // __CUDACC__

