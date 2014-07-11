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


namespace thrust
{
namespace system
{
namespace cuda
{
namespace detail
{
namespace detail
{


template<typename Closure, typename RandomAccessIterator>
  struct virtualized_smem_closure
    : Closure
{
  typedef Closure super_t;

  size_t num_elements_per_block;
  RandomAccessIterator virtual_smem;

  __host__ __device__ __thrust_forceinline__
  virtualized_smem_closure(Closure closure, size_t num_elements_per_block, RandomAccessIterator virtual_smem)
    : super_t(closure),
      num_elements_per_block(num_elements_per_block),
      virtual_smem(virtual_smem)
  {}

  __device__ __thrust_forceinline__
  void operator()()
  {
    typename super_t::context_type ctx;

    RandomAccessIterator smem = virtual_smem + num_elements_per_block * ctx.block_index();

    super_t::operator()(smem);
  }
};


} // end detail
} // end detail
} // end cuda
} // end system
} // end thrust

