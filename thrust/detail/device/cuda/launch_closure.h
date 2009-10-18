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

namespace thrust
{

namespace detail
{

namespace device
{

namespace cuda
{

namespace detail
{

template<typename NullaryFunction>
__global__
void launch_closure_by_value(NullaryFunction f)
{
  f();
}

template<typename NullaryFunction>
__global__
void launch_closure_by_pointer(const NullaryFunction *f)
{
  // copy to registers
  NullaryFunction f_reg = *f;
  f_reg();
}

template<typename NullaryFunction,
         bool launch_by_value = sizeof(NullaryFunction) <= 256>
  struct closure_launcher
{
  static void launch(NullaryFunction f, size_t num_blocks, size_t block_size)
  {
    detail::launch_closure_by_value<<<num_blocks,block_size>>>(f);
  }
};

template<typename NullaryFunction>
  struct closure_launcher<NullaryFunction,false>
{
  static void launch(NullaryFunction f, size_t num_blocks, size_t block_size)
  {
    // allocate device memory for the argument
    thrust::device_ptr<void> temp_ptr = thrust::detail::device::cuda::malloc(sizeof(NullaryFunction));

    // cast to NullaryFunction *
    thrust::device_ptr<NullaryFunction> f_ptr(reinterpret_cast<NullaryFunction*>(temp_ptr.get()));

    // copy
    *f_ptr = f;

    // launch
    detail::launch_closure_by_pointer<<<num_blocks, block_size>>>(f_ptr.get());

    // free device memory
    thrust::detail::device::cuda::free(temp_ptr);
  }
};

} // end detail


template<typename NullaryFunction, typename Size1, typename Size2>
  void launch_closure(NullaryFunction f, Size1 num_blocks, Size2 block_size)
{
  detail::closure_launcher<NullaryFunction>::launch(f, num_blocks, block_size);
}


} // end cuda
  
} // end device
  
} // end detail

} // end thrust

#endif // __CUDACC__

