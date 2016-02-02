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

/*! \file thrust/system/cuda/experimental/managed_allocator.h
 *  \brief An allocator which creates new elements in "managed" memory with \p cudaManagedMalloc
 */

#include <algorithm>
#include <functional>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>

#include <thrust/device_vector.h>
#include <thrust/system/cpp/execution_policy.h>
#include <thrust/system/cuda/execution_policy.h>
#include <thrust/sort.h>

template<typename T> class managed_allocator;

template<>
  class managed_allocator<void>
{
  public:
    typedef void           value_type;
    typedef void       *   pointer;
    typedef const void *   const_pointer;
    typedef std::size_t    size_type;
    typedef std::ptrdiff_t difference_type;

    // convert a managed_allocator<void> to managed_allocator<U>
    template<typename U>
      struct rebind
    {
      typedef managed_allocator<U> other;
    }; // end rebind
}; // end managed_allocator


template<typename T>
  class managed_allocator
{
  public:
    typedef T              value_type;
    typedef T*             pointer;
    typedef const T*       const_pointer;
    typedef T&             reference;
    typedef const T&       const_reference;
    typedef std::size_t    size_type;
    typedef std::ptrdiff_t difference_type;

    template<typename U>
      struct rebind
    {
      typedef managed_allocator<U> other;
    }; // end rebind

    __host__ __device__
    inline managed_allocator() {}

    __host__ __device__
    inline ~managed_allocator() {}

    __host__ __device__
    inline managed_allocator(managed_allocator const &) {}

    template<typename U>
    __host__ __device__
    inline managed_allocator(managed_allocator<U> const &) {}

    __host__ __device__
    inline pointer address(reference r) { return &r; }

    __host__ __device__
    inline const_pointer address(const_reference r) { return &r; }

    __host__
    inline pointer allocate(size_type cnt,
                            const_pointer = 0)
    {
      if(cnt > this->max_size())
      {
        throw std::bad_alloc();
      } // end if

      pointer result(0);
      cudaError_t error = cudaMallocManaged(reinterpret_cast<void**>(&result), cnt * sizeof(value_type), cudaMemAttachGlobal);

      if(error)
      {
        throw std::bad_alloc();
      } // end if

      return result;
    } // end allocate()

    __host__
    inline void deallocate(pointer p, size_type cnt)
    {
      cudaError_t error = cudaFree(p);

      if(error)
      {
        throw thrust::system_error(error, thrust::cuda_category());
      } // end if
    } // end deallocate()

    inline size_type max_size() const
    {
      return (std::numeric_limits<size_type>::max)() / sizeof(T);
    } // end max_size()

    __host__ __device__
    inline bool operator==(managed_allocator const& x) { return true; }

    __host__ __device__
    inline bool operator!=(managed_allocator const &x) { return !operator==(x); }
}; // end managed_allocator

template<typename Vector>
void operate(Vector& vec)
{
  try
  {
    // initialize data on host
    thrust::tabulate(thrust::cpp::par, vec.begin(), vec.end(), thrust::placeholders::_1 % 1024);

    // sort data on device
    thrust::sort(thrust::cuda::par, vec.begin(), vec.end());

    // synchronize to avoid bus error
    cudaDeviceSynchronize();
  }
  catch(std::exception)
  {
    std::cout << "caught std::exception from sort" << std::endl;
  }
}


int main()
{
  // check whether device supports mapped host memory
  int device;
  cudaGetDevice(&device);
  cudaDeviceProp properties;
  cudaGetDeviceProperties(&properties, device);

  // this example requires both unified addressing and memory mapping
  if(!properties.unifiedAddressing || !properties.canMapHostMemory)
  {
    std::cout << "Device #" << device 
              << " [" << properties.name << "] does not support memory mapping" << std::endl;
    return 0;
  }
  else
  {
    std::cout << "Testing managed_allocator on device #" << device 
              << " [" << properties.name << "] with " 
              << properties.totalGlobalMem << " bytes of device memory" << std::endl;
  }

  size_t size = 1 << 5;
  thrust::host_vector<int, managed_allocator<int> > host_vec(size);
  operate(host_vec);

  thrust::device_vector<int, managed_allocator<int> > device_vec(size);
  operate(device_vec);

  thrust::copy(device_vec.begin(), device_vec.end(), std::ostream_iterator<int>(std::cout, " "));
  std::cout << std::endl;

  return 0;
}
