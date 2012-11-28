#include <thrust/functional.h>
#include <thrust/tabulate.h>
#include <thrust/sort.h>
#include <thrust/memory.h>
#include <thrust/system/cuda/memory.h>

#include <new> // for std::bad_alloc
#include <iostream>

// This example demonstrates how to intercept calls to malloc
// and free to implement a fallback for cudaMalloc.
// When cudaMalloc fails to allocate device memory the fallback_allocator
// attempts to allocate pinned host memory and then map the host buffer 
// into the device address space.  The fallback_allocator enables
// the GPU to process data sets that are larger than the device
// memory, albeit with a significantly reduced performance.


// derive a simple allocator from cuda::dispatchable for using pinned host memory as a functional fallback
struct fallback_allocator : thrust::cuda::dispatchable<fallback_allocator> {};


// overload malloc on fallback_allocator to implement our special malloc 
// its job to is allocate host memory as a functional fallback when cudaMalloc fails
void *malloc(fallback_allocator, std::size_t n)
{
  void *result = 0;

  // attempt to allocate device memory
  if(cudaMalloc(&result, n) == cudaSuccess)
  {
    std::cout << "  allocated " << n << " bytes of device memory" << std::endl;
  }
  else
  {
    // attempt to allocate pinned host memory
    void *h_ptr = 0;
    if(cudaMallocHost(&h_ptr, n) == cudaSuccess)
    {
      // attempt to map host pointer into device memory space
      if(cudaHostGetDevicePointer(&result, h_ptr, 0) == cudaSuccess)
      {
        std::cout << "  allocated " << n << " bytes of pinned host memory (fallback successful)" << std::endl;
      }
      else
      {
        // attempt to deallocate buffer
        std::cout << "  failed to map host memory into device address space (fallback failed)" << std::endl;
        cudaError_t error = cudaFreeHost(h_ptr);
        if(error)
        {
          throw thrust::system_error(error, thrust::cuda_category(), "cudaFreeHost failed");
        }

        result = 0;
      }
    }
    else
    {
      std::cout << "  failed to allocate " << n << " bytes of memory (fallback failed)" << std::endl;
      result = 0;
    }
  }

  return result;
}


// overload free on fallback_allocator to implement our special free 
// its job to is inspect where the pointer lives and free it appropriately
template<typename Pointer>
void free(fallback_allocator, Pointer ptr)
{
  void *raw_ptr = thrust::raw_pointer_cast(ptr);

  // determine where memory resides
  cudaPointerAttributes	attributes;

  if(cudaPointerGetAttributes(&attributes, raw_ptr) == cudaSuccess)
  {
    // free the memory in the appropriate way
    if(attributes.memoryType == cudaMemoryTypeHost)
    {
      cudaFreeHost(raw_ptr);
    }
    else
    {
      cudaFree(raw_ptr);
    }
  }
}


int main(void)
{
  // check whether device supports mapped host memory
  int device;
  cudaGetDevice(&device);
  cudaDeviceProp properties;
  cudaGetDeviceProperties(&properties, device);

  fallback_allocator alloc;

  if(!properties.canMapHostMemory)
  {
    std::cout << "Device #" << device 
              << " [" << properties.name << "] does not support memory mapping" << std::endl;
    return 0;
  }
  else
  {
    std::cout << "Testing fallback_allocator on device #" << device 
              << " [" << properties.name << "] with " 
              << properties.totalGlobalMem << " bytes of device memory" << std::endl;
  }

  try
  {
    size_t one_million = 1 << 20;
    size_t one_billion = 1 << 30;

    for(size_t n = one_million; n < one_billion; n *= 2)
    {
      // TODO ideally we'd use the fallback_allocator in the vector too
      //thrust::cuda::vector<int, fallback_allocator> d_vec(n);

      std::cout << "attempting to sort " << n << " values" << std::endl;

      // use our special malloc to allocate
      int *raw_ptr = (int *) malloc(alloc, n * sizeof(int));
      if(!raw_ptr)
      {
        throw std::bad_alloc();
      }

      thrust::cuda::pointer<int> begin = thrust::cuda::pointer<int>(raw_ptr);
      thrust::cuda::pointer<int> end   = begin + n;

      // generate unsorted values
      thrust::tabulate(begin, end, thrust::placeholders::_1 % 1024);

      // sort the data using our special allocator
      // if temporary memory is required during the sort,
      // our versions of malloc & free will be called
      try
      {
        thrust::sort(alloc, begin, end);
      }
      catch(std::bad_alloc)
      {
        std::cout << "  caught std::bad_alloc from thrust::sort" << std::endl;
      }

      free(alloc, raw_ptr);
    }
  }
  catch(std::bad_alloc)
  {
    std::cout << "caught std::bad_alloc from malloc" << std::endl;
  }

  return 0;
}

