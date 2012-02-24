#include <thrust/copy.h>
#include <thrust/sort.h>
#include <thrust/memory.h>
#include <thrust/system/cuda/memory.h>

#include <new> // for std::bad_alloc
#include <cassert>
#include <iostream>
#include <iterator>

// This example demonstrates how to intercept calls to get_temporary_buffer
// and return_temporary_buffer to implement a fallback for cudaMalloc.
// When cudaMalloc fails to allocate device memory the fallback_allocator
// attempts to allocate pinned host memory and then map the host buffer 
// into the device address space.  The fallback_allocator enables
// the GPU to process larger data sets that are larger than the device
// memory, albeit with a significantly reduced performance.


// initialize some unsorted data
__global__
void kernel(int * d_ptr, size_t N)
{
  size_t thread_id = blockDim.x * blockIdx.x + threadIdx.x;
  size_t grid_size = blockDim.x * gridDim.x;

  for (size_t i = thread_id; i < N; i += grid_size)
    d_ptr[i] = i % 1024;
}

void * robust_cudaMalloc(size_t N)
{
  void * h_ptr;
  void * d_ptr;

  // attempt to allocate device memory
  if (cudaMalloc(&d_ptr, N) == cudaSuccess)
  {
    std::cout << "  allocated " << N << " bytes of device memory" << std::endl;
    return d_ptr;
  }
  else
  {
    // attempt to allocate pinned host memory
    if (cudaMallocHost(&h_ptr, N) == cudaSuccess)
    {
      // attempt to map host pointer into device memory space
      if (cudaHostGetDevicePointer(&d_ptr, h_ptr, 0) == cudaSuccess)
      {
        std::cout << "  allocated " << N << " bytes of pinned host memory (fallback successful)" << std::endl;
        return d_ptr;
      }
      else
      {
        // attempt to deallocate buffer
        std::cout << "  failed to map host memory into device address space (fallback failed)" << std::endl;
        cudaFreeHost(h_ptr);
        return 0;
      }
    }
    else
    {
      std::cout << "  failed to allocate " << N << " bytes of memory (fallback failed)" << std::endl;
      return 0;
    }
  }
}

void robust_cudaFree(void * ptr)
{
  // determine where memory resides
  cudaPointerAttributes	attributes;

  if (cudaPointerGetAttributes(&attributes, ptr) == cudaSuccess)
  {
    // free the memory in the appropriate way
    if (attributes.memoryType == cudaMemoryTypeHost)
      cudaFreeHost(ptr);
    else
      cudaFree(ptr);
  }
}

// build a simple allocator for using pinned host memory as a functional fallback
struct fallback_allocator
{
  fallback_allocator() {}

  void *allocate(std::ptrdiff_t num_bytes)
  {
    void *result = robust_cudaMalloc(num_bytes);

    if (!result)
      throw std::bad_alloc();

    return result;
  }

  void deallocate(void *ptr)
  {
    robust_cudaFree(ptr);
  }
};


// the fallback allocator is simply a global variable
// XXX ideally this variable is declared thread_local
fallback_allocator g_allocator;

// create a tag derived from cuda::tag for distinguishing
// our overloads of get_temporary_buffer and return_temporary_buffer
struct my_tag : thrust::cuda::tag {};


// overload get_temporary_buffer on my_tag
// its job is to forward allocation requests to g_allocator
template<typename T>
  thrust::pair<T*, std::ptrdiff_t>
    get_temporary_buffer(my_tag, std::ptrdiff_t n)
{
  // ask the allocator for sizeof(T) * n bytes
  T* result = reinterpret_cast<T*>(g_allocator.allocate(sizeof(T) * n));

  // return the pointer and the number of elements allocated
  return thrust::make_pair(result,n);
}

// overload return_temporary_buffer on my_tag
// its job is to forward deallocations to g_allocator
// an overloaded return_temporary_buffer should always accompany
// an overloaded get_temporary_buffer
template<typename Pointer>
  void return_temporary_buffer(my_tag, Pointer p)
{
  // return the pointer to the allocator
  g_allocator.deallocate(thrust::raw_pointer_cast(p));
}


int main(void)
{
  // check whether device supports mapped host memory
  int device;
  cudaGetDevice(&device);
  cudaDeviceProp properties;
  cudaGetDeviceProperties(&properties, device);

  if (!properties.canMapHostMemory)
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

    for (size_t n = one_million; n < one_billion; n *= 2)
    {
      // TODO ideally we'd use the fallback_allocator in the vector too
      //thrust::cuda::vector<int, fallback_allocator> d_vec(n);

      std::cout << "attempting to sort " << n << " values" << std::endl;

      int * raw_ptr = (int *) robust_cudaMalloc(n * sizeof(int));

      if (raw_ptr)
      {
        kernel<<<100,256>>>(raw_ptr, n); // generate unsorted values

        thrust::pointer<int,my_tag> begin = thrust::pointer<int,my_tag>(raw_ptr);
        thrust::pointer<int,my_tag> end   = begin + n;

        thrust::sort(begin, end);

        robust_cudaFree(raw_ptr);
      }
    }
  } catch (std::bad_alloc)
  {
    return 0;
  }

  return 0;
}

