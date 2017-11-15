#include <thrust/functional.h>
#include <thrust/tabulate.h>
#include <thrust/sort.h>
#include <thrust/memory.h>
#include <thrust/system/cuda/memory.h>

#include <new> // for std::bad_alloc
#include <iostream>

// This example demonstrates how to implement a fallback for cudaMalloc
// with a custom allocator. When cudaMalloc fails to allocate device memory
// the fallback_allocator attempts to allocate pinned host memory and
// then map the host buffer into the device address space. The
// fallback_allocator enables the GPU to process data sets that are larger
// than the device memory, albeit with a significantly reduced performance.


// fallback_allocator is a memory allocator which uses pinned host memory as a functional fallback
template <typename T>
struct fallback_allocator
{
 typedef T                                 value_type;
 typedef thrust::device_reference<T>       reference;
 typedef thrust::device_reference<T const> const_reference;
 typedef thrust::device_ptr<T>             pointer;
 typedef thrust::device_ptr<T const>       const_pointer;
 typedef size_t                            size_type;

 template <typename U>
 struct rebind {
   typedef fallback_allocator<U> other;
 };

 // allocate's job to is allocate host memory as a functional fallback when cudaMalloc fails
 pointer allocate(size_type n)
 {
   T *raw_ptr = 0;

   // attempt to allocate device memory
   if (cudaMalloc(&raw_ptr, n * sizeof(T)) == cudaSuccess)
   {
     std::cout << "  allocated " << n * sizeof(T) << " bytes of device memory" << std::endl;
   }
   else
   {
     // reset the last CUDA error
     cudaGetLastError();

     // attempt to allocate pinned host memory
     void *h_ptr = 0;
     if (cudaMallocHost(&h_ptr, n * sizeof(T)) == cudaSuccess)
     {
       // attempt to map host pointer into device memory space
       if (cudaHostGetDevicePointer(&raw_ptr, h_ptr, 0) == cudaSuccess)
       {
         std::cout << "  allocated " << n * sizeof(T) << " bytes of pinned host memory (fallback successful)" << std::endl;
       }
       else
       {
         // reset the last CUDA error
         cudaGetLastError();

         // attempt to deallocate buffer
         std::cout << "  failed to map host memory into device address space (fallback failed)" << std::endl;
         cudaFreeHost(h_ptr);

         throw std::bad_alloc();
       }
     }
     else
     {
       // reset the last CUDA error
       cudaGetLastError();

       std::cout << "  failed to allocate " << n * sizeof(T) << " bytes of memory (fallback failed)" << std::endl;

       throw std::bad_alloc();
     }
   }

   return pointer(raw_ptr);
 }

 // deallocate's job to is inspect where the pointer lives and free it appropriately
 void deallocate(pointer ptr, size_type n)
 {
   void *raw_ptr = thrust::raw_pointer_cast(ptr);

   // determine where memory resides
   cudaPointerAttributes attributes;

   if (cudaPointerGetAttributes(&attributes, raw_ptr) == cudaSuccess)
   {
     // free the memory in the appropriate way
     if (attributes.memoryType == cudaMemoryTypeHost)
     {
       cudaFreeHost(raw_ptr);
     }
     else
     {
       cudaFree(raw_ptr);
     }
   }
 }
};

void sort_with_fallback_allocator(size_t n)
{
  std::cout << "attempting to sort " << n << " values" << std::endl;

  // use our special malloc to allocate the storage
  thrust::device_vector<int, fallback_allocator<int> > d(n);

  // generate unsorted values
  thrust::tabulate(d.begin(), d.end(), thrust::placeholders::_1 % 1024);

  // sort the data using our special allocator
  // if temporary memory is required during the sort, our allocator will be called
  try
  {
    fallback_allocator<int> alloc;
    thrust::sort(thrust::cuda::par(alloc), d.begin(), d.end());
  }
  catch (std::bad_alloc)
  {
    std::cout << "  caught std::bad_alloc from thrust::sort" << std::endl;
  }
}

int main(void)
{
  // check whether device supports mapped host memory
  int device;
  cudaGetDevice(&device);
  cudaDeviceProp properties;
  cudaGetDeviceProperties(&properties, device);

  // this example doesn't work on integrated GPUs
  if (properties.integrated)
  {
    std::cout << "Device #" << device 
              << " [" << properties.name << "] is discrete, not integrated" << std::endl;
    return 0;
  }
  // this example requires both unified addressing and memory mapping
  if (!properties.unifiedAddressing || !properties.canMapHostMemory)
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
    // this sort should not need to fallback to host memory
    sort_with_fallback_allocator((properties.totalGlobalMem / sizeof(int)) / 16);

    // this sort should need to fallback to host memory
    sort_with_fallback_allocator(((properties.totalGlobalMem / sizeof(int)) * 3) / 5);
  }
  catch (std::bad_alloc)
  {
    std::cout << "caught std::bad_alloc from malloc" << std::endl;
  }

  return 0;
}

