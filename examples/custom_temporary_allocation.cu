#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/pair.h>
#include <cstdlib>
#include <iostream>
#include <map>
#include <cassert>


// This example demonstrates how to intercept calls to get_temporary_buffer
// and return_temporary_buffer to control how Thrust allocates temporary storage
// during algorithms such as thrust::sort. The idea will be to create a simple
// cache of allocations to search when temporary storage is requested. If a hit
// is found in the cache, we quickly return the cached allocation instead of
// resorting to the more expensive thrust::device_malloc.


// build a simple allocator for caching allocation requests
struct cached_allocator
{
  cached_allocator() {}

  void *allocate(std::ptrdiff_t num_bytes)
  {
    void *result = 0;

    // XXX omp critical will have to do in the absence of std::mutex or thread_local below
    #if _OPENMP
    #pragma omp critical
    #endif // _OPENMP
    {
      // search the cache for a free block
      free_blocks_type::iterator free_block = free_blocks.find(num_bytes);

      if(free_block != free_blocks.end())
      {
        std::cout << "cached_allocator::allocator(): found a hit" << std::endl;

        // get the pointer
        result = free_block->second;

        // erase from the free_blocks map
        free_blocks.erase(free_block);
      }
      else
      {
        // no allocation of the right size exists
        // create a new one with device_malloc
        // throw if device_malloc can't satisfy the request
        try
        {
          std::cout << "cached_allocator::allocator(): no free block found; calling device_malloc" << std::endl;

          result = thrust::device_malloc(num_bytes).get();
        }
        catch(std::runtime_error &e)
        {
          throw;
        }
      }

      // insert the allocated pointer into the allocated_blocks map
      allocated_blocks.insert(std::make_pair(result, num_bytes));
    }

    return result;
  }

  void deallocate(void *ptr)
  {
    // XXX omp critical will have to do in the absence of std::mutex or thread_local below
    #if _OPENMP
    #pragma omp critical
    #endif // _OPENMP
    {
      // erase the allocated block from the allocated blocks map
      allocated_blocks_type::iterator iter = allocated_blocks.find(ptr);
      std::ptrdiff_t num_bytes = iter->second;
      allocated_blocks.erase(iter);

      // insert the block into the free blocks map
      free_blocks.insert(std::make_pair(num_bytes, ptr));
    }
  }

  void free_all()
  {
    std::cout << "cached_allocator::free_all(): cleaning up after ourselves..." << std::endl;

    // deallocate all outstanding blocks in both lists
    for(free_blocks_type::iterator i = free_blocks.begin();
        i != free_blocks.end();
        ++i)
    {
      // transform the pointer to device_ptr before calling device_free
      thrust::device_free(thrust::device_pointer_cast(i->second));
    }

    for(allocated_blocks_type::iterator i = allocated_blocks.begin();
        i != allocated_blocks.end();
        ++i)
    {
      // transform the pointer to device_ptr before calling device_free
      thrust::device_free(thrust::device_pointer_cast(i->first));
    }
  }

  typedef std::multimap<std::ptrdiff_t, void*> free_blocks_type;
  typedef std::map<void *, std::ptrdiff_t>     allocated_blocks_type;

  free_blocks_type      free_blocks;
  allocated_blocks_type allocated_blocks;
};


// the cache is simply a global variable
// XXX ideally this variable is declared thread_local
cached_allocator g_allocator;


// create a tag derived from device_system_tag for distinguishing
// our overloads of get_temporary_buffer and return_temporary_buffer
struct my_tag : thrust::device_system_tag {};


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


int main()
{
  size_t n = 1 << 22;

  thrust::host_vector<int> h_input(n);

  // generate random input
  thrust::generate(h_input.begin(), h_input.end(), rand);

  thrust::device_vector<int> d_input = h_input;
  thrust::device_vector<int> d_result(n);

  size_t num_trials = 5;

  for(size_t i = 0; i < num_trials; ++i)
  {
    // initialize data to sort
    d_result = d_input;

    // tag iterators with my_tag to cause invocations of our
    // get_temporary_buffer and return_temporary_buffer
    // during sort
    thrust::sort(thrust::retag<my_tag>(d_result.begin()),
                 thrust::retag<my_tag>(d_result.end()));

    // ensure the result is sorted
    assert(thrust::is_sorted(d_result.begin(), d_result.end()));
  }

  // free all allocations before the underlying
  // device backend (e.g., CUDART) goes out of scope
  g_allocator.free_all();

  return 0;
}

