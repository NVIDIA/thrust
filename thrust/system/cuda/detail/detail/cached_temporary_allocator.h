/*
 *  Copyright 2008-2012 NVIDIA Corporation
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

#include <thrust/detail/config.h>
#include <thrust/detail/raw_pointer_cast.h>
#include <thrust/detail/allocator/allocator_traits.h>
#include <thrust/detail/allocator/temporary_allocator.h>
#include <thrust/pair.h>
#include <map>

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


template<typename System, template<typename> class BaseSystem>
  class cached_temporary_allocator
    : public BaseSystem<cached_temporary_allocator<System,BaseSystem> >
{
  private:
    typedef thrust::detail::temporary_allocator<char,System> base_allocator_type;
    typedef thrust::detail::allocator_traits<base_allocator_type> traits;
    typedef typename traits::pointer                              allocator_pointer;
    typedef std::multimap<std::ptrdiff_t, void*>                  free_blocks_type;
    typedef std::map<void *, std::ptrdiff_t>                      allocated_blocks_type;

    base_allocator_type   m_base_allocator;
    free_blocks_type      free_blocks;
    allocated_blocks_type allocated_blocks;

    void free_all()
    {
      // deallocate all outstanding blocks in both lists
      for(free_blocks_type::iterator i = free_blocks.begin();
          i != free_blocks.end();
          ++i)
      {
        // transform the pointer to allocator_pointer before calling deallocate
        traits::deallocate(m_base_allocator, allocator_pointer(reinterpret_cast<char*>(i->second)), i->first);
      }

      for(allocated_blocks_type::iterator i = allocated_blocks.begin();
          i != allocated_blocks.end();
          ++i)
      {
        // transform the pointer to allocator_pointer before calling deallocate
        traits::deallocate(m_base_allocator, allocator_pointer(reinterpret_cast<char*>(i->first)), i->second);
      }
    }

  public:
    cached_temporary_allocator(thrust::dispatchable<System> &system)
      : m_base_allocator(system)
    {}

    ~cached_temporary_allocator()
    {
      // free all allocations when cached_allocator goes out of scope
      free_all();
    }

    void *allocate(std::ptrdiff_t num_bytes)
    {
      void *result = 0;

      // search the cache for a free block
      free_blocks_type::iterator free_block = free_blocks.find(num_bytes);

      if(free_block != free_blocks.end())
      {
        // get the pointer
        result = free_block->second;

        // erase from the free_blocks map
        free_blocks.erase(free_block);
      }
      else
      {
        // no allocation of the right size exists
        // create a new one with m_base_allocator
        // allocate memory and convert to raw pointer
        result = thrust::raw_pointer_cast(traits::allocate(m_base_allocator, num_bytes));
      }

      // insert the allocated pointer into the allocated_blocks map
      allocated_blocks.insert(std::make_pair(result, num_bytes));

      return result;
    }

    void deallocate(void *ptr)
    {
      // erase the allocated block from the allocated blocks map
      allocated_blocks_type::iterator iter = allocated_blocks.find(ptr);
      std::ptrdiff_t num_bytes = iter->second;
      allocated_blocks.erase(iter);

      // insert the block into the free blocks map
      free_blocks.insert(std::make_pair(num_bytes, ptr));
    }
};


// overload get_temporary_buffer on cached_temporary_allocator
// note that we take a reference to cached_temporary_allocator
template<typename T, typename System, template<typename> class BaseSystem>
  thrust::pair<T*, std::ptrdiff_t>
    get_temporary_buffer(cached_temporary_allocator<System,BaseSystem> &alloc, std::ptrdiff_t n)
{
  // ask the allocator for sizeof(T) * n bytes
  T* result = reinterpret_cast<T*>(alloc.allocate(sizeof(T) * n));

  // return the pointer and the number of elements allocated
  return thrust::make_pair(result,n);
}


// overload return_temporary_buffer on cached_temporary_allocator
// an overloaded return_temporary_buffer should always accompany
// an overloaded get_temporary_buffer
template<typename Pointer, typename System, template<typename> class BaseSystem>
  void return_temporary_buffer(cached_temporary_allocator<System,BaseSystem> &alloc, Pointer p)
{
  // return the pointer to the allocator
  alloc.deallocate(thrust::raw_pointer_cast(p));
}


} // end detail
} // end detail
} // end cuda
} // end system
} // end thrust

