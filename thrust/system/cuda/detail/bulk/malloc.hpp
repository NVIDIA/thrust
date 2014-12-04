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

#include <thrust/system/cuda/detail/bulk/detail/config.hpp>
#include <thrust/system/cuda/detail/bulk/detail/pointer_traits.hpp>
#include <thrust/system/cuda/detail/bulk/detail/alignment.hpp>
#include <thrust/system/cuda/detail/bulk/uninitialized.hpp>
#include <thrust/detail/config.h>
#include <cstdlib>


BULK_NAMESPACE_PREFIX
namespace bulk
{


inline __device__ bool is_on_chip(void *ptr)
{
  return bulk::detail::is_shared(ptr);
} // end is_on_chip()


template<typename T>
inline __device__ T *on_chip_cast(T *ptr)
{
  extern __shared__ char s_begin[];
  void *result = (reinterpret_cast<char*>(ptr) - s_begin) + s_begin;
  return reinterpret_cast<T*>(result);
} // end on_chip_cast()


namespace detail
{


extern __shared__ int s_data_segment_begin[];


class os
{
  public:
    __device__ inline os(size_t max_data_segment_size)
      : m_program_break(s_data_segment_begin),
        m_max_data_segment_size(max_data_segment_size)
    {
    }


    __device__ inline int brk(void *end_data_segment)
    {
      if(end_data_segment <= m_program_break)
      {
        m_program_break = end_data_segment;
        return 0;
      }

      return -1;
    }


    __device__ inline void *sbrk(size_t increment)
    {
      if(data_segment_size() + increment <= m_max_data_segment_size)
      {
        m_program_break = reinterpret_cast<char*>(m_program_break) + increment;
      } // end if
      else
      {
        return reinterpret_cast<void*>(-1);
      } // end else

      return m_program_break;
    }


    __device__ inline void *program_break() const
    {
      return m_program_break;
    }

    
    __device__ inline void *data_segment_begin() const
    {
      return s_data_segment_begin;
    }


  private:
    __device__ inline size_t data_segment_size()
    {
      return reinterpret_cast<char*>(m_program_break) - reinterpret_cast<char*>(s_data_segment_begin);
    } // end data_segment_size()


    void *m_program_break;

    // XXX this can safely be uint32
    size_t m_max_data_segment_size;
};


// only one instance of this class can logically exist per CTA, and its use is thread-unsafe
class singleton_unsafe_on_chip_allocator
{
  public:
    __device__ inline singleton_unsafe_on_chip_allocator(size_t max_data_segment_size)
      : m_os(max_data_segment_size)
    {}
  
    __device__ inline void *allocate(size_t size)
    {
      size_t aligned_size = align8(size);
    
      block *prev = find_first_free_insertion_point(heap_begin(), heap_end(), aligned_size);
    
      block *b;
    
      if(prev != heap_end() && (b = prev->next()) != heap_end())
      {
        // can we split?
        if((b->size() - aligned_size) >= sizeof(block))
        {
          split_block(b, aligned_size);
        } // end if
    
        b->set_is_free(false);
      } // end if
      else
      {
        // nothing fits, extend the heap
        b = extend_heap(prev, aligned_size);
        if(b == heap_end())
        {
          return 0;
        } // end if
      } // end else
    
      return b->data();
    } // end allocate()
  
  
    __device__ inline void deallocate(void *ptr)
    {
      if(ptr != 0)
      {
        block *b = get_block(ptr);
    
        // free the block
        b->set_is_free(true);
    
        // try to fuse the freed block the previous block
        if(b->prev() && b->prev()->is_free())
        {
          b = b->prev();
          fuse_block(b);
        } // end if
    
        // now try to fuse with the next block
        if(b->next() != heap_end())
        {
          fuse_block(b);
        } // end if
        else
        {
          // the the OS know where the new break is
          m_os.brk(b);
        } // end else
      } // end if
    } // end deallocate()


  private:
    // align to two words
    class block : public bulk::detail::aligned_type<sizeof(size_t) + sizeof(block*)>::type
    {
      public:
        __device__ inline size_t size() const
        {
          return m_size;
        } // end size()

        __device__ void set_size(size_t sz)
        {
          m_size = sz;
        } // end set_size()

        __device__ inline block *prev() const
        {
          return m_prev;
        } // end prev()

        __device__ void set_prev(block *p)
        {
          m_prev = p;
        } // end set_prev()

        // returns a pointer to the indexth byte within this block's data
        __device__ inline void *byte_at(size_t index) const
        {
          return reinterpret_cast<char*>(data()) + index;
        } // end byte_at()

        __device__ inline block *next() const
        {
          return reinterpret_cast<block*>(byte_at(size()));
        } // end next()

        __device__ inline bool is_free() const
        {
          return m_is_free;
        } // end is_free()

        __device__ inline void set_is_free(bool f)
        {
          m_is_free = f;
        } // end set_is_free()

        __device__ inline void *data() const
        {
          return reinterpret_cast<char*>(const_cast<block*>(this)) + sizeof(block);
        } // end data()

      private:
        // this packing ensures that sizeof(block) is compatible with 64b alignment, because:
        // on a 32b platform, sizeof(block) == 64b
        // on a 64b platform, sizeof(block) == 128b
        bool   m_is_free : 1;
        size_t m_size    : 8 * sizeof(size_t) - 1;
        block *m_prev;
    };
  
  
    os     m_os;

    __device__ inline block *heap_begin() const
    {
      return reinterpret_cast<block*>(m_os.data_segment_begin());
    } // end heap_begin()


    __device__ inline block *heap_end() const
    {
      return reinterpret_cast<block*>(m_os.program_break());
    } // end heap_end();
  
  
    __device__ inline void split_block(block *b, size_t size)
    {
      block *new_block;
    
      // emplace a new block within the old one's data segment
      new_block = reinterpret_cast<block*>(b->byte_at(size));
    
      // the new block's size is the old block's size less the size of the split less the size of a block
      new_block->set_size(b->size() - size - sizeof(block));
    
      new_block->set_prev(b);
      new_block->set_is_free(true);
    
      // the old block's size is the size of the split
      b->set_size(size);
    
      // link the old block to the new one
      if(new_block->next() != heap_end())
      {
        new_block->next()->set_prev(new_block);
      } // end if
    } // end split_block()
  
  
    __device__ inline bool fuse_block(block *b)
    {
      if(b->next() != heap_end() && b->next()->is_free())
      {
        // increment b's size by sizeof(block) plus the next's block's data size
        b->set_size(sizeof(block) + b->next()->size() + b->size());
    
        if(b->next() != heap_end())
        {
          b->next()->set_prev(b);
        }
    
        return true;
      }
    
      return false;
    } // end fuse_block()
  
  
    __device__ inline static block *get_block(void *data)
    {
      // the block metadata lives sizeof(block) bytes to the left of data
      void *ptr = reinterpret_cast<char*>(data) - sizeof(block);
      return reinterpret_cast<block *>(ptr);
    } // end get_block()
  
  
    __device__ inline static block *find_first_free_insertion_point(block *first, block *last, size_t size)
    {
      block *prev = last;
    
      while(first != last && !(first->is_free() && first->size() >= size))
      {
        prev = first;
        first = first->next();
      }
    
      return prev;
    } // end find_first_free_insertion_point()
  
  
    __device__ inline block *extend_heap(block *prev, size_t size)
    {
      // the new block goes at the current end of the heap
      block *new_block = heap_end();
    
      // move the break to the right to accomodate both a block and the requested allocation
      if(m_os.sbrk(sizeof(block) + size) == reinterpret_cast<void*>(-1))
      {
        // allocation failed
        return new_block;
      }
    
      on_chip_cast(new_block)->set_size(size);
      on_chip_cast(new_block)->set_prev(prev);
      on_chip_cast(new_block)->set_is_free(false);
    
      return new_block;
    } // end extend_heap()
  
  
    __device__ inline static size_t align8(size_t size)
    {
      return ((((size - 1) >> 3) << 3) + 8);
    } // end align4()
}; // end singleton_unsafe_on_chip_allocator


class singleton_on_chip_allocator
{
  public:
    // XXX mark as __host__ to WAR a warning from uninitialized.construct
    inline __device__ __host__
    singleton_on_chip_allocator(size_t max_data_segment_size)
      : m_mutex(),
        m_alloc(max_data_segment_size)
    {}


    inline __device__
    void *unsafe_allocate(size_t size)
    {
      return m_alloc.allocate(size);
    }


    inline __device__
    void *allocate(size_t size)
    {
      void *result;

      m_mutex.lock();
      {
        result = unsafe_allocate(size);
      } // end critical section
      m_mutex.unlock();

      return result;
    } // end allocate()


    inline __device__
    void unsafe_deallocate(void *ptr)
    {
      m_alloc.deallocate(ptr);
    } // end unsafe_deallocate()


    inline __device__
    void deallocate(void *ptr)
    {
      m_mutex.lock();
      {
        unsafe_deallocate(ptr);
      } // end critical section
      m_mutex.unlock();
    } // end deallocate()


  private:
    class mutex
    {
      public:
        inline __device__
        mutex()
          : m_in_use(0)
        {}


        inline __device__
        bool try_lock()
        {
#if __CUDA_ARCH__ >= 110
          return atomicCAS(&m_in_use, 0, 1) != 0;
#else
          return false;
#endif
        } // end try_lock()


        inline __device__
        void lock()
        {
          // spin while waiting
          while(try_lock()) ;
        } // end lock()


        inline __device__
        void unlock()
        {
          m_in_use = 0;
        } // end unlock()


      private:
        unsigned int m_in_use;
    }; // end mutex


    mutex m_mutex;
    singleton_unsafe_on_chip_allocator m_alloc;
}; // end singleton_on_chip_allocator


// put the object in an anonymous namespace so that non-CUDA compilers don't complain about multiple definitions
namespace
{

__shared__  uninitialized<singleton_on_chip_allocator> s_on_chip_allocator;

} // end anon namespace


inline __device__ void init_on_chip_malloc(size_t max_data_segment_size)
{
  s_on_chip_allocator.construct(max_data_segment_size);
} // end init_on_chip_malloc()


inline __device__ void *on_chip_malloc(size_t size)
{
  void *result = s_on_chip_allocator.get().allocate(size);
  return on_chip_cast(result);
} // end on_chip_malloc()


inline __device__ void on_chip_free(void *ptr)
{
  s_on_chip_allocator.get().deallocate(ptr);
} // end on_chip_free()


inline __device__ void *unsafe_on_chip_malloc(size_t size)
{
  void *result = s_on_chip_allocator.get().unsafe_allocate(size);
  return on_chip_cast(result);
} // end unsafe_on_chip_malloc()


inline __device__ void unsafe_on_chip_free(void *ptr)
{
  s_on_chip_allocator.get().unsafe_deallocate(ptr);
} // end unsafe_on_chip_free()


} // end detail


inline __device__ void *shmalloc(size_t num_bytes)
{
  // first try on_chip_malloc
  void *result = detail::on_chip_malloc(num_bytes);
  
#if __CUDA_ARCH__ >= 200
  if(!result)
  {
    result = std::malloc(num_bytes);
  } // end if
#endif // __CUDA_ARCH__

  return result;
} // end shmalloc()


inline __device__ void *unsafe_shmalloc(size_t num_bytes)
{
  // first try on_chip_malloc
  void *result = detail::unsafe_on_chip_malloc(num_bytes);
  
#if __CUDA_ARCH__ >= 200
  if(!result)
  {
    result = std::malloc(num_bytes);
  } // end if
#endif // __CUDA_ARCH__

  return result;
} // end unsafe_shmalloc()


inline __device__ void shfree(void *ptr)
{
#if __CUDA_ARCH__ >= 200
  if(bulk::is_on_chip(ptr))
  {
    bulk::detail::on_chip_free(bulk::on_chip_cast(ptr));
  } // end if
  else
  {
    std::free(ptr);
  } // end else
#else
  bulk::detail::on_chip_free(bulk::on_chip_cast(ptr));
#endif
} // end shfree()


inline __device__ void unsafe_shfree(void *ptr)
{
#if __CUDA_ARCH__ >= 200
  if(bulk::is_on_chip(ptr))
  {
    bulk::detail::unsafe_on_chip_free(bulk::on_chip_cast(ptr));
  } // end if
  else
  {
    std::free(ptr);
  } // end else
#else
  bulk::detail::unsafe_on_chip_free(bulk::on_chip_cast(ptr));
#endif
} // end unsafe_shfree()


template<typename ConcurrentGroup>
__device__
inline void *malloc(ConcurrentGroup &g, size_t num_bytes)
{
  __shared__ void *s_result;

  // we need to guard access to s_result from other
  // invocations of malloc, so we put a wait at the beginning
  g.wait();

  if(g.this_exec.index() == 0)
  {
    s_result = bulk::unsafe_shmalloc(num_bytes);
  } // end if

  g.wait();

  return s_result;
} // end malloc()


template<typename ConcurrentGroup>
__device__
inline void free(ConcurrentGroup &g, void *ptr)
{
  if(g.this_exec.index() == 0)
  {
    bulk::unsafe_shfree(ptr);
  } // end if

  g.wait();
} // end free()


} // end namespace bulk
BULK_NAMESPACE_SUFFIX

