/*
 *  Copyright 2008-2011 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in ctbbliance with the License.
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

/*! \file tbb/memory.h
 *  \brief Classes for managing OpenMP-typed memory.
 */

#pragma once

#include <thrust/detail/config.h>
#include <thrust/system/tbb/detail/tag.h>
#include <thrust/memory.h>
#include <thrust/detail/type_traits.h>
#include <thrust/detail/allocator/malloc_allocator.h>
#include <ostream>

namespace thrust
{
namespace system
{
namespace tbb
{

template<typename> class pointer;

} // end tbb
} // end system
} // end thrust


// specialize std::iterator_traits to avoid problems with the name of
// pointer's constructor shadowing its nested pointer type
// do this before pointer is defined so the specialization is correctly
// used inside the definition
namespace std
{

template<typename Element>
  struct iterator_traits<thrust::system::tbb::pointer<Element> >
{
  private:
    typedef thrust::system::tbb::pointer<Element> ptr;

  public:
    typedef typename ptr::iterator_category       iterator_category;
    typedef typename ptr::value_type              value_type;
    typedef typename ptr::difference_type         difference_type;
    typedef ptr                                   pointer;
    typedef typename ptr::reference               reference;
}; // end iterator_traits

} // end std


namespace thrust
{
namespace system
{
namespace tbb
{

// forward declaration of reference for pointer
template<typename Element> class reference;

// XXX nvcc + msvc have trouble instantiating reference below
//     this is a workaround
namespace detail
{

template<typename Element>
  struct reference_msvc_workaround
{
  typedef thrust::system::tbb::reference<Element> type;
}; // end reference_msvc_workaround

} // end detail

template<typename T>
  class pointer
    : public thrust::pointer<
               T,
               thrust::system::tbb::tag,
               thrust::system::tbb::reference<T>,
               thrust::system::tbb::pointer<T>
             >
{
  private:
    typedef thrust::pointer<
      T,
      thrust::system::tbb::tag,
      //thrust::system::tbb::reference<T>,
      typename detail::reference_msvc_workaround<T>::type,
      thrust::system::tbb::pointer<T>
    > super_t;

  public:
    // XXX doxygenate these

    // note that tbb::pointer's member functions need __host__ __device__
    // to interoperate with nvcc + iterators' dereference member function

    __host__ __device__
    pointer() : super_t() {}

    template<typename OtherT>
    __host__ __device__
    explicit pointer(OtherT *ptr) : super_t(ptr) {}

    template<typename OtherPointer>
    __host__ __device__
    pointer(const OtherPointer &other,
            typename thrust::detail::enable_if_pointer_is_convertible<
              OtherPointer,
              pointer
            >::type * = 0) : super_t(other) {}

    template<typename OtherPointer>
    __host__ __device__
    typename thrust::detail::enable_if_pointer_is_convertible<
      OtherPointer,
      pointer,
      pointer &
    >::type
    operator=(const OtherPointer &other)
    {
      return super_t::operator=(other);
    }
}; // end pointer


template<typename T>
  class reference
    : public thrust::reference<
               T,
               thrust::system::tbb::pointer<T>,
               thrust::system::tbb::reference<T>
             >
{
  private:
    typedef thrust::reference<
      T,
      thrust::system::tbb::pointer<T>,
      thrust::system::tbb::reference<T>
    > super_t;

  public:
    typedef typename super_t::value_type value_type;
    typedef typename super_t::pointer    pointer;

    template<typename OtherT>
    __host__ __device__
    reference(const reference<OtherT> &other,
              typename thrust::detail::enable_if_convertible<
                typename reference<OtherT>::pointer,
                pointer
              >::type * = 0)
      : super_t(other)
    {}

    __host__ __device__
    explicit reference(const pointer &ptr)
      : super_t(ptr)
    {}

    template<typename OtherT>
    reference &operator=(const reference<OtherT> &other);

    reference &operator=(const value_type &x);
}; // end reference

template<typename T>
__host__ __device__
void swap(reference<T> x, reference<T> y);

inline pointer<void> malloc(std::size_t n);

inline void free(pointer<void> ptr);

// XXX upon c++11
// template<typename T> using allocator = thrust::detail::malloc_allocator<T,tag,pointer<T> >;

template<typename T>
  struct allocator
    : thrust::detail::malloc_allocator<
        T,
        tag,
        pointer<T>
      >
{
  template<typename U>
    struct rebind
  {
    typedef allocator<U> other;
  };

  __host__ __device__
  inline allocator() {}

  __host__ __device__
  inline allocator(const allocator &) {}

  template<typename U>
  __host__ __device__
  inline allocator(const allocator<U> &) {}

  __host__ __device__
  inline ~allocator() {}
}; // end allocator

} // end tbb
} // end system

// alias tbb's members at top-level
namespace tbb
{

using thrust::system::tbb::pointer;
using thrust::system::tbb::reference;
using thrust::system::tbb::malloc;
using thrust::system::tbb::free;
using thrust::system::tbb::allocator;

} // end tbb

} // end thrust

#include <thrust/system/tbb/detail/memory.inl>

// make the backend's entry points available to dispatch
#include <thrust/system/tbb/detail/algorithm.h>

