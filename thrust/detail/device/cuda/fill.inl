/*
 *  Copyright 2008-2010 NVIDIA Corporation
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


/*! \file fill.inl
 *  \brief Inline file for fill.h.
 */

#include <thrust/detail/config.h>

#if THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_NVCC

/////////////////////
// NVCC definition //
/////////////////////

#include <thrust/detail/util/align.h>
#include <thrust/detail/device/generate.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/detail/type_traits.h>
#include <thrust/extrema.h>

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

template <typename T>
struct fill_functor
{
  T exemplar;

  fill_functor(T _exemplar) 
    : exemplar(_exemplar) {}

  __host__ __device__
  T operator()(void)
  { 
    return exemplar;
  }
}; // end fill_functor


// XXX use this verbose idiom to WAR type-punning problems
template<typename T, size_t num_narrow = sizeof(unsigned long long) / sizeof(T)> struct wide_type;

template<typename T>
struct wide_type<T,2>
{
  wide_type(const T &x)
    : narrow_element1(x),
      narrow_element2(x)
  {}

  T narrow_element1;
  T narrow_element2;
};

template<typename T>
struct wide_type<T,4>
{
  wide_type(const T &x)
    : narrow_element1(x),
      narrow_element2(x),
      narrow_element3(x),
      narrow_element4(x)
  {}

  T narrow_element1;
  T narrow_element2;
  T narrow_element3;
  T narrow_element4;
};


template<typename T>
struct wide_type<T,8>
{
  wide_type(const T &x)
    : narrow_element1(x),
      narrow_element2(x),
      narrow_element3(x),
      narrow_element4(x),
      narrow_element5(x),
      narrow_element6(x),
      narrow_element7(x),
      narrow_element8(x)
  {}

  T narrow_element1;
  T narrow_element2;
  T narrow_element3;
  T narrow_element4;
  T narrow_element5;
  T narrow_element6;
  T narrow_element7;
  T narrow_element8;
};


template<typename Pointer, typename T>
  void wide_fill(Pointer first,
                 Pointer last,
                 const T &exemplar)
{
  typedef typename thrust::iterator_value<Pointer>::type OutputType;

  size_t ALIGNMENT_BOUNDARY = 128; // begin copying blocks at this byte boundary

  size_t n = last - first;

  // type used to pack the Ts
  typedef wide_type<OutputType> WideType;
  WideType wide_exemplar(static_cast<OutputType>(exemplar));

  OutputType *first_raw = thrust::raw_pointer_cast(first);
  OutputType *last_raw  = thrust::raw_pointer_cast(last);

  OutputType *block_first_raw = thrust::min(first_raw + n,   thrust::detail::util::align_up(first_raw, ALIGNMENT_BOUNDARY));
  OutputType *block_last_raw  = thrust::max(block_first_raw, thrust::detail::util::align_down(last_raw, sizeof(WideType)));

  thrust::device_ptr<WideType> block_first_wide = thrust::device_pointer_cast(reinterpret_cast<WideType*>(block_first_raw));
  thrust::device_ptr<WideType> block_last_wide  = thrust::device_pointer_cast(reinterpret_cast<WideType*>(block_last_raw));

  thrust::detail::device::generate(first, thrust::device_pointer_cast(block_first_raw), fill_functor<OutputType>(exemplar));
  thrust::detail::device::generate(block_first_wide, block_last_wide,
                                   fill_functor<WideType>(wide_exemplar));
  thrust::detail::device::generate(thrust::device_pointer_cast(block_last_raw), last, fill_functor<OutputType>(exemplar));
}

template<typename ForwardIterator, typename T>
  void fill(ForwardIterator first,
            ForwardIterator last,
            const T &exemplar,
            thrust::detail::false_type)
{
  fill_functor<T> func(exemplar); 
  thrust::detail::device::generate(first, last, func);
}

template<typename ForwardIterator, typename T>
  void fill(ForwardIterator first,
            ForwardIterator last,
            const T &exemplar,
            thrust::detail::true_type)
{
    typedef typename thrust::iterator_traits<ForwardIterator>::value_type OutputType;

    
    if ( thrust::detail::util::is_aligned<OutputType>(thrust::raw_pointer_cast(&*first)) )
    {
        wide_fill(&*first, &*last, exemplar);
    }
    else
    {
        fill(first, last, exemplar, thrust::detail::false_type());
    }
}

} // end detail

template<typename ForwardIterator, typename T>
  void fill(ForwardIterator first,
            ForwardIterator last,
            const T &exemplar)
{
  typedef typename thrust::iterator_traits<ForwardIterator>::value_type      OutputType;
  typedef typename thrust::iterator_traits<ForwardIterator>::difference_type IndexType;

  // we're compiling with nvcc, launch a kernel
  const bool use_wide_fill = thrust::detail::is_trivial_iterator<ForwardIterator>::value
      && thrust::detail::has_trivial_assign<OutputType>::value
      && (sizeof(OutputType) == 1 || sizeof(OutputType) == 2 || sizeof(OutputType) == 4);

  // XXX WAR nvcc 3.0 usused variable warning
  (void)use_wide_fill;

  detail::fill(first, last, exemplar, thrust::detail::integral_constant<bool, use_wide_fill>());
}

} // end namespace cuda
} // end namespace device
} // end namespace detail
} // end namespace thrust

#else

///////////////////////////
// C++ (only) definition //
///////////////////////////

#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/distance.h>

namespace thrust
{
namespace detail
{
namespace device
{
namespace cuda
{

template<typename ForwardIterator, typename T>
  void fill(ForwardIterator first,
            ForwardIterator last,
            const T &exemplar)
{
  typedef typename thrust::iterator_traits<ForwardIterator>::value_type      OutputType;
  typedef typename thrust::iterator_traits<ForwardIterator>::difference_type IndexType;

  // we can't launch a kernel, implement this with a copy
  IndexType n = thrust::distance(first,last);
  raw_host_buffer<OutputType> temp(n);
  thrust::fill(temp.begin(), temp.end(), exemplar);
  thrust::copy(temp.begin(), temp.end(), first);
}

} // end namespace cuda
} // end namespace device
} // end namespace detail
} // end namespace thrust

#endif // THRUST_DEVICE_COMPILER_NVCC

