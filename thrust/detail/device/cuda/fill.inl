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
#include <thrust/generate.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/detail/type_traits.h>
#include <thrust/extrema.h>
#include <thrust/detail/internal_functional.h>

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


template<typename Pointer, typename Size, typename T>
  Pointer wide_fill_n(Pointer first,
                      Size n,
                      const T &value)
{
  typedef typename thrust::iterator_value<Pointer>::type OutputType;

  size_t ALIGNMENT_BOUNDARY = 128; // begin copying blocks at this byte boundary

  // type used to pack the Ts
  typedef wide_type<OutputType> WideType;
  WideType wide_exemplar(static_cast<OutputType>(value));

  OutputType *first_raw = thrust::raw_pointer_cast(first);
  OutputType *last_raw  = first_raw + n;

  OutputType *block_first_raw = (thrust::min)(first_raw + n,   thrust::detail::util::align_up(first_raw, ALIGNMENT_BOUNDARY));
  OutputType *block_last_raw  = (thrust::max)(block_first_raw, thrust::detail::util::align_down(last_raw, sizeof(WideType)));

  thrust::device_ptr<WideType> block_first_wide = thrust::device_pointer_cast(reinterpret_cast<WideType*>(block_first_raw));
  thrust::device_ptr<WideType> block_last_wide  = thrust::device_pointer_cast(reinterpret_cast<WideType*>(block_last_raw));

  thrust::generate(first, thrust::device_pointer_cast(block_first_raw), fill_functor<OutputType>(value));
  thrust::generate(block_first_wide, block_last_wide,
                   fill_functor<WideType>(wide_exemplar));
  thrust::generate(thrust::device_pointer_cast(block_last_raw), first + n, fill_functor<OutputType>(value));

  return first + n;
}

template<typename OutputIterator, typename Size, typename T>
  OutputIterator fill_n(OutputIterator first,
                        Size n,
                        const T &value,
                        thrust::detail::false_type)
{
  thrust::detail::fill_functor<T> func(value); 
  return thrust::generate_n(first, n, func);
}

template<typename OutputIterator, typename Size, typename T>
  OutputIterator fill_n(OutputIterator first,
                        Size n,
                        const T &value,
                        thrust::detail::true_type)
{
  typedef typename thrust::iterator_traits<OutputIterator>::value_type OutputType;
  
  if ( thrust::detail::util::is_aligned<OutputType>(thrust::raw_pointer_cast(&*first)) )
  {
      wide_fill_n(&*first, n, value);
      return first + n;
  }

  return fill_n(first, n, value, thrust::detail::false_type());
}

} // end detail

template<typename OutputIterator, typename Size, typename T>
  OutputIterator fill_n(OutputIterator first,
                        Size n,
                        const T &value)
{
  typedef typename thrust::iterator_traits<OutputIterator>::value_type      OutputType;

  // we're compiling with nvcc, launch a kernel
  const bool use_wide_fill = thrust::detail::is_trivial_iterator<OutputIterator>::value
      && thrust::detail::has_trivial_assign<OutputType>::value
      && (sizeof(OutputType) == 1 || sizeof(OutputType) == 2 || sizeof(OutputType) == 4);

  // XXX WAR nvcc 3.0 usused variable warning
  (void)use_wide_fill;

  return detail::fill_n(first, n, value, thrust::detail::integral_constant<bool, use_wide_fill>());
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

template<typename OutputIterator, typename Size, typename T>
  OutputIterator fill_n(OutputIterator first,
                        Size n,
                        const T &value)
{
  typedef typename thrust::iterator_traits<OutputIterator>::value_type      OutputType;

  // we can't launch a kernel, implement this with a copy
  raw_host_buffer<OutputType> temp(n);
  thrust::fill_n(temp.begin(), n, value);

  // XXX implement this with copy_n
  thrust::copy(temp.begin(), temp.end(), first);

  return first + n;
}

} // end namespace cuda
} // end namespace device
} // end namespace detail
} // end namespace thrust

#endif // THRUST_DEVICE_COMPILER_NVCC

