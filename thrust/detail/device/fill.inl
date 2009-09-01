/*
 *  Copyright 2008-2009 NVIDIA Corporation
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

#ifdef __CUDACC__

/////////////////////
// NVCC definition //
/////////////////////

#include <thrust/detail/util/align.h>
#include <thrust/detail/device/dereference.h>
#include <thrust/detail/device/cuda/vectorize.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/detail/raw_buffer.h>

namespace thrust
{

namespace detail
{

namespace device
{

namespace detail
{

template <typename ForwardIterator, typename T>
struct fill_functor
{
    ForwardIterator first;
    T exemplar;

    fill_functor(ForwardIterator _first, T _exemplar) 
        : first(_first), exemplar(_exemplar) {}

    template <typename IntegerType>
        __device__
        void operator()(const IntegerType i)
        { 
            thrust::detail::device::dereference(first, i) = exemplar;
        }
}; // end fill_functor


template<typename OutputType, typename T>
  void wide_fill(OutputType * first,
                 OutputType * last,
                 const T &exemplar)
{
    typedef unsigned long long WideType; // type used to pack the Ts

    size_t ALIGNMENT_BOUNDARY = 128; // begin copying blocks at this byte boundary

    size_t n = last - first;

    WideType wide_exemplar;
    for(int i = 0; i < sizeof(WideType)/sizeof(T); i++)
        reinterpret_cast<T *>(&wide_exemplar)[i] = exemplar;

    OutputType * block_first = std::min(first + n,   thrust::detail::util::align_up(first, ALIGNMENT_BOUNDARY));
    OutputType * block_last  = std::max(block_first, thrust::detail::util::align_down(last, sizeof(WideType)));

    size_t pass1 = block_first - first;
    size_t pass2 = (block_last - block_first) / (sizeof(WideType) / sizeof(T));
    size_t pass3 = last - block_last;

    thrust::detail::device::cuda::vectorize(pass1, detail::fill_functor<OutputType *, T       >(first, exemplar));
    thrust::detail::device::cuda::vectorize(pass2, detail::fill_functor<WideType  *, WideType>((WideType *) block_first, wide_exemplar));
    thrust::detail::device::cuda::vectorize(pass3, detail::fill_functor<OutputType *, T       >(block_last, exemplar));
}

template<typename ForwardIterator, typename T>
  void fill(ForwardIterator first,
            ForwardIterator last,
            const T &exemplar,
            thrust::detail::false_type)
{
    detail::fill_functor<ForwardIterator, T> func(first, exemplar); 
    thrust::detail::device::cuda::vectorize(last - first, func);
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
        detail::wide_fill(thrust::raw_pointer_cast(&*first), thrust::raw_pointer_cast(&*last), exemplar);
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
      && (sizeof(OutputType) == 1 || sizeof(OutputType) == 2 || sizeof(OutputType) == 4);
  detail::fill(first, last, exemplar, thrust::detail::integral_constant<bool, use_wide_fill>());
}

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

} // end namespace device

} // end namespace detail

} // end namespace thrust

#endif // __CUDACC__

