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


#include <thrust/detail/util/static.h>
#include <thrust/detail/util/align.h>
#include <thrust/detail/device/cuda/vectorize.h>
#include <thrust/copy.h>

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

template <typename ValueType>
struct fill_functor
{
        ValueType * ptr;
  const ValueType   exemplar;

  fill_functor(ValueType * _ptr, const ValueType _exemplar) 
      : ptr(_ptr), exemplar(_exemplar) {}
  
  template <typename IntegerType>
      __host__ __device__
  void operator()(const IntegerType i) const { ptr[i] = exemplar; }
}; // end fill_functor


template<typename InputIterator, typename T>
  void wide_fill(InputIterator first,
                 InputIterator last,
                 const T &exemplar)
{
    typedef typename thrust::iterator_traits<InputIterator>::value_type      InputType;
    typedef typename thrust::iterator_traits<InputIterator>::difference_type IndexType;

    typedef unsigned long long WideType; // type used to pack the Ts

    IndexType ALIGNMENT_BOUNDARY = 128; // begin copying blocks at this byte boundary

    // XXX use make_device_dereferenceable here instead of assuming &*first & &*last are device_ptr
    InputType * begin = (&*first).get();
    InputType * end  = (&*last).get();

    IndexType n = (last - first);

    WideType wide_exemplar;
    for(int i = 0; i < sizeof(WideType)/sizeof(T); i++)
        reinterpret_cast<T *>(&wide_exemplar)[i] = exemplar;

    InputType * block_begin = std::min(begin + n,   thrust::detail::util::align_up(begin, ALIGNMENT_BOUNDARY));
    InputType * block_end   = std::max(block_begin, thrust::detail::util::align_down(end, sizeof(WideType)));

    IndexType pass1 = block_begin - begin;
    IndexType pass2 = (block_end - block_begin) / (sizeof(WideType) / sizeof(T));
    IndexType pass3 = end - block_end;

    thrust::detail::device::cuda::vectorize(pass1, detail::fill_functor<InputType>(begin, exemplar));
    thrust::detail::device::cuda::vectorize(pass2, detail::fill_functor<WideType>((WideType *) block_begin, wide_exemplar));
    thrust::detail::device::cuda::vectorize(pass3, detail::fill_functor<InputType>(block_end, exemplar));
}

template<typename InputIterator, typename T>
  void fill(InputIterator first,
            InputIterator last,
            const T &exemplar,
            thrust::detail::util::Bool2Type<false>)
{
    typedef typename thrust::iterator_traits<InputIterator>::value_type      InputType;
    // XXX use make_device_dereferenceable here instead of assuming &*first is a device_ptr
    thrust::detail::device::cuda::vectorize(last - first, detail::fill_functor<InputType>((&*first).get(), exemplar));
}

template<typename InputIterator, typename T>
  void fill(InputIterator first,
            InputIterator last,
            const T &exemplar,
            thrust::detail::util::Bool2Type<true>)
{
    typedef typename thrust::iterator_traits<InputIterator>::value_type      InputType;

    // XXX use make_device_dereferenceable here instead of assuming &*first is a device_ptr
    if (thrust::detail::util::is_aligned<InputType>((&*first).get()))
    {
        detail::wide_fill<InputIterator,T>(first, last, exemplar);
    }
    else
    {
        fill(first, last, exemplar, thrust::detail::util::Bool2Type<false>());
        // XXX use make_device_dereferenceable here instead of assuming that first is a device_ptr
        thrust::detail::device::cuda::vectorize(last - first, detail::fill_functor<InputType>((&*first).get(), exemplar));
    }
}

} // end detail

template<typename InputIterator, typename T>
  void fill(InputIterator first,
            InputIterator last,
            const T &exemplar)
{
  typedef typename thrust::iterator_traits<InputIterator>::value_type      InputType;
  typedef typename thrust::iterator_traits<InputIterator>::difference_type IndexType;

#ifdef __CUDACC__
  // we're compiling with nvcc, launch a kernel
  const bool use_wide_fill = sizeof(InputType) == 1 || sizeof(InputType) == 2 || sizeof(InputType) == 4;
  detail::fill(first, last, exemplar, thrust::detail::util::Bool2Type<use_wide_fill>());
#else
  // we can't launch a kernel, implement this with a copy
  IndexType n = thrust::distance(first,last);
  InputType *temp = reinterpret_cast<InputType*>(::malloc(n * sizeof(InputType)));
  thrust::fill(temp, temp + n, exemplar);
  thrust::copy(temp, temp + n, first);
  ::free(temp);
#endif // __CUDACC__
}

} // end namespace cuda

} // end namespace device

} // end namespace detail

} // end namespace thrust

