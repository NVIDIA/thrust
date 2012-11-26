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


/*! \file fill.inl
 *  \brief Inline file for fill.h.
 */

#include <thrust/detail/config.h>

#include <thrust/distance.h>
#include <thrust/detail/util/align.h>
#include <thrust/generate.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/detail/type_traits.h>
#include <thrust/detail/raw_pointer_cast.h>
#include <thrust/detail/minmax.h>
#include <thrust/detail/internal_functional.h>

#include <thrust/system/cuda/detail/runtime_introspection.h>

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

template<typename WideType, typename System, typename Pointer, typename Size, typename T>
  Pointer wide_fill_n(dispatchable<System> &system,
                      Pointer first,
                      Size n,
                      const T &value)
{
  typedef typename thrust::iterator_value<Pointer>::type OutputType;

  size_t ALIGNMENT_BOUNDARY = 128; // begin copying blocks at this byte boundary

  WideType   wide_exemplar;
  OutputType narrow_exemplars[sizeof(WideType) / sizeof(OutputType)];

  for (size_t i = 0; i < sizeof(WideType) / sizeof(OutputType); i++)
      narrow_exemplars[i] = static_cast<OutputType>(value);

  // cast through char * to avoid type punning warnings
  for (size_t i = 0; i < sizeof(WideType); i++)
      reinterpret_cast<char *>(&wide_exemplar)[i] = reinterpret_cast<char *>(narrow_exemplars)[i];

  OutputType *first_raw = thrust::raw_pointer_cast(first);
  OutputType *last_raw  = first_raw + n;

  OutputType *block_first_raw = (thrust::min)(first_raw + n,   thrust::detail::util::align_up(first_raw, ALIGNMENT_BOUNDARY));
  OutputType *block_last_raw  = (thrust::max)(block_first_raw, thrust::detail::util::align_down(last_raw, sizeof(WideType)));

  // rebind Pointer to WideType
  typedef typename thrust::detail::rebind_pointer<Pointer,WideType>::type WidePtr;
  typedef thrust::detail::pointer_traits<WidePtr> WideTraits;

  // point to the widened range
  WidePtr block_first_wide = WideTraits::pointer_to(reinterpret_cast<WideType&>(*block_first_raw));
  WidePtr block_last_wide  = WideTraits::pointer_to(reinterpret_cast<WideType&>(*block_last_raw));

  thrust::generate(system, first,                   Pointer(block_first_raw),    thrust::detail::fill_functor<OutputType>(value));
  thrust::generate(system, block_first_wide,        block_last_wide,             thrust::detail::fill_functor<WideType>(wide_exemplar));
  thrust::generate(system, Pointer(block_last_raw), first + n,                   thrust::detail::fill_functor<OutputType>(value));

  return first + n;
}

template<typename System, typename OutputIterator, typename Size, typename T>
  OutputIterator fill_n(dispatchable<System> &system,
                        OutputIterator first,
                        Size n,
                        const T &value,
                        thrust::detail::false_type)
{
  thrust::detail::fill_functor<T> func(value); 
  return thrust::generate_n(system, first, n, func);
}

template<typename System, typename OutputIterator, typename Size, typename T>
  OutputIterator fill_n(dispatchable<System> &system,
                        OutputIterator first,
                        Size n,
                        const T &value,
                        thrust::detail::true_type)
{
  typedef typename thrust::iterator_traits<OutputIterator>::value_type OutputType;
  
  if ( thrust::detail::util::is_aligned<OutputType>(thrust::raw_pointer_cast(&*first)) )
  {
      if (compute_capability() < 20)
      {
        // 32-bit writes are faster on G80 and GT200
        typedef unsigned int WideType;
        wide_fill_n<WideType>(system, &*first, n, value);
      }
      else
      {
        // 64-bit writes are faster on Fermi
        typedef unsigned long long WideType;
        wide_fill_n<WideType>(system, &*first, n, value);
      }

      return first + n;
  }
  else
  {
    return fill_n(system, first, n, value, thrust::detail::false_type());
  }
}

} // end detail

template<typename System, typename OutputIterator, typename Size, typename T>
  OutputIterator fill_n(dispatchable<System> &system,
                        OutputIterator first,
                        Size n,
                        const T &value)
{
  typedef typename thrust::iterator_traits<OutputIterator>::value_type      OutputType;

  // we're compiling with nvcc, launch a kernel
  const bool use_wide_fill = thrust::detail::is_trivial_iterator<OutputIterator>::value
      && thrust::detail::has_trivial_assign<OutputType>::value
      && (sizeof(OutputType) == 1 || sizeof(OutputType) == 2 || sizeof(OutputType) == 4);

  // XXX WAR usused variable warning
  (void)use_wide_fill;

  return detail::fill_n(system, first, n, value, thrust::detail::integral_constant<bool, use_wide_fill>());
}

template<typename System, typename ForwardIterator, typename T>
  void fill(dispatchable<System> &system,
            ForwardIterator first,
            ForwardIterator last,
            const T &value)
{
  thrust::system::cuda::detail::fill_n(system, first, thrust::distance(first,last), value);
} // end fill()

} // end namespace detail
} // end namespace cuda
} // end namespace system
} // end namespace thrust

