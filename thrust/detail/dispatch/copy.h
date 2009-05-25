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


/*! \file copy.h
 *  \brief Dispatch layer for copy.
 */

#pragma once

#include <cuda_runtime_api.h> // for cudaMemcpy
#include <stdlib.h>           // for malloc & free
#include <algorithm>          // for std::copy

#include <vector>  // TODO remove this when vector is no longer used

#include <thrust/transform.h>
#include <thrust/distance.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/detail/type_traits.h>
#include <thrust/detail/make_device_dereferenceable.h>
#include <thrust/device_ptr.h>

#include <thrust/detail/device/cuda/vectorize.h>

namespace thrust
{

namespace detail
{

namespace dispatch
{

////////////////////////
// Host to Host Paths //
////////////////////////

template<typename InputIterator,
         typename OutputIterator>
  OutputIterator copy(InputIterator begin,
                      InputIterator end,
                      OutputIterator result,
                      thrust::random_access_host_iterator_tag,
                      thrust::random_access_host_iterator_tag)
{
    // regular old std::copy for host types
    return std::copy(begin, end, result);
}

template<typename InputIterator,
         typename OutputIterator>
  OutputIterator copy(InputIterator begin,
                      InputIterator end,
                      OutputIterator result,
                      thrust::input_host_iterator_tag,
                      thrust::output_host_iterator_tag)
{
    // regular old std::copy for host types
    return std::copy(begin,end,result);
}

template<typename InputIterator,
         typename OutputIterator>
  OutputIterator copy(InputIterator begin,
                      InputIterator end,
                      OutputIterator result,
                      thrust::input_host_iterator_tag,
                      thrust::forward_host_iterator_tag)
{
    // regular old std::copy for host types
    return std::copy(begin,end,result);
}


//////////////////////////
// Host to Device Paths //
//////////////////////////

template<typename InputIterator,
         typename OutputIterator>
  OutputIterator copy(InputIterator begin,
                      InputIterator end,
                      OutputIterator result,
                      thrust::input_host_iterator_tag, 
                      thrust::random_access_device_iterator_tag)
{
    // host container to device vector
    typedef typename thrust::iterator_traits<InputIterator>::value_type InputType;

    typename thrust::iterator_traits<InputIterator>::difference_type n = thrust::distance(begin, end);

    // allocate temporary storage
    // do not use a std::vector for temp storage here -- constructors must not be invoked
    InputType *temp = reinterpret_cast<InputType*>(malloc(sizeof(InputType) * n));
    thrust::copy(begin, end, temp);

    result = thrust::copy(temp, temp + n, result);

    free(temp);
    return result;
}

// host pointer to device pointer with matching types
template<typename InputIterator,
         typename OutputIterator>
  OutputIterator copy(InputIterator begin,
                      InputIterator end,
                      OutputIterator result,
                      thrust::random_access_host_iterator_tag,
                      thrust::random_access_device_iterator_tag,
                      typename thrust::iterator_traits<OutputIterator>::value_type *,            // InputIterator::pointer
                      thrust::device_ptr<typename thrust::iterator_traits<OutputIterator>::value_type>)  // OutputIterator::pointer
{
  // specialization for host to device when the types pointed to match (and operator= is not overloaded)
  // this is a trivial copy which is implemented with cudaMemcpy

  // how many elements to copy?
  typename thrust::iterator_traits<OutputIterator>::difference_type n = end - begin;

  // what is the output type?
  typedef typename thrust::iterator_traits<OutputIterator>::value_type OutputType;

  // call cudaMemcpy
  // XXX TODO use make_device_dereferenceable here instead of assuming device_ptr.get() will work
  cudaMemcpy((&*result).get(),
             &*begin,
             n * sizeof(OutputType),
             cudaMemcpyHostToDevice);

  return result + n;
}

// random access host iterator to random access device iterator with mixed types
template<typename InputIterator,
         typename OutputIterator,
         typename InputIteratorPointer,
         typename OutputIteratorPointer>
  OutputIterator copy(InputIterator begin,
                      InputIterator end,
                      OutputIterator result,
                      thrust::random_access_host_iterator_tag,
                      thrust::random_access_device_iterator_tag,
                      InputIteratorPointer,
                      OutputIteratorPointer)
{
  typedef typename thrust::iterator_traits<OutputIterator>::value_type OutputType;

  typename thrust::iterator_traits<InputIterator>::difference_type n = thrust::distance(begin,end);

  // allocate temporary storage
  // do not use a std::vector here - constructors must not be invoked
  OutputType *temp = reinterpret_cast<OutputType*>(malloc(sizeof(OutputType) * n));
  thrust::copy(begin, end, temp);

  result = thrust::copy(temp, temp + n, result);

  free(temp);
  return result;
}

// random access host iterator to random access device iterator
template<typename InputIterator,
         typename OutputIterator>
  OutputIterator copy(InputIterator begin,
                      InputIterator end,
                      OutputIterator result,
                      thrust::random_access_host_iterator_tag input_tag,
                      thrust::random_access_device_iterator_tag output_tag)
{
    // dispatch on the type of each iterator's pointers
    // XXX also need to dispatch on if output type fulfills has_trivial_assign_operator
    return copy(begin, end, result, input_tag, output_tag,
            typename thrust::iterator_traits<InputIterator>::pointer(),
            typename thrust::iterator_traits<OutputIterator>::pointer());
}


//////////////////////////
// Device to Host Paths //
//////////////////////////

template<typename InputIterator,
         typename OutputIterator>
  OutputIterator copy(InputIterator begin,
                      InputIterator end,
                      OutputIterator result,
                      thrust::input_device_iterator_tag, 
                      thrust::output_host_iterator_tag)
{
    // XXX throw a compiler error here
}

template<typename InputIterator,
         typename OutputIterator>
  OutputIterator copy(InputIterator begin,
                      InputIterator end,
                      OutputIterator result,
                      thrust::random_access_device_iterator_tag, 
                      thrust::output_host_iterator_tag)
{
    typedef typename thrust::iterator_traits<InputIterator>::value_type T;
    // XXX do not use a std::vector here - constructors must not be invoked
    std::vector<T> temp(end - begin);
    thrust::copy(begin, end, temp.begin());
    return std::copy(temp.begin(), temp.end(), result);
}

template<typename InputIterator,
         typename OutputIterator>
  OutputIterator copy(InputIterator begin,
                      InputIterator end,
                      OutputIterator result,
                      thrust::random_access_device_iterator_tag, 
                      thrust::forward_host_iterator_tag)
{
    typedef typename thrust::iterator_traits<InputIterator>::value_type T;
    // XXX do not use a std::vector here - constructors must not be invoked
    std::vector<T> temp(end - begin);
    thrust::copy(begin, end, temp.begin());
    return std::copy(temp.begin(), temp.end(), result);
}

// const device pointer to host pointer with matching types
template<typename InputIterator,
         typename OutputIterator>
  OutputIterator copy(InputIterator begin,
                      InputIterator end,
                      OutputIterator result,
                      thrust::random_access_device_iterator_tag,
                      thrust::random_access_host_iterator_tag,
                      thrust::device_ptr<const typename thrust::iterator_traits<InputIterator>::value_type>, // match device_ptr<const T>
                      typename thrust::iterator_traits<InputIterator>::value_type *)                          // match T *
{
    // specialization for host to device when the types pointed to match (and operator= is not overloaded)
    // this is a trivial copy which is implemented with cudaMemcpy

    // how many elements to copy?
    typename thrust::iterator_traits<OutputIterator>::difference_type n = end - begin;

    // what is the input type?
    typedef typename thrust::iterator_traits<InputIterator>::value_type InputType;

    // call cudaMemcpy
    // XXX use make_device_dereferenceable here instead of ssuming that &*begin is device_ptr
    cudaMemcpy(&*result, (&*begin).get(), n * sizeof(InputType), cudaMemcpyDeviceToHost);

    return result + n;
}


// device pointer to host pointer with matching types
template<typename InputIterator,
         typename OutputIterator>
  OutputIterator copy(InputIterator begin,
                      InputIterator end,
                      OutputIterator result,
                      thrust::random_access_device_iterator_tag,
                      thrust::random_access_host_iterator_tag,
                      thrust::device_ptr<typename thrust::iterator_traits<InputIterator>::value_type>, // match device_ptr<T>
                      typename thrust::iterator_traits<InputIterator>::value_type *)                    // match T *
{
    // use a typedef here so that old versions of gcc on OSX don't crash
    typedef typename thrust::device_ptr<const typename thrust::iterator_traits<InputIterator>::value_type> InputDevicePointer;

    return thrust::detail::dispatch::copy(begin, end, result,
            thrust::random_access_device_iterator_tag(),
            thrust::random_access_host_iterator_tag(),
            InputDevicePointer(),
            typename thrust::iterator_traits<OutputIterator>::pointer());
}

// random access device iterator to random access host iterator with mixed types
template<typename InputIterator,
         typename OutputIterator,
         typename InputIteratorPointer,
         typename OutputIteratorPointer>
  OutputIterator copy(InputIterator begin,
                      InputIterator end,
                      OutputIterator result,
                      thrust::random_access_device_iterator_tag,
                      thrust::random_access_host_iterator_tag,
                      InputIteratorPointer,
                      OutputIteratorPointer)
{
    typedef typename thrust::iterator_traits<InputIterator>::value_type InputType;

    // allocate temporary storage
    // XXX do not use a vector here - copy constructors must not be invoked
    std::vector<InputType> temp(end - begin);
    thrust::copy(begin, end, temp.begin());

    return thrust::copy(temp.begin(), temp.end(), result);
}

// random access device iterator to random access host iterator
template<typename InputIterator,
         typename OutputIterator>
  OutputIterator copy(InputIterator begin,
                      InputIterator end,
                      OutputIterator result,
                      thrust::random_access_device_iterator_tag input_tag,
                      thrust::random_access_host_iterator_tag output_tag)
{
    // dispatch on the type of each iterator's pointers
    // XXX also need to dispatch on if output type fulfills has_trivial_assign_operator
    return copy(begin, end, result, input_tag, output_tag,
            typename thrust::iterator_traits<InputIterator>::pointer(),
            typename thrust::iterator_traits<OutputIterator>::pointer());
}


////////////////////////////
// Device to Device Paths //
////////////////////////////


// general case (mixed types)
template<typename InputIterator,
         typename OutputIterator>
  OutputIterator copy(InputIterator begin, 
                      InputIterator end, 
                      OutputIterator result,
                      thrust::random_access_device_iterator_tag, 
                      thrust::random_access_device_iterator_tag)
{
    typedef typename thrust::iterator_traits<InputIterator>::value_type InputType;

#ifdef __CUDACC__
    return thrust::transform(begin, end, result, thrust::identity<InputType>());
#else
    // we're not compiling with nvcc: copy [begin, end) to temp host memory
    typename thrust::iterator_traits<InputIterator>::difference_type n = thrust::distance(begin, end);

    InputType *temp1 = reinterpret_cast<InputType*>(malloc(n * sizeof(InputType)));
    thrust::copy(begin, end, temp1);

    // transform temp1 to OutputType in host memory
    typedef typename thrust::iterator_traits<OutputIterator>::value_type OutputType;
    OutputType *temp2 = reinterpret_cast<OutputType*>(malloc(n * sizeof(OutputType)));
    thrust::transform(temp1, temp1 + n, temp2, thrust::identity<InputType>());

    // copy temp2 to device
    result = thrust::copy(temp2, temp2 + n, result);

    free(temp1);
    free(temp2);

    return result;
#endif // __CUDACC__
}

// special case (types match and has_trivial_assign_operator)
// XXX this case should ensure type has_trivial_assign_operator
template<typename OutputIterator>
  OutputIterator copy(OutputIterator begin,
                      OutputIterator end,
                      OutputIterator result,
                      thrust::random_access_device_iterator_tag,
                      thrust::random_access_device_iterator_tag)
{
  // specialization for device to host when the types pointed to match (and operator= is not overloaded)
  // this is a trivial copy which is implemented with cudaMemcpy

  // how many elements to copy?
  typename thrust::iterator_traits<OutputIterator>::difference_type n = end - begin;

  // figure out the type of the element to copy
  typedef typename thrust::iterator_traits<OutputIterator>::value_type T;

  // call cudaMemcpy
  // XXX use make_device_dereferenceable here instead of assuming &*result & &*begin are device_ptr
  void * dest       = (&*result).get();
  const void * src  = (&*begin).get();
  cudaMemcpy(dest, src, n * sizeof(T), cudaMemcpyDeviceToDevice);

  return result + n;
} // end copy()



/////////////////////////
// Host Path (copy_if) //
/////////////////////////

template<typename InputIterator,
         typename PredicateIterator,
         typename OutputIterator,
         typename Predicate>
  OutputIterator copy_when(InputIterator begin,
                           InputIterator end,
                           PredicateIterator stencil,
                           OutputIterator result,
                           Predicate pred,
                           thrust::forward_host_iterator_tag,
                           thrust::forward_host_iterator_tag,
                           thrust::forward_host_iterator_tag)
{
    while(begin != end){
        if(pred(*stencil))
            *result = *begin;
        ++begin;
        ++stencil;
        ++result;
    } // end while

    return result;
}

namespace detail
{

//////////////
// Functors //
//////////////
template <typename InputType, typename StencilType, typename OutputType, typename Predicate>
class copy_when_functor
{
  private:
    const InputType * src;
    const StencilType * stc;
          OutputType * dst;
    const Predicate pred;
  public:
    copy_when_functor(const InputType * _src,
                      const StencilType * _stc,
                            OutputType * _dst, 
                      const Predicate _pred) 
        : src(_src), stc(_stc), dst(_dst), pred(_pred) {}
    
    template <typename IntegerType>
        __host__ __device__
    void operator()(const IntegerType i) { if (pred(stc[i])) dst[i] = src[i]; }
}; // end copy_when_functor()

} // end namespace detail

///////////////////////////
// Device Path (copy_when) //
///////////////////////////
template<typename InputIterator,
         typename PredicateIterator,
         typename OutputIterator,
         typename Predicate>
  OutputIterator copy_when(InputIterator begin,
                           InputIterator end,
                           PredicateIterator stencil,
                           OutputIterator result,
                           Predicate pred,
                           thrust::random_access_device_iterator_tag,
                           thrust::random_access_device_iterator_tag,
                           thrust::random_access_device_iterator_tag)
{
    typedef typename thrust::iterator_traits<InputIterator>::value_type InputType;
    typedef typename thrust::iterator_traits<PredicateIterator>::value_type StencilType;
    typedef typename thrust::iterator_traits<OutputIterator>::value_type OutputType;

    // XXX TODO use make_device_dereferenceable here instead of assuming device_ptr.get() will work
    detail::copy_when_functor<InputType,StencilType,OutputType,Predicate> func((&*begin).get(),
                                                                               (&*stencil).get(),
                                                                               (&*result).get(),
                                                                               pred);

    thrust::detail::device::cuda::vectorize(end - begin, func);

    return result + (end - begin);
}

} // end dispatch

} // end detail

} // end thrust

