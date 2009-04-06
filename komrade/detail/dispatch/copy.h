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

#include <komrade/transform.h>
#include <komrade/distance.h>
#include <komrade/iterator/iterator_traits.h>
#include <komrade/detail/type_traits.h>
#include <komrade/detail/make_device_dereferenceable.h>
#include <komrade/device_ptr.h>

#include <komrade/detail/device/cuda/vectorize.h>

namespace komrade
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
                      komrade::random_access_host_iterator_tag,
                      komrade::random_access_host_iterator_tag)
{
    // regular old std::copy for host types
    return std::copy(begin, end, result);
}

template<typename InputIterator,
         typename OutputIterator>
  OutputIterator copy(InputIterator begin,
                      InputIterator end,
                      OutputIterator result,
                      komrade::input_host_iterator_tag,
                      komrade::output_host_iterator_tag)
{
    // regular old std::copy for host types
    return std::copy(begin,end,result);
}

template<typename InputIterator,
         typename OutputIterator>
  OutputIterator copy(InputIterator begin,
                      InputIterator end,
                      OutputIterator result,
                      komrade::input_host_iterator_tag,
                      komrade::forward_host_iterator_tag)
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
                      komrade::input_host_iterator_tag, 
                      komrade::random_access_device_iterator_tag)
{
    // host container to device vector
    typedef typename komrade::iterator_traits<InputIterator>::value_type InputType;

    typename komrade::iterator_traits<InputIterator>::difference_type n = komrade::distance(begin, end);

    // allocate temporary storage
    // do not use a std::vector for temp storage here -- constructors must not be invoked
    InputType *temp = reinterpret_cast<InputType*>(malloc(sizeof(InputType) * n));
    komrade::copy(begin, end, temp);

    result = komrade::copy(temp, temp + n, result);

    free(temp);
    return result;
}

// host pointer to device pointer with matching types
template<typename InputIterator,
         typename OutputIterator>
  OutputIterator copy(InputIterator begin,
                      InputIterator end,
                      OutputIterator result,
                      komrade::random_access_host_iterator_tag,
                      komrade::random_access_device_iterator_tag,
                      typename komrade::iterator_traits<OutputIterator>::value_type *,            // InputIterator::pointer
                      komrade::device_ptr<typename komrade::iterator_traits<OutputIterator>::value_type>)  // OutputIterator::pointer
{
  // specialization for host to device when the types pointed to match (and operator= is not overloaded)
  // this is a trivial copy which is implemented with cudaMemcpy

  // how many elements to copy?
  typename komrade::iterator_traits<OutputIterator>::difference_type n = end - begin;

  // what is the output type?
  typedef typename komrade::iterator_traits<OutputIterator>::value_type OutputType;

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
                      komrade::random_access_host_iterator_tag,
                      komrade::random_access_device_iterator_tag,
                      InputIteratorPointer,
                      OutputIteratorPointer)
{
  typedef typename komrade::iterator_traits<OutputIterator>::value_type OutputType;

  typename komrade::iterator_traits<InputIterator>::difference_type n = komrade::distance(begin,end);

  // allocate temporary storage
  // do not use a std::vector here - constructors must not be invoked
  OutputType *temp = reinterpret_cast<OutputType*>(malloc(sizeof(OutputType) * n));
  komrade::copy(begin, end, temp);

  result = komrade::copy(temp, temp + n, result);

  free(temp);
  return result;
}

// random access host iterator to random access device iterator
template<typename InputIterator,
         typename OutputIterator>
  OutputIterator copy(InputIterator begin,
                      InputIterator end,
                      OutputIterator result,
                      komrade::random_access_host_iterator_tag input_tag,
                      komrade::random_access_device_iterator_tag output_tag)
{
    // dispatch on the type of each iterator's pointers
    // XXX also need to dispatch on if output type fulfills has_trivial_assign_operator
    return copy(begin, end, result, input_tag, output_tag,
            typename komrade::iterator_traits<InputIterator>::pointer(),
            typename komrade::iterator_traits<OutputIterator>::pointer());
}


//////////////////////////
// Device to Host Paths //
//////////////////////////

template<typename InputIterator,
         typename OutputIterator>
  OutputIterator copy(InputIterator begin,
                      InputIterator end,
                      OutputIterator result,
                      komrade::input_device_iterator_tag, 
                      komrade::output_host_iterator_tag)
{
    // XXX throw a compiler error here
}

template<typename InputIterator,
         typename OutputIterator>
  OutputIterator copy(InputIterator begin,
                      InputIterator end,
                      OutputIterator result,
                      komrade::random_access_device_iterator_tag, 
                      komrade::output_host_iterator_tag)
{
    typedef typename komrade::iterator_traits<InputIterator>::value_type T;
    // XXX do not use a std::vector here - constructors must not be invoked
    std::vector<T> temp(end - begin);
    komrade::copy(begin, end, temp.begin());
    return std::copy(temp.begin(), temp.end(), result);
}

template<typename InputIterator,
         typename OutputIterator>
  OutputIterator copy(InputIterator begin,
                      InputIterator end,
                      OutputIterator result,
                      komrade::random_access_device_iterator_tag, 
                      komrade::forward_host_iterator_tag)
{
    typedef typename komrade::iterator_traits<InputIterator>::value_type T;
    // XXX do not use a std::vector here - constructors must not be invoked
    std::vector<T> temp(end - begin);
    komrade::copy(begin, end, temp.begin());
    return std::copy(temp.begin(), temp.end(), result);
}

// const device pointer to host pointer with matching types
template<typename InputIterator,
         typename OutputIterator>
  OutputIterator copy(InputIterator begin,
                      InputIterator end,
                      OutputIterator result,
                      komrade::random_access_device_iterator_tag,
                      komrade::random_access_host_iterator_tag,
                      komrade::device_ptr<const typename komrade::iterator_traits<InputIterator>::value_type>, // match device_ptr<const T>
                      typename komrade::iterator_traits<InputIterator>::value_type *)                          // match T *
{
    // specialization for host to device when the types pointed to match (and operator= is not overloaded)
    // this is a trivial copy which is implemented with cudaMemcpy

    // how many elements to copy?
    typename komrade::iterator_traits<OutputIterator>::difference_type n = end - begin;

    // what is the input type?
    typedef typename komrade::iterator_traits<InputIterator>::value_type InputType;

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
                      komrade::random_access_device_iterator_tag,
                      komrade::random_access_host_iterator_tag,
                      komrade::device_ptr<typename komrade::iterator_traits<InputIterator>::value_type>, // match device_ptr<T>
                      typename komrade::iterator_traits<InputIterator>::value_type *)                    // match T *
{
    return komrade::detail::dispatch::copy(begin, end, result,
            komrade::random_access_device_iterator_tag(),
            komrade::random_access_host_iterator_tag(),
            typename komrade::device_ptr<const typename komrade::iterator_traits<InputIterator>::value_type>(),
            typename komrade::iterator_traits<OutputIterator>::pointer());
}

// random access device iterator to random access host iterator with mixed types
template<typename InputIterator,
         typename OutputIterator,
         typename InputIteratorPointer,
         typename OutputIteratorPointer>
  OutputIterator copy(InputIterator begin,
                      InputIterator end,
                      OutputIterator result,
                      komrade::random_access_device_iterator_tag,
                      komrade::random_access_host_iterator_tag,
                      InputIteratorPointer,
                      OutputIteratorPointer)
{
    typedef typename komrade::iterator_traits<InputIterator>::value_type InputType;

    // allocate temporary storage
    // XXX do not use a vector here - copy constructors must not be invoked
    std::vector<InputType> temp(end - begin);
    komrade::copy(begin, end, temp.begin());

    return komrade::copy(temp.begin(), temp.end(), result);
}

// random access device iterator to random access host iterator
template<typename InputIterator,
         typename OutputIterator>
  OutputIterator copy(InputIterator begin,
                      InputIterator end,
                      OutputIterator result,
                      komrade::random_access_device_iterator_tag input_tag,
                      komrade::random_access_host_iterator_tag output_tag)
{
    // dispatch on the type of each iterator's pointers
    // XXX also need to dispatch on if output type fulfills has_trivial_assign_operator
    return copy(begin, end, result, input_tag, output_tag,
            typename komrade::iterator_traits<InputIterator>::pointer(),
            typename komrade::iterator_traits<OutputIterator>::pointer());
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
                      komrade::random_access_device_iterator_tag, 
                      komrade::random_access_device_iterator_tag)
{
    typedef typename komrade::iterator_traits<InputIterator>::value_type InputType;

#ifdef __CUDACC__
    return komrade::transform(begin, end, result, komrade::identity<InputType>());
#else
    // we're not compiling with nvcc: copy [begin, end) to temp host memory
    typename komrade::iterator_traits<InputIterator>::difference_type n = komrade::distance(begin, end);

    InputType *temp1 = reinterpret_cast<InputType*>(malloc(n * sizeof(InputType)));
    komrade::copy(begin, end, temp1);

    // transform temp1 to OutputType in host memory
    typedef typename komrade::iterator_traits<OutputIterator>::value_type OutputType;
    OutputType *temp2 = reinterpret_cast<OutputType*>(malloc(n * sizeof(OutputType)));
    komrade::transform(temp1, temp1 + n, temp2, komrade::identity<InputType>());

    // copy temp2 to device
    result = komrade::copy(temp2, temp2 + n, result);

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
                      komrade::random_access_device_iterator_tag,
                      komrade::random_access_device_iterator_tag)
{
  // specialization for device to host when the types pointed to match (and operator= is not overloaded)
  // this is a trivial copy which is implemented with cudaMemcpy

  // how many elements to copy?
  typename komrade::iterator_traits<OutputIterator>::difference_type n = end - begin;

  // figure out the type of the element to copy
  typedef typename komrade::iterator_traits<OutputIterator>::value_type T;

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
  OutputIterator copy_if(InputIterator begin,
                         InputIterator end,
                         PredicateIterator stencil,
                         OutputIterator result,
                         Predicate pred,
                         komrade::forward_host_iterator_tag,
                         komrade::forward_host_iterator_tag,
                         komrade::forward_host_iterator_tag)
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
class copy_if_functor
{
  private:
    const InputType * src;
    const StencilType * stc;
          OutputType * dst;
    const Predicate pred;
  public:
    copy_if_functor(const InputType * _src,
                    const StencilType * _stc,
                          OutputType * _dst, 
                    const Predicate _pred) 
        : src(_src), stc(_stc), dst(_dst), pred(_pred) {}
    
    template <typename IntegerType>
        __host__ __device__
    void operator()(const IntegerType i) { if (pred(stc[i])) dst[i] = src[i]; }
}; // end copy_if_functor()

} // end namespace detail

///////////////////////////
// Device Path (copy_if) //
///////////////////////////
template<typename InputIterator,
         typename PredicateIterator,
         typename OutputIterator,
         typename Predicate>
  OutputIterator copy_if(InputIterator begin,
                         InputIterator end,
                         PredicateIterator stencil,
                         OutputIterator result,
                         Predicate pred,
                         komrade::random_access_device_iterator_tag,
                         komrade::random_access_device_iterator_tag,
                         komrade::random_access_device_iterator_tag)
{
    typedef typename komrade::iterator_traits<InputIterator>::value_type InputType;
    typedef typename komrade::iterator_traits<PredicateIterator>::value_type StencilType;
    typedef typename komrade::iterator_traits<OutputIterator>::value_type OutputType;

    // XXX TODO use make_device_dereferenceable here instead of assuming device_ptr.get() will work
    detail::copy_if_functor<InputType,StencilType,OutputType,Predicate> func((&*begin).get(),
                                                                             (&*stencil).get(),
                                                                             (&*result).get(),
                                                                             pred);

    komrade::detail::device::cuda::vectorize(end - begin, func);

    return result + (end - begin);
}

} // end dispatch

} // end detail

} // end komrade

