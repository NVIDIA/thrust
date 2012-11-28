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

#include <thrust/detail/config.h>
#include <thrust/system/cuda/detail/copy_cross_system.h>
#include <thrust/detail/copy.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/detail/temporary_array.h>
#include <thrust/detail/dispatch/is_trivial_copy.h>
#include <thrust/system/cuda/detail/trivial_copy.h>

namespace thrust
{
namespace detail
{

// XXX WAR circular #inclusion problem
template<typename,typename> class temporary_array;

} // end detail

namespace system
{
namespace cuda
{
namespace detail
{


// general input to random access case
template<typename System1,
         typename System2,
         typename InputIterator,
         typename RandomAccessIterator>
  RandomAccessIterator copy_cross_system(cross_system<System1,System2> systems,
                                         InputIterator begin,
                                         InputIterator end,
                                         RandomAccessIterator result,
                                         thrust::incrementable_traversal_tag, 
                                         thrust::random_access_traversal_tag)
{
  //std::cerr << std::endl;
  //std::cerr << "general copy_host_to_device(): InputIterator: " << typeid(InputIterator).name() << std::endl;
  //std::cerr << "general copy_host_to_device(): OutputIterator: " << typeid(OutputIterator).name() << std::endl;

  typedef typename thrust::iterator_value<InputIterator>::type InputType;

  // allocate temporary storage in System1
  thrust::detail::temporary_array<InputType, System1> temp(systems.system1,begin,end);
  return thrust::copy(systems, temp.begin(), temp.end(), result);
}

template<typename System1,
         typename System2,
         typename InputIterator,
         typename Size,
         typename RandomAccessIterator>
  RandomAccessIterator copy_cross_system_n(cross_system<System1,System2> systems,
                                           InputIterator first,
                                           Size n,
                                           RandomAccessIterator result,
                                           thrust::incrementable_traversal_tag, 
                                           thrust::random_access_traversal_tag)
{
  typedef typename thrust::iterator_value<InputIterator>::type InputType;

  // allocate and copy to temporary storage System1
  thrust::detail::temporary_array<InputType, System1> temp(systems.system1, first, n);

  // recurse
  return copy_cross_system(systems, temp.begin(), temp.end(), result);
}


// random access to general output case
template<typename System1,
         typename System2,
         typename RandomAccessIterator,
         typename OutputIterator>
  OutputIterator copy_cross_system(cross_system<System1,System2> systems,
                                   RandomAccessIterator begin,
                                   RandomAccessIterator end,
                                   OutputIterator result,
                                   thrust::random_access_traversal_tag, 
                                   thrust::incrementable_traversal_tag)
{
  typedef typename thrust::iterator_value<RandomAccessIterator>::type InputType;

  // copy to temporary storage in System2
  thrust::detail::temporary_array<InputType,System2> temp(systems.system2, systems.system1, begin, end);

  return thrust::copy(systems.system2, temp.begin(), temp.end(), result);
}

template<typename System1,
         typename System2,
         typename RandomAccessIterator,
         typename Size,
         typename OutputIterator>
  OutputIterator copy_cross_system_n(cross_system<System1,System2> systems,
                                     RandomAccessIterator first,
                                     Size n,
                                     OutputIterator result,
                                     thrust::random_access_traversal_tag, 
                                     thrust::incrementable_traversal_tag)
{
  typedef typename thrust::iterator_value<RandomAccessIterator>::type InputType;

  // copy to temporary storage in System2
  thrust::detail::temporary_array<InputType,System2> temp(systems.system2, systems.system1, first, n);

  // copy temp to result
  return thrust::copy(systems.system2, temp.begin(), temp.end(), result);
}


// trivial copy
template<typename System1,
         typename System2,
         typename RandomAccessIterator1,
         typename RandomAccessIterator2>
  RandomAccessIterator2 copy_cross_system(cross_system<System1,System2> systems,
                                          RandomAccessIterator1 begin,
                                          RandomAccessIterator1 end,
                                          RandomAccessIterator2 result,
                                          thrust::random_access_traversal_tag,
                                          thrust::random_access_traversal_tag,
                                          thrust::detail::true_type) // trivial copy
{
//  std::cerr << std::endl;
//  std::cerr << "random access copy_device_to_host(): trivial" << std::endl;
//  std::cerr << "general copy_device_to_host(): RandomAccessIterator1: " << typeid(RandomAccessIterator1).name() << std::endl;
//  std::cerr << "general copy_device_to_host(): RandomAccessIterator2: " << typeid(RandomAccessIterator2).name() << std::endl;
  
  // how many elements to copy?
  typename thrust::iterator_traits<RandomAccessIterator1>::difference_type n = end - begin;

  thrust::system::cuda::detail::trivial_copy_n(systems, begin, n, result);

  return result + n;
}


namespace detail
{

// random access non-trivial iterator to random access iterator
template<typename System1,
         typename System2,
         typename RandomAccessIterator1,
         typename RandomAccessIterator2>
  RandomAccessIterator2 non_trivial_random_access_copy_cross_system(cross_system<System1,System2> systems,
                                                                    RandomAccessIterator1 begin,
                                                                    RandomAccessIterator1 end,
                                                                    RandomAccessIterator2 result,
                                                                    thrust::detail::false_type) // InputIterator is non-trivial
{
  // copy the input to a temporary input system buffer of OutputType
  typedef typename thrust::iterator_value<RandomAccessIterator2>::type OutputType;

  // allocate temporary storage in System1
  thrust::detail::temporary_array<OutputType,System1> temp(systems.system1, begin, end);

  // recurse
  return copy_cross_system(systems, temp.begin(), temp.end(), result);
}

template<typename System1,
         typename System2,
         typename RandomAccessIterator1,
         typename RandomAccessIterator2>
  RandomAccessIterator2 non_trivial_random_access_copy_cross_system(cross_system<System1,System2> systems,
                                                                    RandomAccessIterator1 begin,
                                                                    RandomAccessIterator1 end,
                                                                    RandomAccessIterator2 result,
                                                                    thrust::detail::true_type) // InputIterator is trivial
{
  typename thrust::iterator_difference<RandomAccessIterator1>::type n = thrust::distance(begin, end);

  // allocate temporary storage in System2
  // retain the input's type for the intermediate storage
  // do not initialize the storage (the 0 does this)
  typedef typename thrust::iterator_value<RandomAccessIterator1>::type InputType;
  thrust::detail::temporary_array<InputType,System2> temp(0, systems.system2, n);

  // force a trivial (memcpy) copy of the input to the temporary
  // note that this will not correctly account for copy constructors
  // but there's nothing we can do about that
  // XXX one thing we might try is to use pinned memory for the temporary storage
  //     this might allow us to correctly account for copy constructors
  thrust::system::cuda::detail::trivial_copy_n(systems, begin, n, temp.begin());

  // finally, copy to the result
  return thrust::copy(systems.system2, temp.begin(), temp.end(), result);
}

} // end detail


// random access iterator to random access host iterator with non-trivial copy
template<typename System1,
         typename System2,
         typename RandomAccessIterator1,
         typename RandomAccessIterator2>
  RandomAccessIterator2 copy_cross_system(cross_system<System1,System2> systems,
                                          RandomAccessIterator1 begin,
                                          RandomAccessIterator1 end,
                                          RandomAccessIterator2 result,
                                          thrust::random_access_traversal_tag,
                                          thrust::random_access_traversal_tag,
                                          thrust::detail::false_type) // is_trivial_copy
{
  // dispatch a non-trivial random access cross system copy based on whether or not the InputIterator is trivial
  return detail::non_trivial_random_access_copy_cross_system(systems, begin, end, result,
      typename thrust::detail::is_trivial_iterator<RandomAccessIterator1>::type());
}

// random access iterator to random access iterator
template<typename System1,
         typename System2,
         typename RandomAccessIterator1,
         typename RandomAccessIterator2>
  RandomAccessIterator2 copy_cross_system(cross_system<System1,System2> systems,
                                          RandomAccessIterator1 begin,
                                          RandomAccessIterator1 end,
                                          RandomAccessIterator2 result,
                                          thrust::random_access_traversal_tag input_traversal,
                                          thrust::random_access_traversal_tag output_traversal)
{
  // dispatch on whether this is a trivial copy
  return copy_cross_system(systems, begin, end, result, input_traversal, output_traversal,
          typename thrust::detail::dispatch::is_trivial_copy<RandomAccessIterator1,RandomAccessIterator2>::type());
}

template<typename System1,
         typename System2,
         typename RandomAccessIterator1,
         typename Size,
         typename RandomAccessIterator2>
  RandomAccessIterator2 copy_cross_system_n(cross_system<System1,System2> systems,
                                            RandomAccessIterator1 first,
                                            Size n,
                                            RandomAccessIterator2 result,
                                            thrust::random_access_traversal_tag input_traversal,
                                            thrust::random_access_traversal_tag output_traversal)
{
  // implement with copy_cross_system
  return copy_cross_system(systems, first, first + n, result, input_traversal, output_traversal);
}

/////////////////
// Entry Point //
/////////////////

template<typename System1,
         typename System2,
         typename InputIterator,
         typename OutputIterator>
  OutputIterator copy_cross_system(cross_system<System1,System2> systems,
                                   InputIterator begin, 
                                   InputIterator end, 
                                   OutputIterator result)
{
  return copy_cross_system(systems, begin, end, result, 
          typename thrust::iterator_traversal<InputIterator>::type(),
          typename thrust::iterator_traversal<OutputIterator>::type());
}

template<typename System1,
         typename System2,
         typename InputIterator,
         typename Size,
         typename OutputIterator>
  OutputIterator copy_cross_system_n(cross_system<System1,System2> systems,
                                     InputIterator begin, 
                                     Size n, 
                                     OutputIterator result)
{
  return copy_cross_system_n(systems, begin, n, result, 
          typename thrust::iterator_traversal<InputIterator>::type(),
          typename thrust::iterator_traversal<OutputIterator>::type());
}

} // end detail
} // end cuda
} // end system
} // end thrust

