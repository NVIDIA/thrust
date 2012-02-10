#include <thrust/iterator/iterator_traits.h>
#include <thrust/device_vector.h>
#include <thrust/for_each.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <iostream>

// This example demonstrates how to build a minimal custom
// Thrust backend by intercepting for_each's dispatch.

// We begin by defining a "tag", which distinguishes our novel
// backend from other Thrust backends.
// We'll derive my_tag from thrust::device_system_tag to inherit
// the functionality of the default device backend.
struct my_tag : thrust::device_system_tag {};

// Next, we'll create a novel version of for_each which only
// applies to iterators "tagged" with my_tag.
// Our version of for_each will print a message and then call
// the regular version of for_each.

// The first parameter to our for_each is my_tag. This allows
// Thrust to locate it when dispatching thrust::for_each on iterators
// tagged with my_tag. The following parameters are as normal.
template<typename Iterator, typename Function>
  Iterator for_each(my_tag, 
                    Iterator first, Iterator last,
                    Function f)
{
  // Our version of for_each was invoked because first and last are tagged with my_tag

  // output a message
  std::cout << "Hello, world from for_each(my_tag)!" << std::endl;

  // to call the normal version of for_each, we need to "retag" the iterator
  // arguments with device_system_tag using the retag function. It's safe to
  // retag the iterators with device_system_tag because my_tag is related by
  // convertibility.
  thrust::for_each(thrust::retag<thrust::device_system_tag>(first),
                   thrust::retag<thrust::device_system_tag>(last),
                   f);

  return last;
}

int main()
{
  // Create a device_vector, whose iterators are tagged with device_system_tag
  thrust::device_vector<int> vec(1);

  // To ensure that our version of for_each is invoked during dispatch, we
  // retag vec's iterators with my_tag. It's safe to retag the iterators with
  // my_tag because device_system_tag is a base class of my_tag
  thrust::for_each(thrust::retag<my_tag>(vec.begin()),
                   thrust::retag<my_tag>(vec.end()),
                   thrust::identity<int>());

  // Other algorithms that Thrust implements with thrust::for_each will also
  // cause our version of for_each to be invoked when their iterator arguments
  // are tagged with my_tag. Because we did not define a specialized version of
  // transform, Thrust dispatches the version it knows for device_system_tag,
  // which my_tag inherits.
  thrust::transform(thrust::retag<my_tag>(vec.begin()),
                    thrust::retag<my_tag>(vec.end()),
                    thrust::retag<my_tag>(vec.begin()),
                    thrust::identity<int>());

  // Iterators without my_tag are handled normally.
  thrust::for_each(vec.begin(), vec.end(), thrust::identity<int>());

  return 0;
}

