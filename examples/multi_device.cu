#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/reduce.h>
#include "cuda_multidev.h"

template<typename Pointer>
  struct placed_ptr
{
  placed_ptr(Pointer ptr, int place)
    : ptr_(ptr), place_(place) {}

  Pointer ptr_;
  int place_;
};

template<typename Pointer>
  int get_place(placed_ptr<Pointer> ptr)
{
  return ptr.place_;
}

template<typename Pointer>
  Pointer get_pointer(placed_ptr<Pointer> ptr)
{
  return ptr.ptr_;
}

template<typename Pointer>
  placed_ptr<Pointer> make_placed_pointer(Pointer ptr, int place)
{
  return placed_ptr<Pointer>(ptr,place);
}

template<typename Iterator>
  struct has_place
    : thrust::detail::false_type
{};

template<typename Pointer>
  struct has_place<placed_ptr<Pointer> >
    : thrust::detail::true_type
{};


template<typename Pointer>
  struct has_place<thrust::detail::normal_iterator<Pointer> >
    : has_place<Pointer>
{};


int reduce(const thrust::host_vector<thrust::device_vector<int> > &vectors)
{
  int result = 0;

  for(int d = 0; d < vectors.size(); ++d)
  {
    cudaSetActiveDevice(d);
    result += thrust::reduce(vectors[d].begin(), vectors[d].end());
  }

  return result;
}

template<typename Iterator> int reduce(Iterator,Iterator);

namespace place_dispatch
{

template<typename Iterator>
  int reduce(Iterator first, Iterator last, thrust::detail::false_type)
{
  return thrust::reduce(first, last);
}

template<typename Iterator>
  int reduce(Iterator first, Iterator last, thrust::detail::true_type)
{
  cudaSetActiveDevice(get_place(first));
  int result = ::reduce(get_ptr(first), get_ptr(last));
  return result;
}

}

template<typename Iterator>
  int reduce(Iterator first, Iterator last)
{
  return place_dispatch::reduce(first, last, typename has_place<Iterator>::type());
}

int main(void)
{
  int num_devices = 0;
  int N = 1000;

  cudaGetDeviceCount(&num_devices);

  thrust::host_vector<thrust::device_vector<int> > vectors(num_devices);

  for(int d = 0; d < num_devices; ++d)
  {
    cudaSetActiveDevice(d);
    vectors[d].resize(N);
    thrust::fill(vectors[d].begin(), vectors[d].end(), 13);
  }

  assert(reduce(vectors) == 13 * N * num_devices);

  for(int d = 0; d < num_devices; ++d)
  {
    cudaSetActiveDevice(d);
    cudaThreadExitActiveDevice();
  }

  return 0;
}

