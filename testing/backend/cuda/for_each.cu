#include <unittest/unittest.h>
#include <thrust/for_each.h>

static const size_t NUM_REGISTERS = 100;

template <size_t N> __host__ __device__ void f   (int * x) { int temp = *x; f<N - 1>(x + 1); *x = temp;};
template <>         __host__ __device__ void f<0>(int * x) { }
template <size_t N>
struct CopyFunctorWithManyRegisters
{
  __host__ __device__
  void operator()(int * ptr)
  {
      f<N>(ptr);
  }
};


void TestForEachLargeRegisterFootprint()
{
  int current_device = -1;
  cudaGetDevice(&current_device);
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, current_device);

  thrust::device_vector<int> data(NUM_REGISTERS, 12345);

  thrust::device_vector<int *> input(1, thrust::raw_pointer_cast(&data[0])); // length is irrelevant
  
  thrust::for_each(input.begin(), input.end(), CopyFunctorWithManyRegisters<NUM_REGISTERS>());
}
DECLARE_UNITTEST(TestForEachLargeRegisterFootprint);


void TestForEachNLargeRegisterFootprint()
{
  int current_device = -1;
  cudaGetDevice(&current_device);
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, current_device);

  thrust::device_vector<int> data(NUM_REGISTERS, 12345);

  thrust::device_vector<int *> input(1, thrust::raw_pointer_cast(&data[0])); // length is irrelevant
  
  thrust::for_each_n(input.begin(), input.size(), CopyFunctorWithManyRegisters<NUM_REGISTERS>());
}
DECLARE_UNITTEST(TestForEachNLargeRegisterFootprint);

