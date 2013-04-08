#include <unittest/unittest.h>
#include <thrust/for_each.h>
#include <thrust/execution_policy.h>

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


template <typename T>
struct mark_present_for_each
{
  T * ptr;
  __host__ __device__ void
  operator()(T x){ ptr[(int) x] = 1; }
};


template<typename Iterator, typename Function>
__global__ void test_for_each_device_seq_kernel(Iterator first, Iterator last, Function f)
{
  thrust::for_each(thrust::seq, first, last, f);
}


template<typename T>
void TestForEachDeviceSeq(const size_t n)
{
  const size_t output_size = std::min((size_t) 10, 2 * n);
  
  thrust::host_vector<T> h_input = unittest::random_integers<T>(n);
  
  for(size_t i = 0; i < n; i++)
    h_input[i] =  ((size_t) h_input[i]) % output_size;
  
  thrust::device_vector<T> d_input = h_input;
  
  thrust::host_vector<T>   h_output(output_size, (T) 0);
  thrust::device_vector<T> d_output(output_size, (T) 0);
  
  mark_present_for_each<T> h_f;
  mark_present_for_each<T> d_f;
  h_f.ptr = &h_output[0];
  d_f.ptr = (&d_output[0]).get();
  
  thrust::for_each(h_input.begin(), h_input.end(), h_f);
  
  test_for_each_device_seq_kernel<<<1,1>>>(d_input.begin(), d_input.end(), d_f);
  
  ASSERT_EQUAL(h_output, d_output);
}
DECLARE_VARIABLE_UNITTEST(TestForEachDeviceSeq);


template<typename Iterator, typename Size, typename Function>
__global__
void test_for_each_n_device_seq_kernel(Iterator first, Size n, Function f)
{
  thrust::for_each_n(thrust::seq, first, n, f);
}


template<typename T>
void TestForEachNDeviceSeq(const size_t n)
{
  const size_t output_size = std::min((size_t) 10, 2 * n);
  
  thrust::host_vector<T> h_input = unittest::random_integers<T>(n);
  
  for(size_t i = 0; i < n; i++)
    h_input[i] =  ((size_t) h_input[i]) % output_size;
  
  thrust::device_vector<T> d_input = h_input;
  
  thrust::host_vector<T>   h_output(output_size, (T) 0);
  thrust::device_vector<T> d_output(output_size, (T) 0);
  
  mark_present_for_each<T> h_f;
  mark_present_for_each<T> d_f;
  h_f.ptr = &h_output[0];
  d_f.ptr = (&d_output[0]).get();
  
  thrust::for_each_n(h_input.begin(), h_input.size(), h_f);
  
  test_for_each_n_device_seq_kernel<<<1,1>>>(d_input.begin(), d_input.size(), d_f);
  
  ASSERT_EQUAL(h_output, d_output);
}
DECLARE_VARIABLE_UNITTEST(TestForEachNDeviceSeq);

