#include <thrust/device_vector.h>
#include <thrust/execution_policy.h> // For thrust::device
#include <thrust/reduce.h>
#include <thrust/sequence.h>

#include <cuda_runtime.h>

#include <iostream>

// This example shows how to execute a Thrust device algorithm on an explicit
// CUDA stream. The simple program below fills a vector with the numbers
// [0, 1000) (thrust::sequence) and then sums them (thrust::reduce), executing
// both algorithms on the same custom CUDA stream.

int main()
{
  thrust::device_vector<int> d_vec(1000);

  // Create the stream:
  cudaStream_t custom_stream;
  cudaError_t err = cudaStreamCreate(&custom_stream);
  if (err != cudaSuccess)
  {
    std::cerr << "Error creating stream: " << cudaGetErrorString(err) << "\n";
    return 1;
  }

  // Create a new execution policy with the custom stream:
  auto exec_policy = thrust::device.on(custom_stream);

  // Fill the vector with sequential data.
  // This will execute using the custom stream.
  thrust::sequence(exec_policy, d_vec.begin(), d_vec.end());

  // Sum the data in the vector. This also executes in the custom stream.
  int sum = thrust::reduce(exec_policy, d_vec.cbegin(), d_vec.cend());

  // Free the stream:
  err = cudaStreamDestroy(custom_stream);
  if (err != cudaSuccess)
  {
    std::cerr << "Error destroying stream: " << cudaGetErrorString(err) << "\n";
    return 1;
  }

  // print the sum
  std::cout << "sum is " << sum << std::endl;

  return 0;
}
