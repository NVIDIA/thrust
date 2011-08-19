#include <unittest/unittest.h>
#include <cuda_runtime_api.h>
#include <thrust/detail/util/align.h>

void TestCudaMemcpyD2DNullPointerError(void)
{
  cudaError_t result1 = cudaMemcpy((void*)0, (void*)0, 1, cudaMemcpyDeviceToDevice);
  cudaError_t result2 = cudaGetLastError();

  ASSERT_EQUAL(cudaErrorInvalidValue, result1);
  ASSERT_EQUAL(cudaErrorInvalidValue, result2);
}
DECLARE_UNITTEST(TestCudaMemcpyD2DNullPointerError);

template<typename T>
void TestCudaMallocResultAligned(const std::size_t n)
{
  T *ptr = 0;
  cudaMalloc(&ptr, n * sizeof(T));
  cudaFree(ptr);

  ASSERT_EQUAL(true, thrust::detail::util::is_aligned(ptr));
}
DECLARE_VARIABLE_UNITTEST(TestCudaMallocResultAligned);

__global__ void segfault(int *dst, int *src)
{
  *dst = *src;
}

void TestNullPtrDereferenceYieldsError(void)
{
  KNOWN_FAILURE;

//  segfault<<<1,1>>>((int*)0, (int*)0);
//  cudaError_t result = cudaThreadSynchronize();
//
//  // kill the context so it can revive later
//  cudaThreadExit();
//
//  ASSERT_EQUAL(true, (result != cudaSuccess));
}
DECLARE_UNITTEST(TestNullPtrDereferenceYieldsError);

