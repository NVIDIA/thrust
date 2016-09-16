#include <thrust/version.h>

// do not attempt to compile this code, which relies on 
// CUDART, without system support
#if THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_NVCC

#include <cuda_runtime_api.h>
#if THRUST_VERSION < 100600
#include <thrust/system/cuda_error.h>
#else
#include <thrust/system/cuda/error.h>
#endif
#include <thrust/system_error.h>
#include <string>

void cuda_safe_call(cudaError_t error, const std::string& message = "")
{
  if(error)
    throw thrust::system_error(error, thrust::cuda_category(), message);
}

struct cuda_timer
{
  cudaEvent_t start;
  cudaEvent_t end;

  cuda_timer(void)
  {
    cuda_safe_call(cudaEventCreate(&start));
    cuda_safe_call(cudaEventCreate(&end));
    restart();
  }

  ~cuda_timer(void)
  {
    cuda_safe_call(cudaEventDestroy(start));
    cuda_safe_call(cudaEventDestroy(end));
  }

  void restart(void)
  {
    cuda_safe_call(cudaEventRecord(start, 0));
  }

  double elapsed_seconds(void)
  {
    cuda_safe_call(cudaEventRecord(end, 0));
    cuda_safe_call(cudaEventSynchronize(end));

    float ms_elapsed;
    cuda_safe_call(cudaEventElapsedTime(&ms_elapsed, start, end));
    return ms_elapsed / 1e3;
  }
};

#endif // THRUST_DEVICE_COMPILER_NVCC

