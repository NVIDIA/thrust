#include <thrust/device_vector.h>
#include <thrust/for_each.h>
#include <thrust/system/cuda/execution_policy.h>
#include <cstdio>
#include <cassert>

// This example demonstrates how to achieve asynchronous, concurrent algorithm execution using
// the CUDA backend's low-level stream-based interface. This program uses thrust::for_each to invoke
// two functors, "ping", and "pong", which communicate via a shared variable, "ball". To encourage
// concurrency, we execute thrust::for_each on two independent CUDA streams using the thrust::cuda::par
// execution policy.
//
// Note that stream usage provides no guarantee of concurrency. If the ping and pong functions
// do not happen to be scheduled concurrently, this program will deadlock.

struct ping
{
  // XXX nvcc issue prevents us from making ball volatile
  //__device__
  //void operator()(volatile int &ball)
  __device__
  void operator()(int &ball)
  {
    // we're not guaranteed concurrency, so only attempt this 1000 times
    unsigned int attempt = 0;

    ball = 1;

    for(unsigned int next_state = 2;
        next_state < 25 && attempt < 1000;
        next_state += 2)
    {
      while(ball != next_state && attempt < 1000)
      {
#if __CUDA_ARCH__ >= 200
        printf("ping waiting for return\n");
#endif
        ++attempt;
      }

      ball += 1;

#if __CUDA_ARCH__ >= 200
      printf("ping! ball is now %d\n", next_state + 1);
#endif
    }
  }
};

struct pong
{
  // XXX nvcc issue prevents us from making ball volatile
  //__device__
  //void operator()(volatile int &ball)
  __device__
  void operator()(int &ball)
  {
    // we're not guaranteed concurrency, so only attempt this 1000 times
    unsigned int attempt = 0;

    for(unsigned int next_state = 1;
        next_state < 25 && attempt < 1000;
        next_state += 2)
    {
      while(ball != next_state && attempt < 1000)
      {
#if __CUDA_ARCH__ >= 200
        printf("pong waiting for return\n");
#endif
        ++attempt;
      }

      ball += 1;

#if __CUDA_ARCH__ >= 200
      printf("pong! ball is now %d\n", next_state + 1);
#endif
    }
  }
};

int main()
{
  cudaStream_t s1, s2;
  cudaStreamCreate(&s1);
  cudaStreamCreate(&s2);

  thrust::device_vector<int> ball(1);

  // Invoke thrust::for_each with the thrust::cuda::par
  // execution policy. Pass the stream s1 as an argument
  // to the .on() function
  thrust::for_each(thrust::cuda::par.on(s1),
                   ball.begin(),
                   ball.end(),
                   ping());

  // Invoke thrust::for_each with the thrust::cuda::par
  // execution policy. Pass the stream s2 as an argument
  // to the .on() function
  thrust::for_each(thrust::cuda::par.on(s2),
                   ball.begin(),
                   ball.end(),
                   pong());

  // Wait for all algorithms executed on the streams to terminate.
  cudaStreamSynchronize(s1);
  cudaStreamSynchronize(s2);

  cudaStreamDestroy(s1);
  cudaStreamDestroy(s2);

  return 0;
}


