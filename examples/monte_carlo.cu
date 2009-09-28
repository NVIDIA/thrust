#include <thrust/random/linear_congruential_engine.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform_reduce.h>

#include <iostream>


// we could vary M & N to find the perf sweet spot

struct estimate_pi
{
  __host__ __device__
  float operator()(unsigned int seed)
  {
    using namespace thrust::experimental::random;

    float sum = 0;
    unsigned int N = 10000;

    // seed a random number generator
    // XXX use this with uniform_01
    minstd_rand rng(seed);

    // take N samples in a quarter circle
    for(unsigned int i = 0; i < N; ++i)
    {
      // draw a sample from the unit square
      float u0 = static_cast<float>(rng()) / minstd_rand::max;
      float u1 = static_cast<float>(rng()) / minstd_rand::max;

      // measure distance from the origin
      float dist = sqrtf(u0*u0 + u1*u1);

      // add 1.0f if (u0,u1) is inside the unit circle
      if(dist <= 1.0f)
        sum += 1.0f;
    }

    // multiply by 4 to get the area of the whole circle
    sum *= 4.0f;

    // divide by N
    return sum / N;
  }
};

int main(void)
{
  // use 30K independent seeds
  unsigned int M = 30000;

  thrust::counting_iterator<unsigned int, thrust::device_space_tag> first(0);

  float estimate = thrust::transform_reduce(first,
                                            first + M,
                                            estimate_pi(),
                                            0.0f,
                                            thrust::plus<float>());
  estimate /= M;

  std::cout << "pi is around " << estimate << std::endl;

  return 0;
}

