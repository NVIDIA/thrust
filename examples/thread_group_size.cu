// When `THRUST_DEBUG_SYNC` is defined the configuration of each launch will be
// printed to standard output, which is useful for this example. It will also
// cause additional unnecessary synchronization and activate debugging
// codepaths, so don't use it in production.
#define THRUST_DEBUG_SYNC

#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/thread_group_size.h>
#include <iostream>

struct saxpy
{
  float a;

  saxpy(float a) : a(a) {}

  __host__ __device__
  float operator()(float x, float y)
  {
    return a * x + y;
  }
};

int main()
{
  float a = 2.0f;
  float x[4] = {1, 2, 3, 4};
  float y[4] = {1, 1, 1, 1};

  {
    thrust::device_vector<float> X(x, x + 4);
    thrust::device_vector<float> Y(y, y + 4);

    thrust::transform(X.begin(), X.end(),
                      Y.begin(),
                      Y.begin(),
                      saxpy(a));

    std::cout << "SAXPY (default tuning)" << std::endl;
    for (size_t i = 0; i < 4; i++)
      std::cout << a << " * " << x[i] << " + " << y[i] << " = " << Y[i] << std::endl;
  }

  {
    thrust::device_vector<float> X(x, x + 4);
    thrust::device_vector<float> Y(y, y + 4);

    thrust::transform(X.begin(), X.end(),
                      Y.begin(),
                      Y.begin(),
                      thrust::with_thread_group_size<128>(saxpy(a)));

    std::cout << "SAXPY (with_thread_group_size<128>)" << std::endl;
    for (size_t i = 0; i < 4; i++)
      std::cout << a << " * " << x[i] << " + " << y[i] << " = " << Y[i] << std::endl;
  }
}

