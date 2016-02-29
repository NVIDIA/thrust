#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/gather.h>
#include <thrust/iterator/transform_output_iterator.h>
#include <iostream>

struct Functor : thrust::unary_function<float,float>
{
  __host__ __device__
  float operator()(float x)
  {
    return x*2.0f / 3.0f;
  }
};

int main(void)
{
  float u[4] = {4, 3, 2, 1};
  int idx[3] = {3, 0, 1};
  float w[3] = {0, 0, 0};

  thrust::device_vector<float> U(u, u + 4);
  thrust::device_vector<int> IDX(idx, idx + 3);
  thrust::device_vector<float> W(w, w + 3);

  // gather elements and transform them before writing result in memory
  thrust::gather(IDX.begin(), IDX.end(), U.begin(),
                 thrust::make_transform_output_iterator(W.begin(), Functor()));

  std::cout << "result= [ ";
  for (size_t i = 0; i < 3; i++)
    std::cout << W[i] <<  " ";
  std::cout << "] \n";

  return 0;
}

