#pragma once

#include <stdlib.h>
#include <vector>

#include <thrust/host_vector.h>

const unsigned int DEFAULT_SEED = 13;

namespace thrusttest
{

template<typename T>
  struct random_integer
{
  T operator()(void) const
  {
      T value = 0;
        
      for(int i = 0; i < sizeof(T); i++)
          value ^= T(rand() & 0xff) << (8*i);

      return value;
  }
};

template<>
  struct random_integer<bool>
{
  bool operator()(void) const
  {
    return rand() > RAND_MAX/2 ? false : true;
  }
};

template<>
  struct random_integer<float>
{
  float operator()(void) const
  {
      return rand();
  }
};

template<>
  struct random_integer<double>
{
  double operator()(void) const
  {
      return rand();
  }
};


template<typename T>
  struct random_sample
{
  T operator()(void) const
  {
    random_integer<T> rnd;
    return rnd() % 21 - 10;
  } 
}; 

template<>
  struct random_sample<float>
{
  float operator()(void) const
  {
    return 20.0f * (rand() / (RAND_MAX + 1.0f)) - 10.0f;
  }
};

template<>
  struct random_sample<double>
{
  double operator()(void) const
  {
    return 20.0 * (rand() / (RAND_MAX + 1.0)) - 10.0;
  }
};



template<typename T>
thrust::host_vector<T> random_integers(const size_t N)
{
    srand(DEFAULT_SEED);

    thrust::host_vector<T> vec(N);
    random_integer<T> rnd;

    for(size_t i = 0; i < N; i++)
        vec[i] = rnd();

    return vec;
}

template<typename T>
thrust::host_vector<T> random_samples(const size_t N)
{
    srand(DEFAULT_SEED);

    thrust::host_vector<T> vec(N);
    random_sample<T> rnd;

    for(size_t i = 0; i < N; i++)
        vec[i] = rnd();

    return vec;
}

}; //end namespace thrusttest

