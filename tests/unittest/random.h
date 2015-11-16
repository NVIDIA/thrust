#pragma once

#include <thrust/host_vector.h>
#include <thrust/random.h>
#include <thrust/detail/type_traits.h>

namespace unittest
{

inline unsigned int hash(unsigned int a)
{
    a = (a+0x7ed55d16) + (a<<12);
    a = (a^0xc761c23c) ^ (a>>19);
    a = (a+0x165667b1) + (a<<5);
    a = (a+0xd3a2646c) ^ (a<<9);
    a = (a+0xfd7046c5) + (a<<3);
    a = (a^0xb55a4f09) ^ (a>>16);
    return a;
}

template<typename T, bool is_float = thrust::detail::is_floating_point<T>::value>
  struct random_integer
{
  T operator()(unsigned int i) const
  {
      thrust::default_random_engine rng(hash(i));
      thrust::uniform_int_distribution<T> dist;

      return static_cast<T>(dist(rng));
  }
};

template<typename T>
  struct random_integer<T,true>
{
  T operator()(unsigned int i) const
  {
      thrust::default_random_engine rng(hash(i));

      return static_cast<T>(rng());
  }
};

template<>
  struct random_integer<bool,false>
{
  bool operator()(unsigned int i) const
  {
      thrust::default_random_engine rng(hash(i));
      thrust::uniform_int_distribution<unsigned int> dist(0,1);

      return dist(rng) == 1;
  }
};


template<typename T>
  struct random_sample
{
  T operator()(unsigned int i) const
  {
      thrust::default_random_engine rng(hash(i));
      thrust::uniform_int_distribution<unsigned int> dist(0,20);

      return static_cast<T>(dist(rng));
  } 
}; 



template<typename T>
thrust::host_vector<T> random_integers(const size_t N)
{
    thrust::host_vector<T> vec(N);
    thrust::transform(thrust::counting_iterator<size_t>(0),
                      thrust::counting_iterator<size_t>(N),
                      vec.begin(),
                      random_integer<T>());

    return vec;
}

template<typename T>
thrust::host_vector<T> random_samples(const size_t N)
{
    thrust::host_vector<T> vec(N);
    thrust::transform(thrust::counting_iterator<size_t>(0),
                      thrust::counting_iterator<size_t>(N),
                      vec.begin(),
                      random_sample<T>());

    return vec;
}

}; //end namespace unittest

