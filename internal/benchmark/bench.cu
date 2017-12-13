#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <algorithm>
#include <numeric>

#include <iostream>
#include <iomanip>
#include <cstdlib>

#include "random.h"
#include "timer.h"

#ifndef NO_TBB
#include "tbb_algos.h"
#endif

// Input size
size_t N = 32 << 20;

//////////////////////
// Test Definitions //
//////////////////////

// STL tests
template <typename T>
struct stl_reduce_test
{
  typedef typename std::vector<T> Vector;  Vector v;
  void        setup(void) { v.resize(N);  randomize(v); }
  void        run(void)   { if (std::accumulate(v.begin(), v.end(), T(0)) == 0) std::cout << "xyz"; } // prevent optimizer from removing body
  std::string name(void)  { return std::string("std::accumulate");  }
};

template <typename T>
struct stl_transform_test
{
  typedef typename std::vector<T> Vector;  Vector v;
  void        setup(void) { v.resize(N);  randomize(v); }
  void        run(void)   { std::transform(v.begin(), v.end(), v.begin(), thrust::negate<int>()); }
  std::string name(void)  { return std::string("std::transform");  }
};

template <typename T>
struct stl_scan_test
{
  typedef typename std::vector<T> Vector;  Vector v;
  void        setup(void) { v.resize(N);  randomize(v); }
  void        run(void)   { std::partial_sum(v.begin(), v.end(), v.begin()); }
  std::string name(void)  { return std::string("std::partial_sum");  }
};

template <typename T>
struct stl_sort_test
{
  typedef typename std::vector<T> Vector;  Vector v;
  void        setup(void) { v.resize(N);  randomize(v); }
  void        run(void)   { std::sort(v.begin(), v.end()); }
  std::string name(void)  { return std::string("std::sort");  }
};

#ifndef NO_TBB
// TBB tests
template <typename T>
struct tbb_reduce_test
{
  typedef typename std::vector<T> Vector;  Vector v;
  void        setup(void) { v.resize(N);  randomize(v); }
  void        run(void)   { tbb_reduce(v); }
  std::string name(void)  { return std::string("tbb::parallel_reduce");  }
};

template <typename T>
struct tbb_transform_test
{
  typedef typename std::vector<T> Vector;  Vector v;
  void        setup(void) { v.resize(N);  randomize(v); }
  void        run(void)   { tbb_transform(v); }
  std::string name(void)  { return std::string("tbb::parallel_for");  }
};

template <typename T>
struct tbb_scan_test
{
  typedef typename std::vector<T> Vector;  Vector v;
  void        setup(void) { v.resize(N);  randomize(v); }
  void        run(void)   { tbb_scan(v); }
  std::string name(void)  { return std::string("tbb::parallel_scan");  }
};

template <typename T>
struct tbb_sort_test
{
  typedef typename std::vector<T> Vector;  Vector v;
  void        setup(void) { v.resize(N);  randomize(v); }
  void        run(void)   { tbb_sort(v); }
  std::string name(void)  { return std::string("tbb::parallel_sort");  }
};
#endif

// Thrust tests
template <typename T>
struct thrust_reduce_test
{
  thrust::device_vector<T> v;
  void        setup(void) { v.resize(N);  randomize(v); }
  void        run(void)   { thrust::reduce(v.begin(), v.end()); }
  std::string name(void)  { return std::string("thrust::reduce");  }
};

template <typename T>
struct thrust_transform_test
{
  thrust::device_vector<T> v;
  void        setup(void) { v.resize(N);  randomize(v); }
  void        run(void)   { thrust::transform(v.begin(), v.end(), v.begin(), thrust::negate<int>()); }
  std::string name(void)  { return std::string("thrust::transform");  }
};

template <typename T>
struct thrust_scan_test
{
  thrust::device_vector<T> v;
  void        setup(void) { v.resize(N);  randomize(v); }
  void        run(void)   { thrust::inclusive_scan(v.begin(), v.end(), v.begin()); }
  std::string name(void)  { return std::string("thrust::inclusive_scan");  }
};

template <typename T>
struct thrust_sort_test
{
  thrust::device_vector<T> v;
  void        setup(void) { v.resize(N);  randomize(v); }
  void        run(void)   { thrust::sort(v.begin(), v.end()); }
  std::string name(void)  { return std::string("thrust::sort");  }
};

//////////////////////
// Benchmark Driver //
//////////////////////

template <typename Test>
float rate(Test test)
{
  timer t;

  test.setup();

  t.start();
  test.run();
  t.stop();

  return N / t.seconds_elapsed();
};


template <typename T>
void benchmark_core_primitives(std::string data_type)
{
  printf("Core Primitive Performance for %s (elements per second)\n", data_type.c_str());

#ifdef NO_TBB
  printf("%15s, %12s, %12s, %12s\n", "Algorithm", "STL", "TBB (n/a)", "Thrust");
  printf("%15s, %12.0f, %12.0f, %12.0f\n", "reduce",    rate(stl_reduce_test<T>()),    0.0,  rate(thrust_reduce_test<T>()));
  printf("%15s, %12.0f, %12.0f, %12.0f\n", "transform", rate(stl_transform_test<T>()), 0.0,  rate(thrust_transform_test<T>()));
  printf("%15s, %12.0f, %12.0f, %12.0f\n", "scan",      rate(stl_scan_test<T>()),      0.0,  rate(thrust_scan_test<T>()));
  printf("%15s, %12.0f, %12.0f, %12.0f\n", "sort",      rate(stl_sort_test<T>()),      0.0,  rate(thrust_sort_test<T>()));
#else
  printf("%15s, %12s, %12s, %12s\n", "Algorithm", "STL", "TBB", "Thrust");
  printf("%15s, %12.0f, %12.0f, %12.0f\n", "reduce",    rate(stl_reduce_test<T>()),    rate(tbb_reduce_test<T>()),    rate(thrust_reduce_test<T>()));
  printf("%15s, %12.0f, %12.0f, %12.0f\n", "transform", rate(stl_transform_test<T>()), rate(tbb_transform_test<T>()), rate(thrust_transform_test<T>()));
  printf("%15s, %12.0f, %12.0f, %12.0f\n", "scan",      rate(stl_scan_test<T>()),      rate(tbb_scan_test<T>()),      rate(thrust_scan_test<T>()));
  printf("%15s, %12.0f, %12.0f, %12.0f\n", "sort",      rate(stl_sort_test<T>()),      rate(tbb_sort_test<T>()),      rate(thrust_sort_test<T>()));
#endif

}


int main(void)
{
#ifndef NO_TBB
  tbb::task_scheduler_init init;

  test_tbb();
#endif

  std::cout << "Benchmarking with input size " << N << std::endl;
  benchmark_core_primitives<int>("32-bit integer");
  benchmark_core_primitives<long long>("64-bit integer");
  benchmark_core_primitives<float>("32-bit float");
  benchmark_core_primitives<double>("64-bit float");

  printf("Sorting Performance (keys per second)\n");

#ifdef NO_TBB
  printf("%6s, %12s, %12s, %12s\n", "Type", "STL", "TBB (n/a)", "Thrust");
  printf("%6s, %12.0f, %12.0f, %12.0f\n", "char",   rate(stl_sort_test<char>()),      0.0,  rate(thrust_sort_test<char>()));
  printf("%6s, %12.0f, %12.0f, %12.0f\n", "short",  rate(stl_sort_test<short>()),     0.0,  rate(thrust_sort_test<short>()));
  printf("%6s, %12.0f, %12.0f, %12.0f\n", "int",    rate(stl_sort_test<int>()),       0.0,  rate(thrust_sort_test<int>()));
  printf("%6s, %12.0f, %12.0f, %12.0f\n", "long",   rate(stl_sort_test<long long>()), 0.0,  rate(thrust_sort_test<long long>()));
  printf("%6s, %12.0f, %12.0f, %12.0f\n", "float",  rate(stl_sort_test<float>()),     0.0,  rate(thrust_sort_test<float>()));
  printf("%6s, %12.0f, %12.0f, %12.0f\n", "double", rate(stl_sort_test<double>()),    0.0,  rate(thrust_sort_test<double>()));
#else
  printf("%6s, %12s, %12s, %12s\n", "Type", "STL", "TBB", "Thrust");
  printf("%6s, %12.0f, %12.0f, %12.0f\n", "char",   rate(stl_sort_test<char>()),      rate(tbb_sort_test<char>()),      rate(thrust_sort_test<char>()));
  printf("%6s, %12.0f, %12.0f, %12.0f\n", "short",  rate(stl_sort_test<short>()),     rate(tbb_sort_test<short>()),     rate(thrust_sort_test<short>()));
  printf("%6s, %12.0f, %12.0f, %12.0f\n", "int",    rate(stl_sort_test<int>()),       rate(tbb_sort_test<int>()),       rate(thrust_sort_test<int>()));
  printf("%6s, %12.0f, %12.0f, %12.0f\n", "long",   rate(stl_sort_test<long long>()), rate(tbb_sort_test<long long>()), rate(thrust_sort_test<long long>()));
  printf("%6s, %12.0f, %12.0f, %12.0f\n", "float",  rate(stl_sort_test<float>()),     rate(tbb_sort_test<float>()),     rate(thrust_sort_test<float>()));
  printf("%6s, %12.0f, %12.0f, %12.0f\n", "double", rate(stl_sort_test<double>()),    rate(tbb_sort_test<double>()),    rate(thrust_sort_test<double>()));
#endif

  return 0;
}

