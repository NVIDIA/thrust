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

#include <stdint.h>

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
  typedef typename std::vector<T> Vector; Vector v;
  void        setup(void) { v.resize(N); randomize(v); }
  void        run(void)   { if (std::accumulate(v.begin(), v.end(), T(0)) == 0) std::cout << "xyz"; } // prevent optimizer from removing body
  std::string name(void)  { return std::string("std::accumulate");  }
};

template <typename T>
struct stl_transform_test
{
  typedef typename std::vector<T> Vector; Vector v;
  void        setup(void) { v.resize(N); randomize(v); }
  void        run(void)   { std::transform(v.begin(), v.end(), v.begin(), thrust::negate<int>()); }
  std::string name(void)  { return std::string("std::transform");  }
};

template <typename T>
struct stl_inclusive_scan_test
{
  typedef typename std::vector<T> Vector; Vector v;
  void        setup(void) { v.resize(N); randomize(v); }
  void        run(void)   { std::partial_sum(v.begin(), v.end(), v.begin()); }
  std::string name(void)  { return std::string("std::partial_sum");  }
};

template <typename T>
struct stl_sort_test
{
  typedef typename std::vector<T> Vector; Vector v;
  void        setup(void) { v.resize(N); randomize(v); }
  void        run(void)   { std::sort(v.begin(), v.end()); }
  std::string name(void)  { return std::string("std::sort");  }
};

#ifndef NO_TBB
// TBB tests
template <typename T>
struct tbb_reduce_test
{
  typedef typename std::vector<T> Vector; Vector v;
  void        setup(void) { v.resize(N); randomize(v); }
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
struct tbb_inclusive_scan_test
{
  typedef typename std::vector<T> Vector; Vector v;
  void        setup(void) { v.resize(N); randomize(v); }
  void        run(void)   { tbb_scan(v); }
  std::string name(void)  { return std::string("tbb::parallel_scan");  }
};

template <typename T>
struct tbb_sort_test
{
  typedef typename std::vector<T> Vector; Vector v;
  void        setup(void) { v.resize(N); randomize(v); }
  void        run(void)   { tbb_sort(v); }
  std::string name(void)  { return std::string("tbb::parallel_sort");  }
};
#endif

// Thrust tests
template <typename T>
struct thrust_reduce_test
{
  thrust::device_vector<T> v;
  void        setup(void) { v.resize(N); randomize(v); }
  void        run(void)   { thrust::reduce(v.begin(), v.end()); }
  std::string name(void)  { return std::string("thrust::reduce");  }
};

template <typename T>
struct thrust_transform_test
{
  thrust::device_vector<T> v;
  void        setup(void) { v.resize(N); randomize(v); }
  void        run(void)   { thrust::transform(v.begin(), v.end(), v.begin(), thrust::negate<int>()); }
  std::string name(void)  { return std::string("thrust::transform");  }
};

template <typename T>
struct thrust_inclusive_scan_test
{
  thrust::device_vector<T> v;
  void        setup(void) { v.resize(N); randomize(v); }
  void        run(void)   { thrust::inclusive_scan(v.begin(), v.end(), v.begin()); }
  std::string name(void)  { return std::string("thrust::inclusive_scan");  }
};

template <typename T>
struct thrust_sort_test
{
  thrust::device_vector<T> v;
  void        setup(void) { v.resize(N); randomize(v); }
  void        run(void)   { thrust::sort(v.begin(), v.end()); }
  std::string name(void)  { return std::string("thrust::sort");  }
};

//////////////////////
// Benchmark Driver //
//////////////////////

template <typename Test>
double rate(Test test)
{
  timer t;

  // Warmup.
  test.setup();
  test.run();

  // Reset for benchmark run.
  test.setup();

  // Benchmark.
  t.start();
  test.run();
  t.stop();

  return N / t.seconds_elapsed();
};


template <typename T>
void benchmark_core_primitives(std::string data_type, size_t input_size)
{
  //printf("Core Primitive Performance for %lu-bit %s (items per second)\n", 8*sizeof(T), data_type.c_str());

  //char const* const header_fmt = "%-15s, %-12s, %-12s, %-12s, %-12s, %-12s, %-12s\n";
  //char const* const entry_fmt  = "%-15s, %-12s, %-12lu, %-12lu, %-12e, %-12e, %-12e\n";
  char const* const header_fmt = "%s,%s,%s,%s,%s,%s,%s\n";
  char const* const entry_fmt  = "%s,%s,%lu,%lu,%e,%e,%e\n";

#ifdef NO_TBB
  //printf(header_fmt, "Algorithm", "Type", "Type Size", "Input Size", "STL", "TBB (n/a)", "Thrust");
  //printf(header_fmt, "", "", "[bits]", "[items]", "[items/sec]", "[items/sec]", "[items/sec]");
  {
    double stl    = rate(stl_reduce_test<T>());
    double thrust = rate(thrust_reduce_test<T>());
    printf(entry_fmt, "reduce",         data_type.c_str(), 8*sizeof(T), input_size, stl, 0.0, thrust);
  }
  {
    double stl    = rate(stl_transform_test<T>());
    double thrust = rate(thrust_transform_test<T>());
    printf(entry_fmt, "transform",      data_type.c_str(), 8*sizeof(T), input_size, stl, 0.0, thrust);
  }
  {
    double stl    = rate(stl_inclusive_scan_test<T>());
    double thrust = rate(thrust_inclusive_scan_test<T>());
    printf(entry_fmt, "inclusive_scan", data_type.c_str(), 8*sizeof(T), input_size, stl, 0.0, thrust);
  }
  {
    double stl    = rate(stl_sort_test<T>());
    double thrust = rate(thrust_sort_test<T>());
    printf(entry_fmt, "sort",           data_type.c_str(), 8*sizeof(T), input_size, stl, 0.0, thrust);
  }
#else
  //printf(header_fmt, "Algorithm", "Type", "Type Size", "Input Size", "STL", "TBB", "Thrust");
  //printf(header_fmt, "", "", "[bits]", "[items]", "[items/sec]", "[items/sec]", "[items/sec]");
  {
    double stl    = rate(stl_reduce_test<T>());
    double tbb    = rate(tbb_reduce_test<T>());
    double thrust = rate(thrust_reduce_test<T>());
    printf(entry_fmt, "reduce",         data_type.c_str(), 8*sizeof(T), input_size, stl, tbb, thrust);
  }
  {
    double stl    = rate(stl_transform_test<T>());
    double tbb    = rate(tbb_transform_test<T>());
    double thrust = rate(thrust_transform_test<T>());
    printf(entry_fmt, "transform",      data_type.c_str(), 8*sizeof(T), input_size, stl, tbb, thrust);
  }
  {
    double stl    = rate(stl_inclusive_scan_test<T>());
    double tbb    = rate(tbb_inclusive_scan_test<T>());
    double thrust = rate(thrust_inclusive_scan_test<T>());
    printf(entry_fmt, "inclusive_scan", data_type.c_str(), 8*sizeof(T), input_size, stl, tbb, thrust);
  }
  {
    double stl    = rate(stl_sort_test<T>());
    double tbb    = rate(tbb_sort_test<T>());
    double thrust = rate(thrust_sort_test<T>());
    printf(entry_fmt, "sort",           data_type.c_str(), 8*sizeof(T), input_size, stl, tbb, thrust);
  }
#endif

}


int main(void)
{
#ifndef NO_TBB
  tbb::task_scheduler_init init;

  test_tbb();
#endif

  std::cout << "Benchmarking with input size " << N << std::endl;
  benchmark_core_primitives<char>   ("char",    N);
  benchmark_core_primitives<int>    ("int",     N);
  benchmark_core_primitives<int8_t> ("integer", N);
  benchmark_core_primitives<int16_t>("integer", N);
  benchmark_core_primitives<int32_t>("integer", N);
  benchmark_core_primitives<int64_t>("integer", N);
  benchmark_core_primitives<float>  ("float",   N);
  benchmark_core_primitives<double> ("float",   N);

  return 0;
}

