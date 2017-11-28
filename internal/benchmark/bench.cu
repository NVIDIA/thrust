#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>

#include <utility>
#include <algorithm>
#include <numeric>

#include <iostream>
#include <iomanip>
#include <cstdlib>

#include <stdint.h>
#include <math.h>

#include "random.h"
#include "timer.h"

#ifndef NO_TBB
#include "tbb_algos.h"
#endif

//////////////////////
// Test Definitions //
//////////////////////

template <typename Derived>
struct test_base
{
  Derived& derived()
  {
    return static_cast<Derived&>(*this);
  }

  void setup(size_t n)
  {
    derived().v.resize(n);
    randomize(derived().v);
  }
};

template <typename T>
struct stl_reduce_test : test_base<stl_reduce_test<T> >
{
  std::vector<T> v;

  void run()
  {
    if(std::accumulate(v.begin(), v.end(), T(0)) == 0)
      // Prevent optimizer from removing body.
      std::cout << "xyz";
  }
};

template <typename T>
struct stl_transform_test : test_base<stl_transform_test<T> >
{
  std::vector<T> v;

  void run() { std::transform(v.begin(), v.end(), v.begin(), thrust::negate<int>()); }
};

template <typename T>
struct stl_inclusive_scan_test : test_base<stl_inclusive_scan_test<T> >
{
  std::vector<T> v;

  void run() { std::partial_sum(v.begin(), v.end(), v.begin()); }
};

template <typename T>
struct stl_sort_test : test_base<stl_sort_test<T> >
{
  std::vector<T> v;

  void run() { std::sort(v.begin(), v.end()); }
};

#ifndef NO_TBB
template <typename T>
struct tbb_reduce_test : test_base<tbb_reduce_test<T> >
{
  std::vector<T> v;

  void run() { tbb_reduce(v); }
};

template <typename T>
struct tbb_transform_test : test_base<tbb_transform_test<T> >
{
  std::vector<T> v;

  void run() { tbb_transform(v); }
};

template <typename T>
struct tbb_inclusive_scan_test : test_base<tbb_inclusive_scan_test<T> >
{
  std::vector<T> v;

  void run() { tbb_scan(v); }
};

template <typename T>
struct tbb_sort_test : test_base<tbb_sort_test<T> >
{
  std::vector<T> v;

  void run() { tbb_sort(v); }
};
#endif

template <typename T>
struct thrust_reduce_test : test_base<thrust_reduce_test<T> >
{
  thrust::device_vector<T> v;

  void run() { thrust::reduce(v.begin(), v.end()); }
};

template <typename T>
struct thrust_transform_test : test_base<thrust_transform_test<T> >
{
  thrust::device_vector<T> v;

  void run() { thrust::transform(v.begin(), v.end(), v.begin(), thrust::negate<int>()); }
};

template <typename T>
struct thrust_inclusive_scan_test : test_base<thrust_inclusive_scan_test<T> >
{
  thrust::device_vector<T> v;

  void run() { thrust::inclusive_scan(v.begin(), v.end(), v.begin()); }
};

template <typename T>
struct thrust_sort_test : test_base<thrust_sort_test<T> >
{
  thrust::device_vector<T> v;

  void run() { thrust::sort(v.begin(), v.end()); }
};

//////////////////////
// Benchmark Driver //
//////////////////////

template <typename T>
struct squared_difference
{
private:
  T const average;
public:
  __host__ __device__
  squared_difference(T average_) : average(average_) {}

  __host__ __device__
  squared_difference(squared_difference const& rhs) : average(rhs.average) {}

  __host__ __device__
  double operator() (double x)
  {
    return (x - average) * (x - average);
  }
};

template <typename Test>
std::pair<double, double> rate(Test test, size_t trials, size_t input_size)
{
  // Warmup.
  test.setup(input_size);
  test.run();

  std::vector<double> times;
  times.reserve(trials);

  for(size_t t = 0; t < trials; ++t)
  {
    // Reset for next run. 
    test.setup(input_size);

    // Benchmark.
    timer e;

    e.start();
    test.run();
    e.stop();

    times.push_back(e.seconds_elapsed());
  }

  //for(size_t t = 0; t < trials; ++t)
  //  printf("%e\n", times[t]);

  // Arithmetic mean.
  double time_average =
    std::accumulate(times.begin(), times.end(), double(0.0)) / trials;

  //printf("MEAN: %e\n", time_average);

  // Sample standard deviation.
  double time_stdev = 
    sqrt(  1.0 / double(trials - 1)
         * thrust::transform_reduce(times.begin(), times.end(),
                                    squared_difference<double>(time_average),
                                    double(0.0),
                                    thrust::plus<double>())
    );

  //printf("STDEV: %e\n", time_stdev);

  return std::pair<double, double>(time_average, time_stdev); 
};

template <typename T>
void benchmark_core_primitives(std::string data_type, size_t trials, size_t input_size)
{
#ifdef NO_TBB
  char const* const header_fmt = "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n";
  char const* const entry_fmt  = "%lu,%s,%s,%lu,%lu,%lu,%e,%e,%e,%e\n";

  printf(header_fmt, "Version", "Algorithm", "Type", "Type Size", "Trials", "Input Size", "STL Average", "STL Sample Standard Deviation", "Thrust Average", "Thrust Sample Standard Deviation");
  printf(header_fmt, "", "", "", "[bits]", "[trials]", "[items]", "[items/sec]", "[items/sec]", "[items/sec]", "[items/sec]");
  {
    std::pair<double, double> stl    = rate(stl_reduce_test<T>(), trials, input_size);
    std::pair<double, double> thrust = rate(thrust_reduce_test<T>(), trials, input_size);
    printf(entry_fmt, THRUST_VERSION, "reduce",         data_type.c_str(), 8*sizeof(T), trials, input_size, stl.first, stl.second, thrust.first, thrust.second);
  }
  {
    std::pair<double, double> stl    = rate(stl_transform_test<T>(), trials, input_size);
    std::pair<double, double> thrust = rate(thrust_transform_test<T>(), trials, input_size);
    printf(entry_fmt, THRUST_VERSION, "transform",      data_type.c_str(), 8*sizeof(T), trials, input_size, stl.first, stl.second, thrust.first, thrust.second);
  }
  {
    std::pair<double, double> stl    = rate(stl_inclusive_scan_test<T>(), trials, input_size);
    std::pair<double, double> thrust = rate(thrust_inclusive_scan_test<T>(), trials, input_size);
    printf(entry_fmt, THRUST_VERSION, "inclusive_scan", data_type.c_str(), 8*sizeof(T), trials, input_size, stl.first, stl.second, thrust.first, thrust.second);
  }
  {
    std::pair<double, double> stl    = rate(stl_sort_test<T>(), trials, input_size);
    std::pair<double, double> thrust = rate(thrust_sort_test<T>(), trials, input_size);
    printf(entry_fmt, THRUST_VERSION, "sort",           data_type.c_str(), 8*sizeof(T), trials, input_size, stl.first, stl.second, thrust.first, thrust.second);
  }
#else
  char const* const header_fmt = "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n";
  char const* const entry_fmt  = "%lu,%s,%s,%lu,%lu,%lu,%e,%e,%e,%e,%e,%e\n";

  printf(header_fmt, "Version", "Algorithm", "Type", "Type Size", "Trials", "Input Size", "STL Average", "STL Sample Standard Deviation", "Thrust Average", "Thrust Sample Standard Deviation", "TBB Average", "TBB Sample Standard Deviation");
  printf(header_fmt, "", "", "", "[bits]", "[trials]", "[items]", "[items/sec]", "[items/sec]", "[items/sec]", "[items/sec]", "[items/sec]", "[items/sec]");
  {
    std::pair<double, double> stl    = rate(stl_reduce_test<T>(), trials, input_size);
    std::pair<double, double> thrust = rate(thrust_reduce_test<T>(), trials, input_size);
    std::pair<double, double> tbb    = rate(tbb_reduce_test<T>(), trials, input_size);
    printf(entry_fmt, THRUST_VERSION, "reduce",         data_type.c_str(), 8*sizeof(T), trials, input_size, stl.first, stl.second, thrust.first, thrust.second, tbb.first, tbb.second);
  }
  {
    std::pair<double, double> stl    = rate(stl_transform_test<T>(), trials, input_size);
    std::pair<double, double> thrust = rate(thrust_transform_test<T>(), trials, input_size);
    std::pair<double, double> tbb    = rate(tbb_transform_test<T>(), trials, input_size);
    printf(entry_fmt, THRUST_VERSION, "transform",      data_type.c_str(), 8*sizeof(T), trials, input_size, stl.first, stl.second, thrust.first, thrust.second, tbb.first, tbb.second);
  }
  {
    std::pair<double, double> stl    = rate(stl_inclusive_scan_test<T>(), trials, input_size);
    std::pair<double, double> thrust = rate(thrust_inclusive_scan_test<T>(), trials, input_size);
    std::pair<double, double> tbb    = rate(tbb_inclusive_scan_test<T>(), trials, input_size);
    printf(entry_fmt, THRUST_VERSION, "inclusive_scan", data_type.c_str(), 8*sizeof(T), trials, input_size, stl.first, stl.second, thrust.first, thrust.second, tbb.first, tbb.second);
  }
  {
    std::pair<double, double> stl    = rate(stl_sort_test<T>(), trials, input_size);
    std::pair<double, double> thrust = rate(thrust_sort_test<T>(), trials, input_size);
    std::pair<double, double> tbb    = rate(tbb_sort_test<T>(), trials, input_size);
    printf(entry_fmt, THRUST_VERSION, "sort",           data_type.c_str(), 8*sizeof(T), trials, input_size, stl.first, stl.second, thrust.first, thrust.second, tbb.first, tbb.second);
  }
#endif

}

int main()
{
#ifndef NO_TBB
  tbb::task_scheduler_init init;

  test_tbb();
#endif

  size_t trials = 8;
  
  size_t input_size = 32 << 20;

  benchmark_core_primitives<char>   ("char",    trials, input_size);
  benchmark_core_primitives<int>    ("int",     trials, input_size);
  benchmark_core_primitives<int8_t> ("integer", trials, input_size);
  benchmark_core_primitives<int16_t>("integer", trials, input_size);
  benchmark_core_primitives<int32_t>("integer", trials, input_size);
  benchmark_core_primitives<int64_t>("integer", trials, input_size);
  benchmark_core_primitives<float>  ("float",   trials, input_size);
  benchmark_core_primitives<double> ("float",   trials, input_size);

  return 0;
}

