#pragma once

#include <unittest/testframework.h>
#include <thrust/system/cuda/memory.h>
#include <vector>

class CUDATestDriver
  : public UnitTestDriver
{
  std::vector<int> target_devices(const ArgumentMap &kwargs);

  bool check_cuda_error(bool concise);

  virtual bool post_test_sanity_check(const UnitTest &test, bool concise);

  virtual bool run_tests(const ArgumentSet &args, const ArgumentMap &kwargs);
};

UnitTestDriver &driver_instance(thrust::system::cuda::tag);

