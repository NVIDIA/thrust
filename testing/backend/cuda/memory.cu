#include <unittest/unittest.h>
#include <thrust/system/cuda/memory.h>

template<typename T1, typename T2>
bool is_same(const T1 &, const T2 &)
{
  return false;
}

template<typename T>
bool is_same(const T &, const T &)
{
  return true;
}

void TestSelectSystemCudaToCpp()
{
  using thrust::system::detail::generic::select_system;

  // select_system(cuda::tag, cpp::tag) should return cuda_to_cpp
  bool is_cuda_to_cpp = is_same(thrust::system::cuda::detail::cuda_to_cpp(), select_system(thrust::cuda::tag(), thrust::cpp::tag()));
  ASSERT_EQUAL(true, is_cuda_to_cpp);
}
DECLARE_UNITTEST(TestSelectSystemCudaToCpp);

