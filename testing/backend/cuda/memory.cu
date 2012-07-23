#include <unittest/unittest.h>
#include <thrust/system/cuda/memory.h>
#include <thrust/system/cpp/memory.h>

template<typename T1, typename T2>
bool are_same_type(const T1 &, const T2 &)
{
  return false;
}

template<typename T>
bool are_same_type(const T &, const T &)
{
  return true;
}

void TestSelectSystemCudaToCpp()
{
  using thrust::system::detail::generic::select_system;

  thrust::cuda::tag cuda_tag;
  thrust::cpp::tag cpp_tag;
  thrust::system::cuda::detail::cross_system<thrust::cuda::tag,thrust::cpp::tag> cuda_to_cpp(cuda_tag, cpp_tag);

  // select_system(cuda::tag, thrust::host_system_tag) should return cuda_to_cpp
  bool is_cuda_to_cpp = are_same_type(cuda_to_cpp, select_system(cuda_tag, cpp_tag));
  ASSERT_EQUAL(true, is_cuda_to_cpp);
}
DECLARE_UNITTEST(TestSelectSystemCudaToCpp);

