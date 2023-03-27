#include <thrust/universal_vector.h>

#include <unittest/unittest.h>

template <class VecT>
__host__ __device__ void universal_vector_access(VecT &in, thrust::universal_vector<bool> &out)
{
  const int expected_front  = 4;
  const int expected_back   = 2;

  out[0] = in.size() == 2 &&               //
           in[0] == expected_front &&      //
           in.front() == expected_front && //
           *in.data() == expected_front && //
           in[1] == expected_back &&       //
           in.back() == expected_back;
}

#if defined(THRUST_TEST_DEVICE_SIDE)
template <class VecT>
__global__ void universal_vector_device_access_kernel(VecT &vec,
                                                      thrust::universal_vector<bool> &out)
{
  universal_vector_access(vec, out);
}

template <class VecT>
void test_universal_vector_access(VecT &vec, thrust::universal_vector<bool> &out)
{
  universal_vector_device_access_kernel<<<1, 1>>>(vec, out);
  cudaError_t const err = cudaDeviceSynchronize();
  ASSERT_EQUAL(cudaSuccess, err);
  ASSERT_EQUAL(out[0], true);
}
#else
template <class VecT>
void test_universal_vector_access(VecT &vec, thrust::universal_vector<bool> &out)
{
  universal_vector_access(vec, out);
  ASSERT_EQUAL(out[0], true);
}
#endif

void TestUniversalVectorDeviceAccess()
{
  thrust::universal_vector<thrust::universal_vector<int>> in_storage(1);
  thrust::universal_vector<int> &in = *thrust::raw_pointer_cast(in_storage.data());

  in.resize(2);
  in[0] = 4;
  in[1] = 2;

  thrust::universal_vector<thrust::universal_vector<bool>> out_storage(1);
  thrust::universal_vector<bool> &out = *thrust::raw_pointer_cast(out_storage.data());
  out.resize(1);
  out[0] = false;

  test_universal_vector_access(in, out);
}
DECLARE_UNITTEST(TestUniversalVectorDeviceAccess);

void TestConstUniversalVectorDeviceAccess()
{
  thrust::universal_vector<thrust::universal_vector<int>> in_storage(1);

  {
    thrust::universal_vector<int> &in = *thrust::raw_pointer_cast(in_storage.data());

    in.resize(2);
    in[0] = 4;
    in[1] = 2;
  }

  const thrust::universal_vector<int> &in = *thrust::raw_pointer_cast(in_storage.data());

  thrust::universal_vector<thrust::universal_vector<bool>> out_storage(1);
  thrust::universal_vector<bool> &out = *thrust::raw_pointer_cast(out_storage.data());

  out.resize(1);
  out[0] = false;

  test_universal_vector_access(in, out);
}
DECLARE_UNITTEST(TestConstUniversalVectorDeviceAccess);
