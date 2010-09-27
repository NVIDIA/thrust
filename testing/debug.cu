#define THRUST_DEBUG 1

#include <unittest/unittest.h>
#include <thrust/device_ptr.h>
#include <thrust/transform.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/system_error.h>

#if THRUST_DEVICE_BACKEND == THRUST_DEVICE_BACKEND_CUDA

void TestTransformNullPtr(void)
{
#if defined(__APPLE__)
  KNOWN_FAILURE;
#endif

  thrust::device_ptr<int> ptr = thrust::device_pointer_cast<int>(0);

  bool caught_exception = false;

  try
  {
    thrust::transform(ptr,ptr+1,ptr,thrust::identity<int>());
  }
  catch(thrust::system_error e)
  {
    caught_exception = true;

    // kill the context so it can revive later
    cudaThreadExit();
  }

  ASSERT_EQUAL(true,caught_exception);
}
DECLARE_UNITTEST(TestTransformNullPtr);

void TestReduceNullPtr(void)
{
#if defined(__APPLE__)
  KNOWN_FAILURE;
#endif

  thrust::device_ptr<int> ptr = thrust::device_pointer_cast<int>(0);

  bool caught_exception = false;

  try
  {
    thrust::reduce(ptr,ptr+1);
  }
  catch(thrust::system_error e)
  {
    caught_exception = true;

    // kill the context so it can revive later
    cudaThreadExit();
  }

  ASSERT_EQUAL(true,caught_exception);
}
DECLARE_UNITTEST(TestReduceNullPtr);

void TestExclusiveScanNullPtr(void)
{
#if defined(__APPLE__)
  KNOWN_FAILURE;
#endif

  thrust::device_ptr<int> ptr = thrust::device_pointer_cast<int>(0);

  bool caught_exception = false;

  try
  {
    thrust::exclusive_scan(ptr,ptr+1,ptr);
  }
  catch(thrust::system_error e)
  {
    caught_exception = true;

    // kill the context so it can revive later
    cudaThreadExit();
  }

  ASSERT_EQUAL(true, caught_exception);
}
DECLARE_UNITTEST(TestExclusiveScanNullPtr);

void TestSortNullPtr(void)
{
  // XXX sort(null) below just crashes
  KNOWN_FAILURE;
//  thrust::device_ptr<int> ptr = thrust::device_pointer_cast<int>(0);
//
//  bool caught_exception = false;
//
//  try
//  {
//    thrust::sort(ptr,ptr+1);
//  }
//  catch(thrust::system_error e)
//  {
//    caught_exception = true;
//
//    // reset the cuda error
//    cudaGetLastError();
//  }
//
//  ASSERT_EQUAL(true, caught_exception);
}
DECLARE_UNITTEST(TestSortNullPtr);

#endif

