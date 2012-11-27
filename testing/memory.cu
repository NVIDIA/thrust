#include <iostream>
#include <unittest/unittest.h>
#include <thrust/memory.h>
#include <thrust/sort.h>
#include <thrust/memory.h>
#include <thrust/pair.h>
#include <thrust/fill.h>
#include <thrust/logical.h>
#include <thrust/sequence.h>
#include <thrust/reverse.h>

template<typename T1, typename T2>
bool are_same(const T1 &, const T2 &)
{
  return false;
}


template<typename T>
bool are_same(const T &, const T &)
{
  return true;
}


void TestSelectSystemDifferentTypes()
{
  using thrust::system::detail::generic::select_system;

  my_system my_sys(0);
  thrust::device_system_tag device_sys;

  // select_system(my_system, device_system_tag) should return device_system_tag (the minimum tag)
  bool is_device_system_tag = are_same(device_sys, select_system(my_sys, device_sys));
  ASSERT_EQUAL(true, is_device_system_tag);

  // select_system(device_system_tag, my_tag) should return device_system_tag (the minimum tag)
  is_device_system_tag = are_same(device_sys, select_system(device_sys, my_sys));
  ASSERT_EQUAL(true, is_device_system_tag);
}
DECLARE_UNITTEST(TestSelectSystemDifferentTypes);


void TestSelectSystemSameTypes()
{
  using thrust::system::detail::generic::select_system;

  my_system my_sys(0);
  thrust::device_system_tag device_sys;
  thrust::host_system_tag host_sys;

  // select_system(host_system_tag, host_system_tag) should return host_system_tag
  bool is_host_system_tag = are_same(host_sys, select_system(host_sys, host_sys));
  ASSERT_EQUAL(true, is_host_system_tag);

  // select_system(device_system_tag, device_system_tag) should return device_system_tag
  bool is_device_system_tag = are_same(device_sys, select_system(device_sys, device_sys));
  ASSERT_EQUAL(true, is_device_system_tag);

  // select_system(my_system, my_system) should return my_system
  bool is_my_system = are_same(my_sys, select_system(my_sys, my_sys));
  ASSERT_EQUAL(true, is_my_system);
}
DECLARE_UNITTEST(TestSelectSystemSameTypes);


void TestGetTemporaryBuffer()
{
  const size_t n = 9001;

  thrust::device_system_tag dev_tag;
  typedef thrust::pointer<int, thrust::device_system_tag> pointer;
  thrust::pair<pointer, std::ptrdiff_t> ptr_and_sz = thrust::get_temporary_buffer<int>(dev_tag, n);

  ASSERT_EQUAL(ptr_and_sz.second, n);

  const int ref_val = 13;
  thrust::device_vector<int> ref(n, ref_val);

  thrust::fill_n(ptr_and_sz.first, n, ref_val);

  ASSERT_EQUAL(true, thrust::all_of(ptr_and_sz.first, ptr_and_sz.first + n, thrust::placeholders::_1 == ref_val));

  thrust::return_temporary_buffer(dev_tag, ptr_and_sz.first);
}
DECLARE_UNITTEST(TestGetTemporaryBuffer);


void TestMalloc()
{
  const size_t n = 9001;

  thrust::device_system_tag dev_tag;
  typedef thrust::pointer<int, thrust::device_system_tag> pointer;
  pointer ptr = pointer(static_cast<int*>(thrust::malloc(dev_tag, sizeof(int) * n).get()));

  const int ref_val = 13;
  thrust::device_vector<int> ref(n, ref_val);

  thrust::fill_n(ptr, n, ref_val);

  ASSERT_EQUAL(true, thrust::all_of(ptr, ptr + n, thrust::placeholders::_1 == ref_val));

  thrust::free(dev_tag, ptr);
}
DECLARE_UNITTEST(TestMalloc);


thrust::pointer<void,my_system>
  malloc(my_system &system, std::size_t)
{
  system.validate_dispatch();

  return thrust::pointer<void,my_system>();
}


void TestMallocDispatchExplicit()
{
  const size_t n = 0;

  my_system sys(0);
  thrust::malloc(sys, n);

  ASSERT_EQUAL(true, sys.is_valid());
}
DECLARE_UNITTEST(TestMallocDispatchExplicit);


template<typename Pointer>
void free(my_system &system, Pointer)
{
  system.validate_dispatch();
}


void TestFreeDispatchExplicit()
{
  thrust::pointer<my_system,void> ptr;

  my_system sys(0);
  thrust::free(sys, ptr);

  ASSERT_EQUAL(true, sys.is_valid());
}
DECLARE_UNITTEST(TestFreeDispatchExplicit);


template<typename T>
  thrust::pair<thrust::pointer<T,my_system>, std::ptrdiff_t>
    get_temporary_buffer(my_system &system, std::ptrdiff_t n)
{
  system.validate_dispatch();

  thrust::device_system_tag device_sys;
  thrust::pair<thrust::pointer<T, thrust::device_system_tag>, std::ptrdiff_t> result = thrust::get_temporary_buffer<T>(device_sys, n);
  return thrust::make_pair(thrust::pointer<T,my_system>(result.first.get()), result.second);
}


void TestGetTemporaryBufferDispatchExplicit()
{
#if defined(THRUST_GCC_VERSION) && (THRUST_GCC_VERSION < 40300)
  // gcc 4.2 does not do adl correctly for get_temporary_buffer
  KNOWN_FAILURE;
#else
  const size_t n = 9001;

  my_system sys(0);
  typedef thrust::pointer<int, thrust::device_system_tag> pointer;
  thrust::pair<pointer, std::ptrdiff_t> ptr_and_sz = thrust::get_temporary_buffer<int>(sys, n);

  ASSERT_EQUAL(ptr_and_sz.second, n);
  ASSERT_EQUAL(true, sys.is_valid());

  const int ref_val = 13;
  thrust::device_vector<int> ref(n, ref_val);

  thrust::fill_n(ptr_and_sz.first, n, ref_val);

  ASSERT_EQUAL(true, thrust::all_of(ptr_and_sz.first, ptr_and_sz.first + n, thrust::placeholders::_1 == ref_val));

  thrust::return_temporary_buffer(sys, ptr_and_sz.first);
#endif
}
DECLARE_UNITTEST(TestGetTemporaryBufferDispatchExplicit);


void TestGetTemporaryBufferDispatchImplicit()
{
  if(are_same(thrust::device_system_tag(), thrust::system::cpp::tag()))
  {
    // XXX cpp uses the internal scalar backend, which currently elides user tags
    KNOWN_FAILURE;
  }
  else
  {
#if defined(THRUST_GCC_VERSION) && (THRUST_GCC_VERSION < 40300)
    // gcc 4.2 does not do adl correctly for get_temporary_buffer
    KNOWN_FAILURE;
#else
    thrust::device_vector<int> vec(9001);

    thrust::sequence(vec.begin(), vec.end());
    thrust::reverse(vec.begin(), vec.end());

    // call something we know will invoke get_temporary_buffer
    my_system sys(0);
    thrust::sort(sys, vec.begin(), vec.end());

    ASSERT_EQUAL(true, thrust::is_sorted(vec.begin(), vec.end()));
    ASSERT_EQUAL(true, sys.is_valid());
#endif
  }
}
DECLARE_UNITTEST(TestGetTemporaryBufferDispatchImplicit);

