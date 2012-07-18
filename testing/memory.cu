#include <iostream>
#include <unittest/unittest.h>
#include <thrust/sort.h>
#include <thrust/memory.h>
#include <thrust/pair.h>
#include <thrust/fill.h>
#include <thrust/logical.h>


struct my_system : thrust::device_system<my_system> {};


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

  // select_system(my_system, device_system_tag) should return device_system_tag (the minimum tag)
  bool is_device_system_tag = are_same(thrust::device_system_tag(), select_system(my_system(), thrust::device_system_tag()));
  ASSERT_EQUAL(true, is_device_system_tag);

  // select_system(device_system_tag, my_tag) should return device_system_tag (the minimum tag)
  is_device_system_tag = are_same(thrust::device_system_tag(), select_system(thrust::device_system_tag(), my_system()));
  ASSERT_EQUAL(true, is_device_system_tag);
}
DECLARE_UNITTEST(TestSelectSystemDifferentTypes);


void TestSelectSystemSameTypes()
{
  using thrust::system::detail::generic::select_system;

  // select_system(host_system_tag, host_system_tag) should return host_system_tag
  bool is_host_system_tag = are_same(thrust::host_system_tag(), select_system(thrust::host_system_tag(), thrust::host_system_tag()));
  ASSERT_EQUAL(true, is_host_system_tag);

  // select_system(device_system_tag, device_system_tag) should return device_system_tag
  bool is_device_system_tag = are_same(thrust::device_system_tag(), select_system(thrust::device_system_tag(), thrust::device_system_tag()));
  ASSERT_EQUAL(true, is_device_system_tag);

  // select_system(my_system, my_system) should return my_system
  bool is_my_system = are_same(my_system(), select_system(my_system(), my_system()));
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


static bool g_correctly_dispatched;


template<typename T>
  thrust::pair<thrust::pointer<T,my_system>, std::ptrdiff_t>
    get_temporary_buffer(my_system sys, std::ptrdiff_t n)
{
  // communicate that my version of get_temporary_buffer
  // was correctly dispatched
  g_correctly_dispatched = true;

  thrust::device_system_tag device_sys;
  thrust::pair<thrust::pointer<T, thrust::device_system_tag>, std::ptrdiff_t> result = thrust::get_temporary_buffer<T>(device_sys, n);
  return thrust::make_pair(thrust::pointer<T,my_system>(result.first.get()), result.second);
}


void TestGetTemporaryBufferDispatchImplicit()
{
  if(are_same(thrust::device_system_tag(), thrust::system::cpp::tag()))
  {
    // XXX cpp uses the internal scalar backend, which currently elides user tags
    KNOWN_FAILURE;
  }
  else
  {
    g_correctly_dispatched = false;

    thrust::device_vector<int> vec(9001);

    // call something we know will invoke get_temporary_buffer
    my_system sys;
    thrust::sort(sys, vec.begin(), vec.end());

    ASSERT_EQUAL(true, g_correctly_dispatched);
  }
}
DECLARE_UNITTEST(TestGetTemporaryBufferDispatchImplicit);

