#include <thrust/detail/config.h>

#if THRUST_CPP_DIALECT >= 2011

#  include <unittest/unittest.h>

#  include <thrust/allocate_unique.h>
#  include <thrust/memory/detail/device_system_resource.h>
#  include <thrust/mr/allocator.h>
#  include <thrust/type_traits/is_contiguous_iterator.h>

#  include <numeric>
#  include <vector>

namespace
{

template <typename T>
using allocator =
  thrust::mr::stateless_resource_allocator<T, thrust::universal_memory_resource>;

// The managed_memory_pointer class should be identified as a
// contiguous_iterator
THRUST_STATIC_ASSERT(
  thrust::is_contiguous_iterator<allocator<int>::pointer>::value);

template <typename T>
struct some_object {
  some_object(T data)
      : m_data(data)
  {}

  void setter(T data) { m_data = data; }
  T getter() const { return m_data; }

private:
  T m_data;
};

} // namespace

template <typename T>
void TestAllocateUnique()
{
  // Simple test to ensure that pointers created with universal_memory_resource
  // can be dereferenced and used with STL code. This is necessary as some
  // STL implementations break when using fancy references that overload
  // `operator&`, so universal_memory_resource uses a special pointer type that
  // returns regular C++ references that can be safely used host-side.

  // These operations fail to compile with fancy references:
  auto pRaw = thrust::allocate_unique<T>(allocator<T>{}, 42);
  auto pObj =
    thrust::allocate_unique<some_object<T> >(allocator<some_object<T> >{}, 42);

  static_assert(
    std::is_same<decltype(pRaw.get()),
                 thrust::system::cuda::detail::managed_memory_pointer<T> >::value,
    "Unexpected pointer returned from unique_ptr::get.");
  static_assert(
    std::is_same<decltype(pObj.get()),
                 thrust::system::cuda::detail::managed_memory_pointer<
                   some_object<T> > >::value,
    "Unexpected pointer returned from unique_ptr::get.");

  ASSERT_EQUAL(*pRaw, T(42));
  ASSERT_EQUAL(*pRaw.get(), T(42));
  ASSERT_EQUAL(pObj->getter(), T(42));
  ASSERT_EQUAL((*pObj).getter(), T(42));
  ASSERT_EQUAL(pObj.get()->getter(), T(42));
  ASSERT_EQUAL((*pObj.get()).getter(), T(42));
}
DECLARE_GENERIC_UNITTEST(TestAllocateUnique);

template <typename T>
void TestIterationRaw()
{
  auto array = thrust::allocate_unique_n<T>(allocator<T>{}, 6, 42);

  static_assert(
    std::is_same<decltype(array.get()),
                 thrust::system::cuda::detail::managed_memory_pointer<T> >::value,
    "Unexpected pointer returned from unique_ptr::get.");

  for (auto iter = array.get(), end = array.get() + 6; iter < end; ++iter)
  {
    ASSERT_EQUAL(*iter, T(42));
    ASSERT_EQUAL(*iter.get(), T(42));
  }
}
DECLARE_GENERIC_UNITTEST(TestIterationRaw);

template <typename T>
void TestIterationObj()
{
  auto array =
    thrust::allocate_unique_n<some_object<T> >(allocator<some_object<T> >{},
                                               6,
                                               42);

  static_assert(
    std::is_same<decltype(array.get()),
                 thrust::system::cuda::detail::managed_memory_pointer<
                   some_object<T> > >::value,
    "Unexpected pointer returned from unique_ptr::get.");

  for (auto iter = array.get(), end = array.get() + 6; iter < end; ++iter)
  {
    ASSERT_EQUAL(iter->getter(), T(42));
    ASSERT_EQUAL((*iter).getter(), T(42));
    ASSERT_EQUAL(iter.get()->getter(), T(42));
    ASSERT_EQUAL((*iter.get()).getter(), T(42));
  }
}
DECLARE_GENERIC_UNITTEST(TestIterationObj);

template <typename T>
void TestStdVector()
{
  // Verify that a std::vector using the universal allocator will work with
  // STL algorithms.
  std::vector<T, allocator<T> > v0;

  static_assert(
    std::is_same<typename std::decay<decltype(v0)>::type::pointer,
                 thrust::system::cuda::detail::managed_memory_pointer<
                   T > >::value,
    "Unexpected pointer returned from unique_ptr::get.");

  v0.resize(6);
  std::iota(v0.begin(), v0.end(), 0);
  ASSERT_EQUAL(v0[0], T(0));
  ASSERT_EQUAL(v0[1], T(1));
  ASSERT_EQUAL(v0[2], T(2));
  ASSERT_EQUAL(v0[3], T(3));
  ASSERT_EQUAL(v0[4], T(4));
  ASSERT_EQUAL(v0[5], T(5));
}
DECLARE_GENERIC_UNITTEST(TestStdVector);

#endif // C++11
