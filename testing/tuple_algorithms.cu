#include <thrust/detail/config.h>

#if THRUST_CPP_DIALECT >= 2011

#include <unittest/unittest.h>

#include <thrust/detail/tuple_algorithms.h>

// FIXME: Replace with C++14 style `thrust::square<>` when we have it.
struct custom_square
{
  template <typename T>
  T operator()(T v) const
  {
    return v * v; 
  }
};

void test_tuple_subset()
{
  auto t0 = std::make_tuple(0, 2, 3.14);

  auto t1 = thrust::tuple_subset(t0, std::index_sequence<2, 0>{}); 

  ASSERT_EQUAL_QUIET(t1, std::make_tuple(3.14, 0));
}
DECLARE_UNITTEST(test_tuple_subset);

void test_tuple_transform()
{
  auto t0 = std::make_tuple(0, 2, 3.14);

  auto t1 = thrust::tuple_transform(t0, custom_square{}); 

  ASSERT_EQUAL_QUIET(t1, std::make_tuple(0, 4, 9.8596));
}
DECLARE_UNITTEST(test_tuple_transform);

void test_tuple_for_each()
{
  auto t = std::make_tuple(0, 2, 3.14);

  thrust::tuple_for_each(t, [](auto& x) { x *= x; }); 

  ASSERT_EQUAL_QUIET(t, std::make_tuple(0, 4, 9.8596));
}
DECLARE_UNITTEST(test_tuple_for_each);
 
#endif // THRUST_CPP_DIALECT >= 2011

