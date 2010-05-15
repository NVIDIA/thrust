#include <unittest/unittest.h>
#include <iterator>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/detail/static_assert.h>
#include <thrust/detail/type_traits.h>


template<typename T, typename P0, typename P1>
struct converts_to_before
{
  static char test(P0);

  static int test(P1);

  static const bool value = sizeof(test(T())) == sizeof(test(P0()));
};


void TestRandomAccessUniversalIteratorTagConversionPriorities(void)
{
#if THRUST_HOST_COMPILER == THRUST_HOST_COMPILER_GCC
  KNOWN_FAILURE
#else
  // random_access_universal_iterator_tag should convert to
  // random_access_iterator_tag
  // before
  // input_iterator_tag
  THRUST_STATIC_ASSERT( (converts_to_before<
                           thrust::random_access_universal_iterator_tag,
                           std::random_access_iterator_tag,
                           std::input_iterator_tag
                         >::value) );

  // random_access_universal_iterator_tag should convert to
  // random_access_host_iterator_tag
  // before
  // input_iterator_tag
  THRUST_STATIC_ASSERT( (converts_to_before<
                           thrust::random_access_universal_iterator_tag,
                           thrust::random_access_host_iterator_tag,
                           std::input_iterator_tag
                         >::value) );

  // random_access_universal_iterator_tag should convert to
  // random_access_device_iterator_tag
  // before
  // input_iterator_tag
  THRUST_STATIC_ASSERT( (converts_to_before<
                           thrust::random_access_universal_iterator_tag,
                           thrust::random_access_device_iterator_tag,
                           std::input_iterator_tag
                         >::value) );


  // random_access_universal_iterator_tag should convert to
  // random_access_host_iterator_tag before
  // bidirectional_universal_iterator_tag
  THRUST_STATIC_ASSERT( (converts_to_before<
                           thrust::random_access_universal_iterator_tag,
                           thrust::random_access_host_iterator_tag,
                           thrust::bidirectional_universal_iterator_tag
                         >::value) );
#endif
}
DECLARE_UNITTEST(TestRandomAccessUniversalIteratorTagConversionPriorities);


void TestRandomAccessUniversalIteratorTagConversion(void)
{
  // random_access_universal_iterator_tag should convert to any tag
  // derived from input

  // STL tags

  // random_access_universal_iterator_tag should convert to
  // input_iterator_tag
  THRUST_STATIC_ASSERT( (thrust::detail::is_convertible<
                           thrust::random_access_universal_iterator_tag,
                           std::input_iterator_tag
                         >::value) );

  // random_access_universal_iterator_tag should convert to
  // forward_iterator_tag
  THRUST_STATIC_ASSERT( (thrust::detail::is_convertible<
                           thrust::random_access_universal_iterator_tag,
                           std::forward_iterator_tag
                         >::value) );

  // random_access_universal_iterator_tag should convert to
  // bidirectional_iterator_tag
  THRUST_STATIC_ASSERT( (thrust::detail::is_convertible<
                           thrust::random_access_universal_iterator_tag,
                           std::bidirectional_iterator_tag
                         >::value) );

  // random_access_universal_iterator_tag should convert to
  // random_access_iterator_tag
  THRUST_STATIC_ASSERT( (thrust::detail::is_convertible<
                           thrust::random_access_universal_iterator_tag,
                           std::random_access_iterator_tag
                         >::value) );



  // host tags

  // random_access_universal_iterator_tag should convert to
  // input_host_iterator_tag
  THRUST_STATIC_ASSERT( (thrust::detail::is_convertible<
                           thrust::random_access_universal_iterator_tag,
                           thrust::input_host_iterator_tag
                         >::value) );

  // random_access_universal_iterator_tag should convert to
  // forward_host_iterator_tag
  THRUST_STATIC_ASSERT( (thrust::detail::is_convertible<
                           thrust::random_access_universal_iterator_tag,
                           thrust::forward_host_iterator_tag
                         >::value) );

  // random_access_universal_iterator_tag should convert to
  // bidirectional_host_iterator_tag
  THRUST_STATIC_ASSERT( (thrust::detail::is_convertible<
                           thrust::random_access_universal_iterator_tag,
                           thrust::bidirectional_host_iterator_tag
                         >::value) );

  // random_access_universal_iterator_tag should convert to
  // random_access_host_iterator_tag
  THRUST_STATIC_ASSERT( (thrust::detail::is_convertible<
                           thrust::random_access_universal_iterator_tag,
                           thrust::random_access_host_iterator_tag
                         >::value) );



  // device tags

  // random_access_universal_iterator_tag should convert to
  // input_device_iterator_tag
  THRUST_STATIC_ASSERT( (thrust::detail::is_convertible<
                           thrust::random_access_universal_iterator_tag,
                           thrust::input_device_iterator_tag
                         >::value) );

  // random_access_universal_iterator_tag should convert to
  // forward_device_iterator_tag
  THRUST_STATIC_ASSERT( (thrust::detail::is_convertible<
                           thrust::random_access_universal_iterator_tag,
                           thrust::forward_device_iterator_tag
                         >::value) );

  // random_access_universal_iterator_tag should convert to
  // bidirectional_device_iterator_tag
  THRUST_STATIC_ASSERT( (thrust::detail::is_convertible<
                           thrust::random_access_universal_iterator_tag,
                           thrust::bidirectional_device_iterator_tag
                         >::value) );

  // random_access_universal_iterator_tag should convert to
  // random_access_device_iterator_tag
  THRUST_STATIC_ASSERT( (thrust::detail::is_convertible<
                           thrust::random_access_universal_iterator_tag,
                           thrust::random_access_device_iterator_tag
                         >::value) );


  // universal tags
  // random_access_universal_iterator_tag should convert to
  // input_universal_iterator_tag
  THRUST_STATIC_ASSERT( (thrust::detail::is_convertible<
                           thrust::random_access_universal_iterator_tag,
                           thrust::input_universal_iterator_tag
                         >::value) );

  // random_access_universal_iterator_tag should convert to
  // forward_universal_iterator_tag
  THRUST_STATIC_ASSERT( (thrust::detail::is_convertible<
                           thrust::random_access_universal_iterator_tag,
                           thrust::forward_universal_iterator_tag
                         >::value) );

  // random_access_universal_iterator_tag should convert to
  // bidirectional_universal_iterator_tag
  THRUST_STATIC_ASSERT( (thrust::detail::is_convertible<
                           thrust::random_access_universal_iterator_tag,
                           thrust::bidirectional_universal_iterator_tag
                         >::value) );

  // random_access_universal_iterator_tag should convert to
  // random_access_universal_iterator_tag
  THRUST_STATIC_ASSERT( (thrust::detail::is_convertible<
                           thrust::random_access_universal_iterator_tag,
                           thrust::random_access_universal_iterator_tag
                         >::value) );
}
DECLARE_UNITTEST(TestRandomAccessUniversalIteratorTagConversion);

