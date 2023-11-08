#include <thrust/iterator/detail/iterator_category_with_system_and_traversal.h>
#include <thrust/iterator/iterator_categories.h>
#include <thrust/iterator/iterator_traits.h>

template <class IteratorCategory>
using _category_to_system_t =
    typename thrust::detail::iterator_category_to_system<IteratorCategory>::type;

static_assert(std::is_same<thrust::device_system_tag, _category_to_system_t<thrust::output_device_iterator_tag>>::value, "");
static_assert(std::is_same<thrust::device_system_tag, _category_to_system_t<thrust::input_device_iterator_tag>>::value, "");
// static_assert(std::is_same<thrust::device_system_tag, _category_to_system_t<thrust::forward_device_iterator_tag>>::value, "");       // Still broken
// static_assert(std::is_same<thrust::device_system_tag, _category_to_system_t<thrust::bidirectional_device_iterator_tag>>::value, ""); // Still broken
// static_assert(std::is_same<thrust::device_system_tag, _category_to_system_t<thrust::random_access_device_iterator_tag>>::value, ""); // Still broken

static_assert(std::is_same<thrust::host_system_tag, _category_to_system_t<thrust::output_host_iterator_tag>>::value, "");
static_assert(std::is_same<thrust::host_system_tag, _category_to_system_t<thrust::input_host_iterator_tag>>::value, "");
static_assert(std::is_same<thrust::host_system_tag, _category_to_system_t<thrust::forward_host_iterator_tag>>::value, "");
static_assert(std::is_same<thrust::host_system_tag, _category_to_system_t<thrust::bidirectional_host_iterator_tag>>::value, "");
static_assert(std::is_same<thrust::host_system_tag, _category_to_system_t<thrust::random_access_host_iterator_tag>>::value, "");

// static_assert(!std::is_convertible<thrust::input_device_iterator_tag, thrust::input_host_iterator_tag>::value, "");                  // Still broken
static_assert(!std::is_convertible<thrust::input_host_iterator_tag, thrust::input_device_iterator_tag>::value, "");
