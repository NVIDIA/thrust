#pragma once

#include <thrust/iterator/iterator_traits.h>
#include <thrust/iterator/iterator_categories.h>
#include <type_traits>


// Wraps an existing iterator into a forward iterator,
// thus removing some of its functionality
template <typename Iterator>
struct forward_iterator_wrapper {
    // LegacyIterator requirements
    using iterator_system_tag = typename thrust::iterator_system<Iterator>::type;
    using reference = typename thrust::iterator_traits<Iterator>::reference;
    using pointer = typename thrust::iterator_traits<Iterator>::pointer;
    using value_type = typename thrust::iterator_traits<Iterator>::value_type;
    using difference_type = typename thrust::iterator_traits<Iterator>::difference_type;
    using iterator_category = typename std::conditional<
        std::is_convertible<iterator_system_tag, thrust::device_system_tag>::value,
        thrust::forward_device_iterator_tag,
        typename std::conditional<
            std::is_convertible<iterator_system_tag, thrust::host_system_tag>::value,
            thrust::forward_host_iterator_tag,
            std::forward_iterator_tag>::type>::type;
    using base_iterator_category = typename thrust::iterator_traits<Iterator>::iterator_category;
    static_assert(
        std::is_convertible<base_iterator_category, std::forward_iterator_tag>::value,
        "Cannot create forward_iterator_wrapper around an iterator that is not itself at least a forward iterator");

    __host__ __device__ reference operator*() const {
        return *wrapped;
    }

    __host__ __device__ forward_iterator_wrapper& operator++() {
        ++wrapped;
        return *this;
    }

    // LegacyInputIterator
    friend __host__ __device__ bool operator==(const forward_iterator_wrapper& a, const forward_iterator_wrapper& b) {
        return a.wrapped == b.wrapped;
    }

    friend __host__ __device__ bool operator!=(const forward_iterator_wrapper& a, const forward_iterator_wrapper& b) {
        return !(a == b);
    }

    __host__ __device__ forward_iterator_wrapper operator++(int) {
        auto cpy = *this;
        ++(*this);
        return cpy;
    }

    template <typename It = Iterator>
    __host__ __device__ typename std::enable_if<std::is_pointer<It>::value, pointer>::type operator->() const {
        return wrapped;
    }

    template <typename It = Iterator>
    __host__ __device__ typename std::enable_if<!std::is_pointer<It>::value, pointer>::type operator->() const {
        return wrapped.operator->();
    }

    Iterator wrapped;
};


template <typename Iterator>
forward_iterator_wrapper<Iterator> make_forward_iterator_wrapper(Iterator it) {
    return {it};
}
