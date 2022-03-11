#pragma once

#include <iterator>


// Wraps an existing iterator into a forward iterator,
// thus removing some of its functionality
template <typename Iterator>
struct forward_iterator_wrapper {
    // LegacyIterator requirements
    using reference = typename Iterator::reference;
    using pointer = typename Iterator::pointer;
    using value_type = typename Iterator::value_type;
    using difference_type = typename Iterator::difference_type;
    using iterator_category = std::forward_iterator_tag;

    __host__ __device__ reference operator*() const {
        return *wrapped;
    }

    __host__ __device__ forward_iterator_wrapper& operator++() {
        ++wrapped;
        return *this;
    }

    // LegacyInputIterator
    __host__ __device__ bool operator==(const forward_iterator_wrapper& other) {
        return wrapped == other.wrapped;
    }

    __host__ __device__ bool operator!=(const forward_iterator_wrapper& other) {
        return !(*this == other);
    }

    __host__ __device__ forward_iterator_wrapper operator++(int) {
        auto cpy = *this;
        ++(*this);
        return cpy;
    }
    
    __host__ __device__ pointer operator->() const {
        return wrapped.operator->();
    }

    Iterator wrapped;
};


template <typename Iterator>
forward_iterator_wrapper<Iterator> make_forward_iterator_wrapper(Iterator it) {
    return {it};
}
