#pragma once

#include <thrust/detail/malloc_and_free_adl_helper.h>
#include <thrust/system/detail/generic/select_system.h>
#include <thrust/system/detail/generic/memory.h>

namespace thrust {
namespace detail {
template<typename Tag>
void* tag_malloc(Tag, size_t cnt) {
    using thrust::system::detail::generic::select_system;
    using thrust::system::detail::generic::malloc;
    return malloc(select_system(Tag()), cnt);
}

template<typename Tag>
void tag_free(Tag, void* p) {
    using thrust::system::detail::generic::select_system;
    using thrust::system::detail::generic::free;
    return free(select_system(Tag()), p);
}
}
}
