# Thrust v1.9.6  (CUDA 10.1 Update 2) #

## Summary

Thrust v1.9.6 is a minor release accompanying the CUDA 10.1 Update 2 release.

## Bug Fixes

- NVBug 2509847 Inconsistent alignment of `thrust::complex`
- NVBug 2586774 Compilation failure with Clang + older libstdc++ that doesn't
    have `std::is_trivially_copyable`
- NVBug 200488234 CUDA header files contain unicode characters which leads
    compiling errors on Windows
- #949, #973, NVBug 2422333, NVBug 2522259, NVBug 2528822 `thrust::detail::aligned_reinterpret_cast`
    must be annotated with __host__ __device__
- NVBug 2599629 Missing include in the OpenMP sort implementation
- NVBug 200513211 Truncation warning in test code under VC142

# Thrust v1.9.5  (CUDA 10.1 Update 1)

## Summary
 
Thrust 1.9.5 is a minor release accompanying the CUDA 10.1 Update 1 release.

## Bug Fixes

- NVBug 2502854: Fixed assignment of
    `thrust::device_vector<thrust::complex<T>>` between host and device.

# Thrust 1.9.4 (CUDA 10.1)

## Summary

Thrust 1.9.4 adds asynchronous interfaces for parallel algorithms, a new
  allocator system including caching allocators and unified memory support, as
  well as a variety of other enhancements, mostly related to
  C++11/C++14/C++17/C++20 support.
The new asynchronous algorithms in the `thrust::async` namespace return
  `thrust::event` or `thrust::future` objects, which can be waited upon to
  synchronize with the completion of the parallel operation.

## Breaking Changes

Synchronous Thrust algorithms now block until all of their operations have
  completed.
Use the new asynchronous Thrust algorithms for non-blocking behavior.

## New Features

- `thrust::event` and `thrust::future<T>`, uniquely-owned asynchronous handles
    consisting of a state (ready or not ready), content (some value; for
    `thrust::future` only), and an optional set of objects that should be
    destroyed only when the future's value is ready and has been consumed.
  - The design is loosely based on C++11's `std::future`.
  - They can be `.wait`'d on, and the value of a future can be waited on and
      retrieved with `.get` or `.extract`.
  - Multiple `thrust::event`s and `thrust::future`s can be combined with
      `thrust::when_all`.
  - `thrust::future`s can be converted to `thrust::event`s.
  - Currently, these primitives are only implemented for the CUDA backend and
      are C++11 only.
- New asynchronous algorithms that return `thrust::event`/`thrust::future`s,
    implemented as C++20 range style customization points:
    - `thrust::async::reduce`.
    - `thrust::async::reduce_into`, which takes a target location to store the
        reduction result into.
    - `thrust::async::copy`, including a two-policy overload that allows
        explicit cross system copies which execution policy properties can be
        attached to.
    - `thrust::async::transform`.
    - `thrust::async::for_each`.
    - `thrust::async::stable_sort`.
    - `thrust::async::sort`.
    - By default the asynchronous algorithms use the new caching allocators.
        Deallocation of temporary storage is deferred until the destruction of
        the returned `thrust::future`. The content of `thrust::future`s is
        stored in either device or universal memory and transferred to the host
        only upon request to prevent unnecessary data migration.
    - Asynchronous algorithms are currently only implemented for the CUDA
        system and are C++11 only.
- `exec.after(f, g, ...)`, a new execution policy method that takes a set of
    `thrust::event`/`thrust::future`s and returns an execution policy that
    operations on that execution policy should depend upon. 
- New logic and mindset for the type requirements for cross-system sequence
    copies (currently only used by `thrust::async::copy`), based on:
  - `thrust::is_contiguous_iterator` and `THRUST_PROCLAIM_CONTIGUOUS_ITERATOR`
      for detecting/indicating that an iterator points to contiguous storage.
  - `thrust::is_trivially_relocatable` and
      `THRUST_PROCLAIM_TRIVIALLY_RELOCATABLE` for detecting/indicating that a
      type is `memcpy`able (based on principles from
      [P1144](https://wg21.link/P1144)).
  - The new approach reduces buffering, increases performance, and increases
      correctness.
  - The fast path is now enabled when copying CUDA `__half` and vector types with
      `thrust::async::copy`.
- All Thrust synchronous algorithms for the CUDA backend now actually
    synchronize. Previously, any algorithm that did not allocate temporary
    storage (counterexample: `thrust::sort`) and did not have a
    computation-dependent result (counterexample: `thrust::reduce`) would
    actually be launched asynchronously. Additionally, synchronous algorithms
    that allocated temporary storage would become asynchronous if a custom
    allocator was supplied that did not synchronize on allocation/deallocation,
    unlike `cudaMalloc`/`cudaFree`. So, now `thrust::for_each`,
    `thrust::transform`, `thrust::sort`, etc are truly synchronous. In some
    cases this may be a performance regression; if you need asynchrony, use the
    new asynchronous algorithms.
- Thrust's allocator framework has been rewritten. It now uses a memory
    resource system, similar to C++17's `std::pmr` but supporting static
    polymorphism. Memory resources are objects that allocate untyped storage and
    allocators are cheap handles to memory resources in this new model. The new
    facilities live in `<thrust/mr/*>`.
  - `thrust::mr::memory_resource<Pointer>`, the memory resource base class,
      which takes a (possibly tagged) pointer to `void` type as a parameter.
  - `thrust::mr::allocator<T, MemoryResource>`, an allocator backed by a memory
      resource object.
  - `thrust::mr::polymorphic_adaptor_resource<Pointer>`, a type-erased memory
      resource adaptor.
  - `thrust::mr::polymorphic_allocator<T>`, a C++17-style polymorphic allocator
      backed by a type-erased memory resource object.
  - New tunable C++17-style caching memory resources,
      `thrust::mr::(disjoint_)?(un)?synchronized_pool_resource`, designed to
      cache both small object allocations and large repetitive temporary
      allocations. The disjoint variants use separate storage for management of
      the pool, which is necessary if the memory being allocated cannot be
      accessed on the host (e.g.  device memory).
  - System-specific allocators were rewritten to use the new memory resource
      framework.
  - New `thrust::device_memory_resource` for allocating device memory.    
  - New `thrust::universal_memory_resource` for allocating memory that can be
      accessed from both the host and device (e.g. `cudaMallocManaged`).
  - New `thrust::universal_host_pinned_memory_resource` for allocating memory
      that can be accessed from the host and the device but always resides in
      host memory (e.g. `cudaMallocHost`).
  - `thrust::get_per_device_resource` and `thrust::per_device_allocator`, which
      lazily create and retrieve a per-device singleton memory resource.
  - Rebinding mechanisms (`rebind_traits` and `rebind_alloc`) for
      `thrust::allocator_traits`.
  - `thrust::device_make_unique`, a factory function for creating a
      `std::unique_ptr` to a newly allocated object in device memory.
  - `<thrust/detail/memory_algorithms>`, a C++11 implementation of the C++17
      uninitialized memory algorithms.
  - `thrust::allocate_unique` and friends, based on the proposed C++23
      [`std::allocate_unique`](https://wg21.link/P0211).
- New type traits and metaprogramming facilities. Type traits are slowly being
    migrated out of `thrust::detail::` and `<thrust/detail/*>`; their new home
    will be `thrust::` and `<thrust/type_traits/*>`.
  - `thrust::is_execution_policy`.
  - `thrust::is_operator_less_or_greater_function_object`, which detects
      `thrust::less`, `thrust::greater`, `std::less`, and `std::greater`.
  - `thrust::is_operator_plus_function_object``, which detects `thrust::plus`
      and `std::plus`.
  - `thrust::remove_cvref(_t)?`, a C++11 implementation of C++20's
      `thrust::remove_cvref(_t)?`.
  - `thrust::void_t`, and various other new type traits.
  - `thrust::integer_sequence` and friends, a C++11 implementation of C++20's
      `std::integer_sequence`
  - `thrust::conjunction`, `thrust::disjunction`, and `thrust::disjunction`, a
      C++11 implementation of C++17's logical metafunctions.
  - Some Thrust type traits (such as `thrust::is_constructible`) have been
      redefined in terms of C++11's type traits when they are available.
- `<thrust/detail/tuple_algorithms.h>`, new `std::tuple` algorithms:
  - `thrust::tuple_transform`.
  - `thrust::tuple_for_each`.
  - `thrust::tuple_subset`.
- Miscellaneous new `std::`-like facilities:
  - `thrust::optional`, a C++11 implementation of C++17's `std::optional`.
  - `thrust::addressof`, an implementation of C++11's `std::addressof`.
  - `thrust::next` and `thrust::prev`, an implementation of C++11's `std::next`
      and `std::prev`.
  - `thrust::square`, a `<functional>` style unary function object that
      multiplies its argument by itself.
  - `<thrust/limits.h>` and `thrust::numeric_limits`, a customized version of
      `<limits>` and `std::numeric_limits`.
- `<thrust/detail/preprocessor.h>`, new general purpose preprocessor facilities:
  - `THRUST_PP_CAT[2-5]`, concatenates two to five tokens.
  - `THRUST_PP_EXPAND(_ARGS)?`, performs double expansion.
  - `THRUST_PP_ARITY` and `THRUST_PP_DISPATCH`, tools for macro overloading.
  - `THRUST_PP_BOOL`, boolean conversion.
  - `THRUST_PP_INC` and `THRUST_PP_DEC`, increment/decrement.
  - `THRUST_PP_HEAD`, a variadic macro that expands to the first argument.
  - `THRUST_PP_TAIL`, a variadic macro that expands to all its arguments after
      the first.
  - `THRUST_PP_IIF`, bitwise conditional.
  - `THRUST_PP_COMMA_IF`, and `THRUST_PP_HAS_COMMA`, facilities for adding and
      detecting comma tokens.
  - `THRUST_PP_IS_VARIADIC_NULLARY`, returns true if called with a nullary
      `__VA_ARGS__`.
  - `THRUST_CURRENT_FUNCTION`, expands to the name of the current function.
- New C++11 compatibility macros:
  - `THRUST_NODISCARD`, expands to `[[nodiscard]]` when available and the best
      equivalent otherwise.
  - `THRUST_CONSTEXPR`, expands to `constexpr` when available and the best
      equivalent otherwise.
  - `THRUST_OVERRIDE`, expands to `override` when available and the best
      equivalent otherwise.
  - `THRUST_DEFAULT`, expands to `= default;` when available and the best
      equivalent otherwise.
  - `THRUST_NOEXCEPT`, expands to `noexcept` when available and the best
      equivalent otherwise.
  - `THRUST_FINAL`, expands to `final` when available and the best equivalent
      otherwise.
  - `THRUST_INLINE_CONSTANT`, expands to `inline constexpr` when available and
      the best equivalent otherwise.
- `<thrust/detail/type_deduction.h>`, new C++11-only type deduction helpers:
  - `THRUST_DECLTYPE_RETURNS*`, expand to function definitions with suitable
      conditional `noexcept` qualifiers and trailing return types.
  - `THRUST_FWD(x)`, expands to `::std::forward<decltype(x)>(x)`.
  - `THRUST_MVCAP`, expands to a lambda move capture.
  - `THRUST_RETOF`, expands to a decltype computing the return type of an
      invocable.
- New CMake build system.
   
## New Examples

- `mr_basic` demonstrates how to use the new memory resource allocator system.

## Other Enhancements

- Tagged pointer enhancements:
  - New `thrust::pointer_traits` specialization for `void const*`.
  - `nullptr` support to Thrust tagged pointers.
  - New `explicit operator bool` for Thrust tagged pointers when using C++11
      for `std::unique_ptr` interoperability.
  - Added `thrust::reinterpret_pointer_cast` and `thrust::static_pointer_cast`
      for casting Thrust tagged pointers.
- Iterator enhancements:
  - `thrust::iterator_system` is now SFINAE friendly.
  - Removed cv qualifiers from iterator types when using
      `thrust::iterator_system`.
- Static assert enhancements:
  - New `THRUST_STATIC_ASSERT_MSG`, takes an optional string constant to be
      used as the error message when possible.
  - Update `THRUST_STATIC_ASSERT(_MSG)` to use C++11's `static_assert` when
      it's available.
  - Introduce a way to test for static assertions.
- Testing enhancements:
  - Additional scalar and sequence types, including non-builtin types and
      vectors with unified memory allocators, have been added to the list of
      types used by generic unit tests.
  - The generation of random input data has been improved to increase the range
      of values used and catch more corner cases.
  - New `unittest::truncate_to_max_representable` utility for avoiding the
      generation of ranges that cannot be represented by the underlying element
      type in generic unit test code. 
  - The test driver now synchronizes with CUDA devices and check for errors
      after each test, when switching devices, and after each raw kernel launch.
  - The `warningtester` uber header is now compiled with NVCC to avoid needing
      to disable CUDA-specific code with the preprocessor.
  - Fixed the unit test framework's `ASSERT_*` to print `char`s as `int`s.
  - New `DECLARE_INTEGRAL_VARIABLE_UNITTEST` test declaration macro.
  - New `DECLARE_VARIABLE_UNITTEST_WITH_TYPES_AND_NAME` test declaration macro.
  - `thrust::system_error` in the CUDA backend now print out its `cudaError_t`
      enumerator in addition to the diagnostic message.
  - Stopped using conditionally signed types like `char`.

## Bug Fixes

- #897, NVBug 2062242: Fix compilation error when using `__device__` lambdas
    with `thrust::reduce` on MSVC.
- #908, NVBug 2089386: Static assert that `thrust::generate`/`thrust::fill`
    isn't operating on const iterators.
- #919 Fix compilation failure with `thrust::zip_iterator` and
    `thrust::complex`.
- #924, NVBug 2096679, NVBug 2315990: Fix dispatch for the CUDA backend's
    `thrust::reduce` to use two functions (one with the pragma for disabling
    exec checks, one with `THRUST_RUNTIME_FUNCTION`) instead of one. This fixes
    a regression with device compilation that started in CUDA 9.2.
- #928, NVBug 2341455: Add missing `__host__ __device__` annotations to a
    `thrust::complex::operator=` to satisfy GoUDA.
- NVBug 2094642: Make `thrust::vector_base::clear` not depend on the element
    type being default constructible.
- NVBug 2289115: Remove flaky `simple_cuda_streams` example.
- NVBug 2328572: Add missing `thrust::device_vector` constructor that takes an
    allocator parameter.
- NVBug 2455740: Update the `range_view` example to not use device-side launch.
- NVBug 2455943: Ensure that sized unit tests that use
    `thrust::counting_iterator` perform proper truncation.
- NVBug 2455952: Refactor questionable `thrust::copy_if` unit tests.

# Thrust 1.9.3 (CUDA 10.0)     

## Summary

Thrust 1.9.3 unifies and integrates CUDA Thrust and GitHub Thrust.

## Bug Fixes

- #725, #850, #855, #859, #860: Unify the `thrust::iter_swap` interface and fix
    `thrust::device_reference` swapping.
- NVBug 2004663: Add a `data` method to `thrust::detail::temporary_array` and
    refactor temporary memory allocation in the CUDA backend to be exception
    and leak safe.
- #886, #894, #914: Various documentation typo fixes.
- #724: Provide `NVVMIR_LIBRARY_DIR` environment variable to NVCC.
- #878: Optimize `thrust::min/max_element` to only use
    `thrust::detail::get_iterator_value` for non-numeric types.
- #899: Make `thrust::cuda::experimental::pinned_allocator`'s comparison
    operators `const`.
- NVBug 2092152: Remove all includes of `<cuda.h>`.
- #911: Fix default comparator element type for `thrust::merge_by_key`. 

## Acknowledgments

- Thanks to Andrew Corrigan for contributing fixes for swapping interfaces.
- Thanks to Francisco Facioni for contributing optimizations for
    `thrust::min/max_element`.

# Thrust 1.9.2 (CUDA 9.2)      

## Summary

Thrust 1.9.2 brings a variety of performance enhancements, bug fixes and test
  improvements.
CUB 1.7.5 was integrated, enhancing the performance of `thrust::sort` on
  small data types and `thrust::reduce`.
Changes were applied to `complex` to optimize memory access.
Thrust now compiles with compiler warnings enabled and treated as errors.
Additionally, the unit test suite and framework was enhanced to increase
  coverage.

## Breaking Changes

- The `fallback_allocator` example was removed, as it was buggy and difficult
    to support.

## New Features

- `<thrust/detail/alignment.h>`, utilities for memory alignment:
  - `thrust::aligned_reinterpret_cast`.
  - `thrust::aligned_storage_size`, which computes the amount of storage needed
      for an object of a particular size and alignment.
  - `thrust::alignment_of`, a C++03 implementation of C++11's
      `std::alignment_of`. 
  - `thrust::aligned_storage`, a C++03 implementation of C++11's
      `std::aligned_storage`. 
  - `thrust::max_align_t`, a C++03 implementation of C++11's
      `std::max_align_t`. 

## Bug Fixes
- NVBug 200385527, NVBug 200385119, NVBug 200385113, NVBug 200349350, NVBug
    2058778: Various compiler warning issues.
- NVBug 200355591: `thrust::reduce` performance issues.
- NVBug 2053727: Fixed an ADL bug that caused user-supplied `allocate` to be
    overlooked but `deallocate` to be called with GCC <= 4.3.
- NVBug 1777043: Fixed `thrust::complex` to work with `thrust::sequence`.

# Thrust 1.9.1 (CUDA 9.1)      

## Summary

Thrust 1.9.1 integrates version 1.7.4 of CUB and introduces a new CUDA backend
for `thrust::reduce` based on CUB.

## Bug Fixes

- NVBug 1965743: Remove unnecessary static qualifiers.
- NVBug 1940974: Fix regression causing a compilation error when using
    `thrust::merge_by_key` with `thrust::constant_iterator`s.
- NVBug 1904217: Allow callables that take non-const refs to be used with
    `thrust::reduce` and `thrust::*_scan`.

# Thrust 1.9.0 (CUDA 9.0)      

## Summary

Thrust 1.9.0 replaces the original CUDA backend (bulk) with a new one
  written using CUB, a high performance CUDA collectives library.
This brings a substantial performance improvement to the CUDA backend across
  the board.

## Breaking Changes

- Any code depending on CUDA backend implementation details will likely be
    broken.

## New Features

- New CUDA backend based on CUB which delivers substantially higher performance.
- `thrust::transform_output_iterator`, a fancy iterator that applies a function
    to the output before storing the result. 

## New Examples

- `transform_output_iterator` demonstrates use of the new fancy iterator
    `thrust::transform_output_iterator`.

## Other Enhancements

- When C++11 is enabled, functors do not have to inherit from
    `thrust::(unary|binary)_function` anymore to be used with
    `thrust::transform_iterator`. 
- Added C++11 only move constructors and move assignment operators for
    `thrust::detail::vector_base`-based classes, e.g. `thrust::host_vector`,
    `thrust::device_vector`, and friends.

## Bug Fixes

- `sin(thrust::complex<double>)` no longer has precision loss to float.

## Acknowledgments

- Thanks to Manuel Schiller for contributing a C++11 based enhancement
    regarding the deduction of functor return types, improving the performance
    of `thrust::unique` and implementing `thrust::transform_output_iterator`.
- Thanks to Thibault Notargiacomo for the implementation of move semantics for 
    the `thrust::vector_base`-based classes.
- Thanks to Duane Merrill for developing CUB and helping to integrate it into
    Thrust's backend.

# Thrust 1.8.3 (CUDA 8.0)      

Thrust 1.8.3 is a small bug fix release.

## New Examples

- `range_view` demonstrates the use of a view (a non-owning wrapper for an
    iterator range with a container-like interface).

## Bug Fixes

- `thrust::(min|max|minmax)_element` can now accept raw device pointers when 
    an explicit device execution policy is used.
- `thrust::clear` operations on vector types no longer requires the element
    type to have a default constructor.

# Thrust 1.8.2 (CUDA 7.5)      

Thrust 1.8.2 is a small bug fix release.

## Bug Fixes

- Avoid warnings and errors concerning user functions called from
    `__host__ __device__` functions.
- #632: Fix an error in `thrust::set_intersection_by_key` with the CUDA backend.
- #651: `thrust::copy` between host and device now accepts execution policies
    with streams attached, i.e. `thrust::::cuda::par.on(stream)`.
- #664: `thrust::for_each` and algorithms based on it no longer ignore streams
    attached to execution policys.

## Known Issues

- #628: `thrust::reduce_by_key` for the CUDA backend fails for Compute
    Capability 5.0 devices.

# Thrust 1.8.1 (CUDA 7.0)      

Thrust 1.8.1 is a small bug fix release.

## Bug Fixes

- #615, #620: Fixed `thrust::for_each` and `thrust::reduce` to no longer fail on
    large inputs.

## Known Issues

- #628: `thrust::reduce_by_key` for the CUDA backend fails for Compute
    Capability 5.0 devices.

# Thrust 1.8.0            

Summary
- Thrust 1.8.0 introduces support for algorithm invocation from CUDA __device__ code, support for CUDA streams,
- and algorithm performance improvements. Users may now invoke Thrust algorithms from CUDA __device__ code,
- providing a parallel algorithms library to CUDA programmers authoring custom kernels, as well as allowing
- Thrust programmers to nest their algorithm calls within functors. The thrust::seq execution policy
- allows users to require sequential algorithm execution in the calling thread and makes a
- sequential algorithms library available to individual CUDA threads. The .on(stream) syntax allows users to
- request a CUDA stream for kernels launched during algorithm execution. Finally, new CUDA algorithm
- implementations provide substantial performance improvements.

## New Features
- Algorithms in CUDA __device__ code
      Thrust algorithms may now be invoked from CUDA __device__ and __host__ __device__ functions.

      Algorithms invoked in this manner must be invoked with an execution policy as the first parameter:

      __device__ int my_device_sort(int *data, size_t n)
      {
        thrust::sort(thrust::device, data, data + n);
      }

      The following execution policies are supported in CUDA __device__ code:
        thrust::seq
        thrust::cuda::par
        thrust::device, when THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA

      Parallel algorithm execution may not be accelerated unless CUDA Dynamic Parallelism is available.

- Execution Policies
      CUDA Streams
        The thrust::cuda::par.on(stream) syntax allows users to request that CUDA __global__ functions launched during algorithm 
        execution should occur on a given stream:

        // execute for_each on stream s
        thrust::for_each(thrust::cuda::par.on(s), begin, end, my_functor);

        Algorithms executed with a CUDA stream in this manner may still synchronize with other streams when allocating temporary
        storage or returning results to the CPU.

      thrust::seq
        The thrust::seq execution policy allows users to require that an algorithm execute sequentially in the calling thread:

        // execute for_each sequentially in this thread
        thrust::for_each(thrust::seq, begin, end, my_functor);
        
- Other
      The new thrust::complex template provides complex number support.

## New Examples
- simple_cuda_streams demonstrates how to request a CUDA stream during algorithm execution.
- async_reduce demonstrates ways to achieve algorithm invocations which are asynchronous with the calling thread.

## Other Enhancements
- CUDA sort performance for user-defined types is 300% faster on Tesla K20c for large problem sizes.
- CUDA merge performance is 200% faster on Tesla K20c for large problem sizes.
- CUDA sort performance for primitive types is 50% faster on Tesla K20c for large problem sizes.
- CUDA reduce_by_key performance is 25% faster on Tesla K20c for large problem sizes.
- CUDA scan performance is 15% faster on Tesla K20c for large problem sizes.
- fallback_allocator example is simpler.

## Bug Fixes
- #364 iterators with unrelated system tags may be used with algorithms invoked with an execution policy
- #371 do not redefine __CUDA_ARCH__
- #379 fix crash when dereferencing transform_iterator on the CPU
- #391 avoid use of uppercase variable names
- #392 fix thrust::copy between cusp::complex & std::complex
- #396 program compiled with gcc < 4.3 hangs during comparison sort
- #406 fallback_allocator.cu example checks device for unified addressing support
- #417 avoid using std::less<T> in binary search algorithms
- #418 avoid various warnings
- #443 including version.h no longer configures default systems
- #578 nvcc produces warnings when sequential algorithms are used with cpu systems

## Known Issues
- When invoked with primitive data types, thrust::sort, thrust::sort_by_key, thrust::stable_sort, & thrust::stable_sort_by_key may
- fail to link in some cases with nvcc -rdc=true.

- The CUDA implementation of thrust::reduce_by_key incorrectly outputs the last element in a segment of equivalent keys instead of the first.

Acknowledgments
- Thanks to Sean Baxter for contributing faster CUDA reduce, merge, and scan implementations.
- Thanks to Duane Merrill for contributing a faster CUDA radix sort implementation.
- Thanks to Filipe Maia for contributing the implementation of thrust::complex.

# Thrust 1.7.2 (CUDA 6.5)      

Summary
- Small bug fixes

## Bug Fixes
- Avoid use of std::min in generic find implementation

# Thrust 1.7.1 (CUDA 6.0)      

Summary
- Small bug fixes

## Bug Fixes
- Eliminate identifiers in set_operations.cu example with leading underscore
- Eliminate unused variable warning in CUDA reduce_by_key implementation
- Avoid deriving function objects from std::unary_function and std::binary_function

# Thrust 1.7.0 (CUDA 5.5)      

Summary
- Thrust 1.7.0 introduces a new interface for controlling algorithm execution as
- well as several new algorithms and performance improvements. With this new
- interface, users may directly control how algorithms execute as well as details
- such as the allocation of temporary storage. Key/value versions of thrust::merge
- and the set operation algorithms have been added, as well stencil versions of
- partitioning algorithms. thrust::tabulate has been introduced to tabulate the
- values of functions taking integers. For 32b types, new CUDA merge and set
- operations provide 2-15x faster performance while a new CUDA comparison sort
- provides 1.3-4x faster performance. Finally, a new TBB reduce_by_key implementation
- provides 80% faster performance.

## Breaking Changes
- Dispatch
      Custom user backend systems' tag types must now inherit from the corresponding system's execution_policy template (e.g. thrust::cuda::execution_policy) instead
      of the tag struct (e.g. thrust::cuda::tag). Otherwise, algorithm specializations will silently go unfound during dispatch.
      See examples/minimal_custom_backend.cu and examples/cuda/fallback_allocator.cu for usage examples.

      thrust::advance and thrust::distance are no longer dispatched based on iterator system type and thus may no longer be customized.

- Iterators
      iterator_facade and iterator_adaptor's Pointer template parameters have been eliminated.
      iterator_adaptor has been moved into the thrust namespace (previously thrust::experimental::iterator_adaptor).
      iterator_facade has been moved into the thrust namespace (previously thrust::experimental::iterator_facade).
      iterator_core_access has been moved into the thrust namespace (previously thrust::experimental::iterator_core_access).
      All iterators' nested pointer typedef (the type of the result of operator->) is now void instead of a pointer type to indicate that such expressions are currently impossible.
      Floating point counting_iterators' nested difference_type typedef is now a signed integral type instead of a floating point type.

- Other
      normal_distribution has been moved into the thrust::random namespace (previously thrust::random::experimental::normal_distribution).
      Placeholder expressions may no longer include the comma operator.

## New Features
- Execution Policies
      Users may directly control the dispatch of algorithm invocations with optional execution policy arguments.
      For example, instead of wrapping raw pointers allocated by cudaMalloc with thrust::device_ptr, the thrust::device execution_policy may be passed as an argument to an algorithm invocation to enable CUDA execution.
      The following execution policies are supported in this version:

        thrust::host
        thrust::device
        thrust::cpp::par
        thrust::cuda::par
        thrust::omp::par
        thrust::tbb::par

- Algorithms
	free
	get_temporary_buffer
	malloc
        merge_by_key
        partition with stencil
        partition_copy with stencil
	return_temporary_buffer
        set_difference_by_key
        set_intersection_by_key
        set_symmetric_difference_by_key
        set_union_by_key
        stable_partition with stencil
        stable_partition_copy with stencil
	tabulate

## New Examples
- uninitialized_vector demonstrates how to use a custom allocator to avoid the automatic initialization of elements in thrust::device_vector.

## Other Enhancements
- Authors of custom backend systems may manipulate arbitrary state during algorithm dispatch by incorporating it into their execution_policy parameter.
- Users may control the allocation of temporary storage during algorithm execution by passing standard allocators as parameters via execution policies such as thrust::device.
- THRUST_DEVICE_SYSTEM_CPP has been added as a compile-time target for the device backend. 
- CUDA merge performance is 2-15x faster.
- CUDA comparison sort performance is 1.3-4x faster.
- CUDA set operation performance is 1.5-15x faster.
- TBB reduce_by_key performance is 80% faster.
- Several algorithms have been parallelized with TBB.
- Support for user allocators in vectors has been improved.
- The sparse_vector example is now implemented with merge_by_key instead of sort_by_key.
- Warnings have been eliminated in various contexts.
- Warnings about __host__ or __device__-only functions called from __host__ __device__ functions have been eliminated in various contexts.
- Documentation about algorithm requirements have been improved.
- Simplified the minimal_custom_backend example.
- Simplified the cuda/custom_temporary_allocation example.
- Simplified the cuda/fallback_allocator example.

## Bug Fixes
- #248 fix broken counting_iterator<float> behavior with OpenMP
- #231, #209 fix set operation failures with CUDA
- #187 fix incorrect occupancy calculation with CUDA
- #153 fix broken multigpu behavior with CUDA
- #142 eliminate warning produced by thrust::random::taus88 and MSVC 2010
- #208 correctly initialize elements in temporary storage when necessary
- #16 fix compilation error when sorting bool with CUDA
- #10 fix ambiguous overloads of reinterpret_tag

## Known Issues
- g++ versions 4.3 and lower may fail to dispatch thrust::get_temporary_buffer correctly causing infinite recursion in examples such as cuda/custom_temporary_allocation.

Acknowledgments
- Thanks to Sean Baxter, Bryan Catanzaro, and Manjunath Kudlur for contributing a faster merge implementation for CUDA.
- Thanks to Sean Baxter for contributing a faster set operation implementation for CUDA.
- Thanks to Cliff Woolley for contributing a correct occupancy calculation algorithm.

# Thrust 1.6.0            

Summary
- Thrust v1.6.0 provides an interface for customization and extension and a new
- backend system based on the Threading Building Blocks library. With this
- new interface, programmers may customize the behavior of specific algorithms
- as well as control the allocation of temporary storage or invent entirely new
- backends. These enhancements also allow multiple different backend systems
- such as CUDA and OpenMP to coexist within a single program. Support for TBB
- allows Thrust programs to integrate more naturally into applications which
- may already employ the TBB task scheduler.

## Breaking Changes
- The header <thrust/experimental/cuda/pinned_allocator.h> has been moved to <thrust/system/cuda/experimental/pinned_allocator.h>
- thrust::experimental::cuda::pinned_allocator has been moved to thrust::cuda::experimental::pinned_allocator
- The macro THRUST_DEVICE_BACKEND has been renamed THRUST_DEVICE_SYSTEM
- The macro THRUST_DEVICE_BACKEND_CUDA has been renamed THRUST_DEVICE_SYSTEM_CUDA
- The macro THRUST_DEVICE_BACKEND_OMP has been renamed THRUST_DEVICE_SYSTEM_OMP
- thrust::host_space_tag has been renamed thrust::host_system_tag
- thrust::device_space_tag has been renamed thrust::device_system_tag
- thrust::any_space_tag has been renamed thrust::any_system_tag
- thrust::iterator_space has been renamed thrust::iterator_system
    

## New Features
- Backend Systems
        Threading Building Blocks (TBB) is now supported
- Functions
        for_each_n
        raw_reference_cast
- Types
        pointer
        reference

## New Examples
- cuda/custom_temporary_allocation
- cuda/fallback_allocator
- device_ptr
- expand
- minimal_custom_backend
- raw_reference_cast
- set_operations

## Other Enhancements
- thrust::for_each now returns the end of the input range similar to most other algorithms
- thrust::pair and thrust::tuple have swap functionality
- all CUDA algorithms now support large data types
- iterators may be dereferenced in user __device__ or __global__ functions
- the safe use of different backend systems is now possible within a single binary

## Bug Fixes
- #469 min_element and max_element algorithms no longer require a const comparison operator

## Known Issues
- cudafe++.exe may crash when parsing TBB headers on Windows. 

# Thrust 1.5.3 (CUDA 5.0)      

Summary
- Small bug fixes

## Bug Fixes
- Avoid warnings about potential race due to __shared__ non-POD variable

# Thrust 1.5.2 (CUDA 4.2)      

Summary
- Small bug fixes

## Bug Fixes
- Fixed warning about C-style initialization of structures

# Thrust 1.5.1 (CUDA 4.1)      

Summary
- Small bug fixes

## Bug Fixes
- Sorting data referenced by permutation_iterators on CUDA produces invalid results

# Thrust 1.5.0            

Summary
- Thrust v1.5.0 provides introduces new programmer productivity and performance
- enhancements. New functionality for creating anonymous "lambda" functions has
- been added. A faster host sort provides 2-10x faster performance for sorting
- arithmetic types on (single-threaded) CPUs. A new OpenMP sort provides
- 2.5x-3.0x speedup over the host sort using a quad-core CPU. When sorting
- arithmetic types with the OpenMP backend the combined performance improvement
- is 5.9x for 32-bit integers and ranges from 3.0x (64-bit types) to 14.2x
- (8-bit types). A new CUDA reduce_by_key implementation provides 2-3x faster
- performance.

## Breaking Changes
- device_ptr<void> no longer unsafely converts to device_ptr<T> without an
- explicit cast. Use the expression
- device_pointer_cast(static_cast<int*>(void_ptr.get()))
- to convert, for example, device_ptr<void> to device_ptr<int>.

## New Features
- Functions
        stencil-less transform_if

- Types
        lambda placeholders

## New Examples
- lambda

## Other Enhancements
- host sort is 2-10x faster for arithmetic types
- OMP sort provides speedup over host sort
- reduce_by_key is 2-3x faster
- reduce_by_key no longer requires O(N) temporary storage
- CUDA scan algorithms are 10-40% faster
- host_vector and device_vector are now documented
- out-of-memory exceptions now provide detailed information from CUDART
- improved histogram example
- device_reference now has a specialized swap
- reduce_by_key and scan algorithms are compatible with discard_iterator

Removed Functionality

## Bug Fixes
     #44 allow host_vector to compile when value_type uses __align__
- #198 allow adjacent_difference to permit safe in-situ operation
- #303 make thrust thread-safe
- #313 avoid race conditions in device_vector::insert
- #314 avoid unintended adl invocation when dispatching copy
- #365 fix merge and set operation failures

## Known Issues
- None

Acknowledgments
- Thanks to Manjunath Kudlur for contributing his Carbon library, from which the lambda functionality is derived.
- Thanks to Jean-Francois Bastien for suggesting a fix for issue 303.

# Thrust 1.4.0 (CUDA 4.0)      

Summary
- Thrust v1.4.0 provides support for CUDA 4.0 in addition to many feature
- and performance improvements.  New set theoretic algorithms operating on
- sorted sequences have been added.  Additionally, a new fancy iterator
- allows discarding redundant or otherwise unnecessary output from
- algorithms, conserving memory storage and bandwidth.

## Breaking Changes
- Eliminations
        thrust/is_sorted.h
        thrust/utility.h
        thrust/set_intersection.h
        thrust/experimental/cuda/ogl_interop_allocator.h and the functionality therein
        thrust::deprecated::copy_when
        thrust::deprecated::absolute_value

## New Features
- Functions
        copy_n
        merge
        set_difference
        set_symmetric_difference
        set_union

- Types
        discard_iterator

- Device support
        Compute Capability 2.1 GPUs

## New Examples
- run_length_decoding

## Other Enhancements
- Compilation warnings are substantially reduced in various contexts.
- The compilation time of thrust::sort, thrust::stable_sort, thrust::sort_by_key,
- and thrust::stable_sort_by_key are substantially reduced.
- A fast sort implementation is used when sorting primitive types with thrust::greater.
- The performance of thrust::set_intersection is improved.
- The performance of thrust::fill is improved on SM 1.x devices.
- A code example is now provided in each algorithm's documentation.
- thrust::reverse now operates in-place

Removed Functionality
- thrust::deprecated::copy_when
- thrust::deprecated::absolute_value
- thrust::experimental::cuda::ogl_interop_allocator
- thrust::gather and thrust::scatter from host to device and vice versa are no longer supported.
- Operations which modify the elements of a thrust::device_vector are no longer
- available from source code compiled without nvcc when the device backend is CUDA.
- Instead, use the idiom from the cpp_interop example.

## Bug Fixes
- #212 set_intersection works correctly for large input sizes.
- #275 counting_iterator and constant_iterator work correctly with OpenMP as the
- backend when compiling with optimization
- #256 min and max correctly return their first argument as a tie-breaker
- #248 NDEBUG is interpreted correctly

## Known Issues
- nvcc may generate code containing warnings when compiling some Thrust algorithms.
- When compiling with -arch=sm_1x, some Thrust algorithms may cause nvcc to issue
- benign pointer advisories.
- When compiling with -arch=sm_1x and -G, some Thrust algorithms may fail to execute correctly.
- thrust::inclusive_scan, thrust::exclusive_scan, thrust::inclusive_scan_by_key,
- and thrust::exclusive_scan_by_key are currently incompatible with thrust::discard_iterator.

Acknowledgments
- Thanks to David Tarjan for improving the performance of set_intersection.
- Thanks to Duane Merrill for continued help with sort.
- Thanks to Nathan Whitehead for help with CUDA Toolkit integration.

# Thrust 1.3.0 (CUDA 3.2)      

Summary
- Thrust v1.3.0 provides support for CUDA 3.2 in addition to many feature
- and performance enhancements.
    
- Performance of the sort and sort_by_key algorithms is improved by as much 
- as 3x in certain situations.  The performance of stream compaction algorithms,
- such as copy_if, is improved by as much as 2x.  Reduction performance is 
- also improved, particularly for small input sizes.
    
- CUDA errors are now converted to runtime exceptions using the system_error
- interface.  Combined with a debug mode, also new in v1.3, runtime errors
- can be located with greater precision.

- Lastly, a few header files have been consolidated or renamed for clarity.
- See the deprecations section below for additional details.


## Breaking Changes
- Promotions
        thrust::experimental::inclusive_segmented_scan has been renamed thrust::inclusive_scan_by_key and exposes a different interface
        thrust::experimental::exclusive_segmented_scan has been renamed thrust::exclusive_scan_by_key and exposes a different interface
        thrust::experimental::partition_copy has been renamed thrust::partition_copy and exposes a different interface
        thrust::next::gather has been renamed thrust::gather
        thrust::next::gather_if has been renamed thrust::gather_if
        thrust::unique_copy_by_key has been renamed thrust::unique_by_key_copy
- Deprecations
        thrust::copy_when has been renamed thrust::deprecated::copy_when
        thrust::absolute_value has been renamed thrust::deprecated::absolute_value
        The header thrust/set_intersection.h is now deprecated; use thrust/set_operations.h instead
        The header thrust/utility.h is now deprecated; use thrust/swap.h instead
        The header thrust/swap_ranges.h is now deprecated; use thrust/swap.h instead
- Eliminations
        thrust::deprecated::gather
        thrust::deprecated::gather_if
        thrust/experimental/arch.h and the functions therein
        thrust/sorting/merge_sort.h
        thrust/sorting/radix_sort.h

## New Features
- Functions
        exclusive_scan_by_key
        find
        find_if
        find_if_not
        inclusive_scan_by_key
        is_partitioned
        is_sorted_until
        mismatch
        partition_point
        reverse
        reverse_copy
        stable_partition_copy

- Types
        system_error and related types
        experimental::cuda::ogl_interop_allocator
        bit_and, bit_or, and bit_xor

- Device support
        gf104-based GPUs

## New Examples
- opengl_interop.cu
- repeated_range.cu
- simple_moving_average.cu
- sparse_vector.cu
- strided_range.cu

## Other Enhancements
- Performance of thrust::sort and thrust::sort_by_key is substantially improved for primitive key types
- Performance of thrust::copy_if is substantially improved
- Performance of thrust::reduce and related reductions is improved
- THRUST_DEBUG mode added
- Callers of Thrust functions may detect error conditions by catching thrust::system_error, which derives from std::runtime_error
- The number of compiler warnings generated by Thrust has been substantially reduced
- Comparison sort now works correctly for input sizes > 32M
- min & max usage no longer collides with <windows.h> definitions
- Compiling against the OpenMP backend no longer requires nvcc
- Performance of device_vector initialized in .cpp files is substantially improved in common cases
- Performance of thrust::sort_by_key on the host is substantially improved

Removed Functionality
- nvcc 2.3 is no longer supported

## Bug Fixes
- Debug device code now compiles correctly
- thrust::uninitialized_copy and thrust::unintialized_fill now dispatch constructors on the device rather than the host

## Known Issues
- #212 set_intersection is known to fail for large input sizes
- partition_point is known to fail for 64b types with nvcc 3.2

Acknowledgments
- Thanks to Duane Merrill for contributing a fast CUDA radix sort implementation
- Thanks to Erich Elsen for contributing an implementation of find_if
- Thanks to Andrew Corrigan for contributing changes which allow the OpenMP backend to compile in the absence of nvcc
- Thanks to Andrew Corrigan, Cliff Wooley, David Coeurjolly, Janick Martinez Esturo, John Bowers, Maxim Naumov, Michael Garland, and Ryuta Suzuki for bug reports
- Thanks to Cliff Woolley for help with testing

# Thrust 1.2.1 (CUDA 3.1)      

Summary
- Small fixes for compatibility with CUDA 3.1

## Known Issues
- inclusive_scan & exclusive_scan may fail with very large types
- the Microsoft compiler may fail to compile code using both sort and binary search algorithms
- uninitialized_fill & uninitialized_copy dispatch constructors on the host rather than the device
- # 109 some algorithms may exhibit poor performance with the OpenMP backend with large numbers (>= 6) of CPU threads
- default_random_engine::discard is not accelerated with nvcc 2.3
- nvcc 3.1 may fail to compile code using types derived from thrust::subtract_with_carry_engine, such as thrust::ranlux24 & thrust::ranlux48.

# Thrust 1.2.0            

Summary
- Thrust v1.2 introduces support for compilation to multicore CPUs
- and the Ocelot virtual machine, and several new facilities for
- pseudo-random number generation.  New algorithms such as set
- intersection and segmented reduction have also been added.  Lastly,
- improvements to the robustness of the CUDA backend ensure
- correctness across a broad set of (uncommon) use cases.

## Breaking Changes
- thrust::gather's interface was incorrect and has been removed.
- The old interface is deprecated but will be preserved for Thrust
- version 1.2 at thrust::deprecated::gather &
- thrust::deprecated::gather_if. The new interface is provided at
- thrust::next::gather & thrust::next::gather_if.  The new interface
- will be promoted to thrust:: in Thrust version 1.3. For more details,
- please refer to this thread:
- http://groups.google.com/group/thrust-users/browse_thread/thread/f5f0583cb97b51fd

- The thrust::sorting namespace has been deprecated in favor of the
- top-level sorting functions, such as thrust::sort() and
- thrust::sort_by_key().

## New Features
- Functions
        reduce_by_key
        set_intersection
        tie
        unique_copy
        unique_by_key
        unique_copy_by_key

- Types
        Random Number Generation
            discard_block_engine
            default_random_engine
            linear_congruential_engine
            linear_feedback_shift_engine
            minstd_rand
            minstd_rand0
            normal_distribution (experimental)
            ranlux24
            ranlux48
            ranlux24_base
            ranlux48_base
            subtract_with_carry_engine
            taus88
            uniform_int_distribution
            uniform_real_distribution
            xor_combine_engine
        Functionals
            project1st
            project2nd

- Fancy Iterators
        permutation_iterator
        reverse_iterator

- Device support
        Add support for multicore CPUs via OpenMP
        Add support for Fermi-class GPUs
        Add support for Ocelot virtual machine

## New Examples
- cpp_integration
- histogram
- mode
- monte_carlo
- monte_carlo_disjoint_sequences
- padded_grid_reduction
- permutation_iterator
- row_sum
- run_length_encoding
- segmented_scan
- stream_compaction
- summary_statistics
- transform_iterator
- word_count

## Other Enhancements
- vector functions operator!=, rbegin, crbegin, rend, crend, data, & shrink_to_fit
- integer sorting performance is improved when max is large but (max - min) is small and when min is negative
- performance of inclusive_scan() and exclusive_scan() is improved by 20-25% for primitive types
- support for nvcc 3.0

Removed Functionality
- removed support for equal between host & device sequences
- removed support for gather() and scatter() between host & device sequences

## Bug Fixes
- # 8 cause a compiler error if the required compiler is not found rather than a mysterious error at link time
- # 42 device_ptr & device_reference are classes rather than structs, eliminating warnings on certain platforms
- # 46 gather & scatter handle any space iterators correctly
- # 51 thrust::experimental::arch functions gracefully handle unrecognized GPUs
- # 52 avoid collisions with common user macros such as BLOCK_SIZE
- # 62 provide better documentation for device_reference
- # 68 allow built-in CUDA vector types to work with device_vector in pure C++ mode
- # 102 eliminated a race condition in device_vector::erase
- various compilation warnings eliminated

## Known Issues
   inclusive_scan & exclusive_scan may fail with very large types
   the Microsoft compiler may fail to compile code using both sort and binary search algorithms
   uninitialized_fill & uninitialized_copy dispatch constructors on the host rather than the device
   # 109 some algorithms may exhibit poor performance with the OpenMP backend with large numbers (>= 6) of CPU threads
   default_random_engine::discard is not accelerated with nvcc 2.3

Acknowledgments
   Thanks to Gregory Diamos for contributing a CUDA implementation of set_intersection
   Thanks to Ryuta Suzuki & Gregory Diamos for rigorously testing Thrust's unit tests and examples against Ocelot
   Thanks to Tom Bradley for contributing an implementation of normal_distribution
   Thanks to Joseph Rhoads for contributing the example summary_statistics

# Thrust 1.1.1            

Summary
- Small fixes for compatibility with CUDA 2.3a and Mac OSX Snow Leopard.

# Thrust 1.1.0            

Summary
- Thrust v1.1 introduces fancy iterators, binary search functions, and
- several specialized reduction functions.  Experimental support for
- segmented scan has also been added.

## Breaking Changes
- counting_iterator has been moved into the thrust namespace (previously thrust::experimental)

## New Features
- Functions
        copy_if
        lower_bound
        upper_bound
        vectorized lower_bound
        vectorized upper_bound
        equal_range
        binary_search
        vectorized binary_search
        all_of
        any_of
        none_of
        minmax_element
        advance
        inclusive_segmented_scan (experimental)
        exclusive_segmented_scan (experimental)

- Types
        pair
        tuple
        device_malloc_allocator

- Fancy Iterators
        constant_iterator
        counting_iterator
        transform_iterator
        zip_iterator

## New Examples
- computing the maximum absolute difference between vectors
- computing the bounding box of a two-dimensional point set
- sorting multiple arrays together (lexicographical sorting)
- constructing a summed area table
- using zip_iterator to mimic an array of structs
- using constant_iterator to increment array values

## Other Enhancements
- added pinned memory allocator (experimental)
- added more methods to host_vector & device_vector (issue #4)
- added variant of remove_if with a stencil argument (issue #29)
- scan and reduce use cudaFuncGetAttributes to determine grid size
- exceptions are reported when temporary device arrays cannot be allocated 

## Bug Fixes
     #5 make vector work for larger data types
     #9 stable_partition_copy doesn't respect OutputIterator concept semantics
- #10 scans should return OutputIterator
- #16 make algorithms work for larger data types
- #27 dispatch radix_sort even when comp=less<T> is explicitly provided

## Known Issues
- Using functors with Thrust entry points may not compile on Mac OSX with gcc
    4.0.1.
- `thrust::uninitialized_copy` and `thrust::uninitialized_fill` dispatch
    constructors on the host rather than the device.
- `thrust::inclusive_scan`, `thrust::inclusive_scan_by_key`,
    `thrust::exclusive_scan`, and `thrust::exclusive_scan_by_key` may fail when
    used with large types with the CUDA 3.1 driver.

# Thrust 1.0.0            

## Breaking Changes
- Rename top level namespace `komrade` to `thrust`.
- Move `thrust::partition_copy` & `thrust::stable_partition_copy` into
    `thrust::experimental` namespace until we can easily provide the standard
    interface.
- Rename `thrust::range` to `thrust::sequence` to avoid collision with
    Boost.Range.
- Rename `thrust::copy_if` to `thrust::copy_when` due to semantic differences
    with C++0x copy_if().

## New Features
- Add C++0x style `cbegin` & `cend` methods to `thrust::host_vector` and
    `thrust::device_vector`.
- Add `thrust::transform_if` function.
- Add stencil versions of `thrust::replace_if` & `thrust::replace_copy_if`.
- Allow `counting_iterator` to work with `thrust::for_each`.
- Allow types with constructors in comparison `thrust::sort` and
    `thrust::reduce`.

## Other Enhancements
- `thrust::merge_sort` and `thrust::stable_merge_sort` are now 2x to 5x faster
    when executed on the parallel device.

## Bug Fixes
- Komrade 6: Workaround an issue where an incremented iterator causes NVCC to
    crash.
- Komrade 7: Fix an issue where `const_iterator`s could not be passed to
    `thrust::transform`.

