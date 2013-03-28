#include <unittest/unittest.h>
#include <thrust/device_malloc_allocator.h>
#include <thrust/system/cpp/vector.h>
#include <memory>

struct my_allocator_with_custom_construct1
  : thrust::device_malloc_allocator<int>
{
  __host__ __device__
  my_allocator_with_custom_construct1()
  {}

  template<typename T>
  __host__ __device__
  void construct(T *p)
  {
    *p = 13;
  }
};

void TestAllocatorCustomDefaultConstruct()
{
  thrust::device_vector<int> ref(10,13);
  thrust::device_vector<int, my_allocator_with_custom_construct1> vec(10);

  ASSERT_EQUAL_QUIET(ref, vec);
}
DECLARE_UNITTEST(TestAllocatorCustomDefaultConstruct);


struct my_allocator_with_custom_construct2
  : thrust::device_malloc_allocator<int>
{
  __host__ __device__
  my_allocator_with_custom_construct2()
  {}

  template<typename T, typename Arg>
  __host__ __device__
  void construct(T *p, const Arg &)
  {
    *p = 13;
  }
};

void TestAllocatorCustomCopyConstruct()
{
  thrust::device_vector<int> ref(10,13);
  thrust::device_vector<int> copy_from(10,7);
  thrust::device_vector<int, my_allocator_with_custom_construct2> vec(copy_from.begin(), copy_from.end());

  ASSERT_EQUAL_QUIET(ref, vec);
}
DECLARE_UNITTEST(TestAllocatorCustomCopyConstruct);

static int g_state;

struct my_allocator_with_custom_destroy
{
  typedef int         value_type;
  typedef int &       reference;
  typedef const int & const_reference;

  __host__
  my_allocator_with_custom_destroy(){}

  __host__
  my_allocator_with_custom_destroy(const my_allocator_with_custom_destroy &other)
    : use_me_to_alloc(other.use_me_to_alloc)
  {}

  __host__
  ~my_allocator_with_custom_destroy(){}

  template<typename T>
  __host__ __device__
  void destroy(T *p)
  {
#if !__CUDA_ARCH__
    g_state = 13;
#endif
  }

  value_type *allocate(std::ptrdiff_t n)
  {
    return use_me_to_alloc.allocate(n);
  }

  void deallocate(value_type *ptr, std::ptrdiff_t n)
  {
    use_me_to_alloc.deallocate(ptr,n);
  }
  
  // use composition rather than inheritance
  // to avoid inheriting std::allocator's member
  // function construct
  std::allocator<int> use_me_to_alloc;
};

void TestAllocatorCustomDestroy()
{
  thrust::cpp::vector<int, my_allocator_with_custom_destroy> vec(10);

  // destroy everything
  vec.shrink_to_fit();

  ASSERT_EQUAL(13, g_state);
}
DECLARE_UNITTEST(TestAllocatorCustomDestroy);

struct my_minimal_allocator
{
  typedef int         value_type;

  // XXX ideally, we shouldn't require
  //     these two typedefs
  typedef int &       reference;
  typedef const int & const_reference;

  __host__
  my_minimal_allocator(){}

  __host__
  my_minimal_allocator(const my_minimal_allocator &other)
    : use_me_to_alloc(other.use_me_to_alloc)
  {}

  __host__
  ~my_minimal_allocator(){}

  value_type *allocate(std::ptrdiff_t n)
  {
    return use_me_to_alloc.allocate(n);
  }

  void deallocate(value_type *ptr, std::ptrdiff_t n)
  {
    use_me_to_alloc.deallocate(ptr,n);
  }

  std::allocator<int> use_me_to_alloc;
};

void TestAllocatorMinimal()
{
  thrust::cpp::vector<int, my_minimal_allocator> vec(10, 13);

  // XXX copy to h_vec because ASSERT_EQUAL doesn't know about cpp::vector
  thrust::host_vector<int> h_vec(vec.begin(), vec.end());
  thrust::host_vector<int> ref(10, 13);

  ASSERT_EQUAL(ref, h_vec);
}
DECLARE_UNITTEST(TestAllocatorMinimal);

