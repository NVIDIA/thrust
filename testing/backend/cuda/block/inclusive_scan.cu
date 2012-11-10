#include <unittest/unittest.h>

#if defined(__CUDACC__)

#include <thrust/detail/minmax.h>
#include <thrust/system/cuda/detail/runtime_introspection.h>
#include <thrust/system/cuda/detail/extern_shared_ptr.h>
#include <thrust/system/cuda/detail/detail/launch_closure.h>
#include <thrust/system/cuda/detail/block/inclusive_scan.h>
#include <thrust/scan.h>

template <typename InputIterator,
          typename OutputIterator,
          typename Context>
struct block_inclusive_scan_closure
{
  InputIterator input;
  OutputIterator output;
  Context  context;

  typedef Context context_type;

  typedef typename thrust::iterator_value<OutputIterator>::type ValueType;

  block_inclusive_scan_closure(InputIterator input, OutputIterator output, Context context = Context())
    : input(input), output(output), context(context) {}

  __device__
  void operator()(void)
  {
    thrust::system::cuda::detail::extern_shared_ptr<ValueType> shared_array;

    input  += context.thread_index();
    output += context.thread_index();

    shared_array[context.thread_index()] = *input;

    context.barrier();

    thrust::system::cuda::detail::block::inclusive_scan_n(context, &shared_array[0], context.block_dimension(), thrust::plus<int>());

    *output = shared_array[context.thread_index()];
  }
};


template <typename Closure>
bool is_valid_launch(size_t grid_size, size_t block_size, size_t smem_bytes = 0)
{
  thrust::system::cuda::detail::device_properties_t   properties = thrust::system::cuda::detail::device_properties();
  thrust::system::cuda::detail::function_attributes_t attributes = thrust::system::cuda::detail::detail::closure_attributes<Closure>();

  size_t total_smem = smem_bytes + attributes.sharedSizeBytes;

  if (grid_size  > (size_t) properties.maxGridSize[0])     return false;
  if (block_size > (size_t) attributes.maxThreadsPerBlock) return false;
  if (total_smem > (size_t) properties.sharedMemPerBlock)  return false;

  if (grid_size == 0 || block_size == 0) return false;

  return true;
}

thrust::host_vector<size_t> get_block_sizes(void)
{
  static size_t block_sizes[] = {18, 32, 53, 64, 75, 96, 110, 128, 187, 256, 300, 512, 768, 911, 1024, 1280, 1536, 1776, 2048};
  static size_t N = sizeof(block_sizes) / sizeof(size_t);

  return thrust::host_vector<size_t>(&*block_sizes, &*block_sizes + N);
}

template <typename Context>
void test_inclusive_scan(size_t block_size)
{
  typedef unsigned int                     ValueType;
  typedef thrust::host_vector<ValueType>   HostVector;
  typedef thrust::device_vector<ValueType> DeviceVector;
  typedef DeviceVector::iterator           DeviceIterator;

  typedef block_inclusive_scan_closure<DeviceIterator,DeviceIterator,Context> Closure;

  size_t smem_bytes = sizeof(ValueType) * block_size;

   if (is_valid_launch<Closure>(1, block_size, smem_bytes))
   {
     HostVector   h_input  = unittest::random_integers<bool>(block_size);
     HostVector   h_output(block_size, 0);
     
     thrust::inclusive_scan(h_input.begin(), h_input.end(), h_output.begin());

     DeviceVector d_input(h_input);
     DeviceVector d_output(block_size, 0);

     thrust::system::cuda::detail::detail::launch_closure
       (Closure(d_input.begin(), d_output.begin()), 1, block_size, smem_bytes);

     ASSERT_EQUAL(h_output, d_output);
   }
}

template <unsigned int block_size>
void test_inclusive_scan_static(void)
{
  typedef thrust::system::cuda::detail::detail::statically_blocked_thread_array<block_size>  Context;
  test_inclusive_scan<Context>(block_size);
}

void TestCudaBlockInclusiveScanStaticallyBlockedThreadArray(void)
{
  // test static block sizes
  test_inclusive_scan_static<18>();
  test_inclusive_scan_static<32>();
  test_inclusive_scan_static<53>();
  test_inclusive_scan_static<64>();
  test_inclusive_scan_static<75>();
  test_inclusive_scan_static<96>();
  test_inclusive_scan_static<110>();
  test_inclusive_scan_static<128>();
  test_inclusive_scan_static<187>();
  test_inclusive_scan_static<256>();
  test_inclusive_scan_static<300>();
  test_inclusive_scan_static<512>();
}
DECLARE_UNITTEST(TestCudaBlockInclusiveScanStaticallyBlockedThreadArray);


void test_inclusive_scan_dynamic(unsigned int block_size)
{
  typedef thrust::system::cuda::detail::detail::blocked_thread_array Context;
  test_inclusive_scan<Context>(block_size);
}

void TestCudaBlockInclusiveScanBlockedThreadArray(void)
{
  // test dynamic block sizes
  typedef thrust::system::cuda::detail::detail::blocked_thread_array Context;

  thrust::host_vector<size_t> block_sizes = get_block_sizes();

  for (size_t i = 0; i < block_sizes.size(); i++)
    test_inclusive_scan_dynamic(block_sizes[i]);
}
DECLARE_UNITTEST(TestCudaBlockInclusiveScanBlockedThreadArray);

#endif // defined(__CUDACC__)

