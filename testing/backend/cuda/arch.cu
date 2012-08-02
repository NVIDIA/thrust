#include <unittest/unittest.h>

#if defined(__CUDACC__)

#include <thrust/system/cuda/detail/runtime_introspection.h>
#include <thrust/system/cuda/detail/cuda_launch_config.h>

using namespace thrust::system::cuda::detail;

void set_compute_capability(device_properties_t& properties, int major, int minor)
{
  properties.major = major;
  properties.minor = minor;
}

void set_G80(device_properties_t& properties)
{
  set_compute_capability(properties, 1, 0);
  properties.multiProcessorCount         = 16;
  properties.sharedMemPerBlock           = 16384;
  properties.regsPerBlock                = 8192;
  properties.warpSize                    = 32;
  properties.maxThreadsPerBlock          = 512;
  properties.maxThreadsPerMultiProcessor = 768;
}

void set_G84(device_properties_t& properties)
{
  set_compute_capability(properties, 1, 1);
  properties.multiProcessorCount         = 4;
  properties.sharedMemPerBlock           = 16384;
  properties.regsPerBlock                = 8192;
  properties.warpSize                    = 32;
  properties.maxThreadsPerBlock          = 512;
  properties.maxThreadsPerMultiProcessor = 768;
}

void set_GT200(device_properties_t& properties)
{
  set_compute_capability(properties, 1, 3);
  properties.multiProcessorCount         = 30;
  properties.sharedMemPerBlock           = 16384;
  properties.regsPerBlock                = 16384;
  properties.warpSize                    = 32;
  properties.maxThreadsPerBlock          = 512;
  properties.maxThreadsPerMultiProcessor = 1024;
}

void set_unknown(device_properties_t& properties)
{
  set_compute_capability(properties, 900, 1);
  properties.multiProcessorCount         = 9001;
  properties.sharedMemPerBlock           = 4 * 16384;
  properties.regsPerBlock                = 32768;
  properties.warpSize                    = 32;
  properties.maxThreadsPerBlock          = 4096;
  properties.maxThreadsPerMultiProcessor = 8192;
}

void set_func_attributes(function_attributes_t& attributes,
                         size_t constSizeBytes,           // Size of constant memory in bytes.
                         size_t localSizeBytes,           // Size of local memory in bytes.
                         int maxThreadsPerBlock,          // Maximum number of threads per block.
                         int numRegs,                     // Number of registers used.
                         size_t sharedSizeBytes)          // Size of shared memory in bytes.
{
    attributes.constSizeBytes     = constSizeBytes;
    attributes.localSizeBytes     = localSizeBytes;
    attributes.maxThreadsPerBlock = maxThreadsPerBlock; 
    attributes.numRegs            = numRegs;
    attributes.sharedSizeBytes    = sharedSizeBytes;
}

void TestComputeCapability(void)
{
    device_properties_t properties;

    set_compute_capability(properties, 1, 0);
    ASSERT_EQUAL(compute_capability(properties), 10);

    set_compute_capability(properties, 1, 1);
    ASSERT_EQUAL(compute_capability(properties), 11);
    
    set_compute_capability(properties, 1, 3);
    ASSERT_EQUAL(compute_capability(properties), 13);
    
    set_compute_capability(properties, 2, 0);
    ASSERT_EQUAL(compute_capability(properties), 20);
    
    set_compute_capability(properties, 2, 1);
    ASSERT_EQUAL(compute_capability(properties), 21);
}
DECLARE_UNITTEST(TestComputeCapability);


void TestMaxActiveBlocks(void)
{
    using namespace cuda_launch_config_detail;

    device_properties_t   properties;
    function_attributes_t attributes;

    // Kernel #1 : Full Occupancy on all devices
    set_func_attributes(attributes, 0, 0, 512, 10, 2048);
    
    set_G80(properties);   ASSERT_EQUAL(max_active_blocks_per_multiprocessor(properties, attributes, 256, 0), 3);
    set_G84(properties);   ASSERT_EQUAL(max_active_blocks_per_multiprocessor(properties, attributes, 256, 0), 3);
    set_GT200(properties); ASSERT_EQUAL(max_active_blocks_per_multiprocessor(properties, attributes, 256, 0), 4);
    
    // Kernel #2 : 2/3rds Occupancy on G8x and 100% on GT200
    set_func_attributes(attributes, 0, 0, 512, 16, 2048);

    set_G80(properties);   ASSERT_EQUAL(max_active_blocks_per_multiprocessor(properties, attributes, 256, 0), 2);
    set_G84(properties);   ASSERT_EQUAL(max_active_blocks_per_multiprocessor(properties, attributes, 256, 0), 2);
    set_GT200(properties); ASSERT_EQUAL(max_active_blocks_per_multiprocessor(properties, attributes, 256, 0), 4);
    
    // Kernel #3 : 1/3rds Occupancy on G8x and 75% on GT200
    set_func_attributes(attributes, 0, 0, 512, 20, 2048);

    set_G80(properties);   ASSERT_EQUAL(max_active_blocks_per_multiprocessor(properties, attributes, 256, 0), 1);
    set_G84(properties);   ASSERT_EQUAL(max_active_blocks_per_multiprocessor(properties, attributes, 256, 0), 1);
    set_GT200(properties); ASSERT_EQUAL(max_active_blocks_per_multiprocessor(properties, attributes, 256, 0), 3);
    
    // Kernel #4 : 1/3rds Occupancy on G8x and 50% on GT200
    set_func_attributes(attributes, 0, 0, 512, 21, 2048);

    set_G80(properties);   ASSERT_EQUAL(max_active_blocks_per_multiprocessor(properties, attributes, 256, 0), 1);
    set_G84(properties);   ASSERT_EQUAL(max_active_blocks_per_multiprocessor(properties, attributes, 256, 0), 1);
    set_GT200(properties); ASSERT_EQUAL(max_active_blocks_per_multiprocessor(properties, attributes, 256, 0), 2);
    
    // Kernel #5 : 2/3rds Occupancy on G8x and 50% on GT200
    set_func_attributes(attributes, 0, 0, 512, 10, 8192);

    set_G80(properties);   ASSERT_EQUAL(max_active_blocks_per_multiprocessor(properties, attributes, 256, 0), 2);
    set_G84(properties);   ASSERT_EQUAL(max_active_blocks_per_multiprocessor(properties, attributes, 256, 0), 2);
    set_GT200(properties); ASSERT_EQUAL(max_active_blocks_per_multiprocessor(properties, attributes, 256, 0), 2);
}
DECLARE_UNITTEST(TestMaxActiveBlocks);


void TestMaxBlocksizeWithHighestOccupancy(void)
{
    device_properties_t   properties;
    function_attributes_t attributes;
    
    // Kernel #1 : Full Occupancy on all devices
    set_func_attributes(attributes, 0, 0, 512, 10, 2048);
    
    set_G80(properties);   ASSERT_EQUAL(block_size_with_maximum_potential_occupancy(attributes, properties), 384);
    set_GT200(properties); ASSERT_EQUAL(block_size_with_maximum_potential_occupancy(attributes, properties), 512);
    
    // Kernel #2 : 2/3rds Occupancy on G8x and 100% on GT200
    set_func_attributes(attributes, 0, 0, 512, 16, 2048);

    set_G80(properties);   ASSERT_EQUAL(block_size_with_maximum_potential_occupancy(attributes, properties), 512);
    set_GT200(properties); ASSERT_EQUAL(block_size_with_maximum_potential_occupancy(attributes, properties), 512);
    
    // Kernel #3 : 50% Occupancy on G8x and 75% on GT200
    set_func_attributes(attributes, 0, 0, 256, 20, 2048);
    
    set_G80(properties);   ASSERT_EQUAL(block_size_with_maximum_potential_occupancy(attributes, properties), 192);
    set_GT200(properties); ASSERT_EQUAL(block_size_with_maximum_potential_occupancy(attributes, properties), 256);
    
    // Kernel #4 : 1/3rds Occupancy on G8x and 50% on GT200
    set_func_attributes(attributes, 0, 0, 384, 26, 2048);
    
    set_G80(properties);   ASSERT_EQUAL(block_size_with_maximum_potential_occupancy(attributes, properties), 256);
    set_GT200(properties); ASSERT_EQUAL(block_size_with_maximum_potential_occupancy(attributes, properties), 192);
    
    // Kernel #5 :100% Occupancy on G8x and GT200
    set_func_attributes(attributes, 0, 0, 512, 10, 8192);
    
    set_G80(properties);   ASSERT_EQUAL(block_size_with_maximum_potential_occupancy(attributes, properties), 384);
    set_GT200(properties); ASSERT_EQUAL(block_size_with_maximum_potential_occupancy(attributes, properties), 512);
}
DECLARE_UNITTEST(TestMaxBlocksizeWithHighestOccupancy);

struct return_int
{
  int val;

  return_int(int val)
    : val(val)
  {}

  __host__ __device__
  int operator()(int) const
  {
    return val;
  }
};

static bool validate_nonzero_results(const device_properties_t   &properties,
                                     const function_attributes_t &attributes)
{
  using thrust::system::cuda::detail::cuda_launch_config_detail::max_active_blocks_per_multiprocessor;

  bool result = true;

  // validate that all these calls return something non-zero
  result &= (max_active_blocks_per_multiprocessor(properties, attributes, 512, 512 * 4) > 0);
  ASSERT_EQUAL(true, result);

  result &= block_size_with_maximum_potential_occupancy(attributes, properties) > 0;
  ASSERT_EQUAL(true, result);

  result &= block_size_with_maximum_potential_occupancy(attributes, properties, return_int(4)) > 0;
  ASSERT_EQUAL(true, result);

  return result;
}

void TestUnknownDeviceRobustness(void)
{
    device_properties_t  properties;
    function_attributes_t attributes;

    // create an unknown device
    set_unknown(properties);

    // Kernel #1 : Full Occupancy on all real devices
    set_func_attributes(attributes, 0, 0, 512, 10, 2048);
    ASSERT_EQUAL(true, validate_nonzero_results(properties, attributes));

    // Kernel #2 : 2/3rds Occupancy on G8x and 100% on GT200
    set_func_attributes(attributes, 0, 0, 512, 16, 2048);
    ASSERT_EQUAL(true, validate_nonzero_results(properties, attributes));

    // Kernel #3 : 50% Occupancy on G8x and 75% on GT200
    set_func_attributes(attributes, 0, 0, 512, 20, 2048);
    ASSERT_EQUAL(true, validate_nonzero_results(properties, attributes));

    // Kernel #4 : 1/3rds Occupancy on G8x and 50% on GT200
    set_func_attributes(attributes, 0, 0, 384, 26, 2048);
    ASSERT_EQUAL(true, validate_nonzero_results(properties, attributes));

    // Kernel #5 :100% Occupancy on G8x and GT200
    set_func_attributes(attributes, 0, 0, 512, 10, 8192);
    ASSERT_EQUAL(true, validate_nonzero_results(properties, attributes));
}
DECLARE_UNITTEST(TestUnknownDeviceRobustness);

#endif // defined(__CUDACC__)

