#include <unittest/unittest.h>

#if defined(__CUDACC__)

#include <thrust/detail/backend/cuda/arch.h>

using namespace thrust::detail::backend::cuda::arch;

void set_compute_capability(cudaDeviceProp& properties, int major, int minor)
{
  properties.major = major;
  properties.minor = minor;
}

void set_G80(cudaDeviceProp& properties)
{
  set_compute_capability(properties, 1, 0);
  properties.multiProcessorCount         = 16;
  properties.sharedMemPerBlock           = 16384;
  properties.regsPerBlock                = 8192;
  properties.warpSize                    = 32;
  properties.maxThreadsPerBlock          = 512;
  properties.maxThreadsPerMultiProcessor = 768;
}

void set_G84(cudaDeviceProp& properties)
{
  set_compute_capability(properties, 1, 1);
  properties.multiProcessorCount         = 4;
  properties.sharedMemPerBlock           = 16384;
  properties.regsPerBlock                = 8192;
  properties.warpSize                    = 32;
  properties.maxThreadsPerBlock          = 512;
  properties.maxThreadsPerMultiProcessor = 768;
}

void set_GT200(cudaDeviceProp& properties)
{
  set_compute_capability(properties, 1, 3);
  properties.multiProcessorCount         = 30;
  properties.sharedMemPerBlock           = 16384;
  properties.regsPerBlock                = 16384;
  properties.warpSize                    = 32;
  properties.maxThreadsPerBlock          = 512;
  properties.maxThreadsPerMultiProcessor = 1024;
}

void set_func_attributes(cudaFuncAttributes& attributes,
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
    cudaDeviceProp properties;

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


void TestMaxActiveThreads(void)
{
    cudaDeviceProp properties;

    set_G80(properties);
    ASSERT_EQUAL(max_active_threads_per_multiprocessor(properties), 768);
    
    set_G84(properties);
    ASSERT_EQUAL(max_active_threads_per_multiprocessor(properties), 768);
    
    set_GT200(properties);
    ASSERT_EQUAL(max_active_threads_per_multiprocessor(properties), 1024);
}
DECLARE_UNITTEST(TestMaxActiveThreads);


void TestMaxActiveBlocks(void)
{
    cudaDeviceProp properties;
    cudaFuncAttributes attributes;

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
    cudaDeviceProp properties;
    cudaFuncAttributes attributes;
    
    // Kernel #1 : Full Occupancy on all devices
    set_func_attributes(attributes, 0, 0, 512, 10, 2048);
    
    set_G80(properties);   ASSERT_EQUAL(max_blocksize_with_highest_occupancy(properties, attributes), 384);
    set_GT200(properties); ASSERT_EQUAL(max_blocksize_with_highest_occupancy(properties, attributes), 512);
    
    // Kernel #2 : 2/3rds Occupancy on G8x and 100% on GT200
    set_func_attributes(attributes, 0, 0, 512, 16, 2048);

    set_G80(properties);   ASSERT_EQUAL(max_blocksize_with_highest_occupancy(properties, attributes), 512);
    set_GT200(properties); ASSERT_EQUAL(max_blocksize_with_highest_occupancy(properties, attributes), 512);
    
    // Kernel #3 : 50% Occupancy on G8x and 75% on GT200
    set_func_attributes(attributes, 0, 0, 256, 20, 2048);
    
    set_G80(properties);   ASSERT_EQUAL(max_blocksize_with_highest_occupancy(properties, attributes), 192);
    set_GT200(properties); ASSERT_EQUAL(max_blocksize_with_highest_occupancy(properties, attributes), 256);
    
    // Kernel #4 : 1/3rds Occupancy on G8x and 50% on GT200
    set_func_attributes(attributes, 0, 0, 384, 26, 2048);
    
    set_G80(properties);   ASSERT_EQUAL(max_blocksize_with_highest_occupancy(properties, attributes), 256);
    set_GT200(properties); ASSERT_EQUAL(max_blocksize_with_highest_occupancy(properties, attributes), 192);
    
    // Kernel #5 :100% Occupancy on G8x and GT200
    set_func_attributes(attributes, 0, 0, 512, 10, 8192);
    
    set_G80(properties);   ASSERT_EQUAL(max_blocksize_with_highest_occupancy(properties, attributes), 384);
    set_GT200(properties); ASSERT_EQUAL(max_blocksize_with_highest_occupancy(properties, attributes), 512);
}
DECLARE_UNITTEST(TestMaxBlocksizeWithHighestOccupancy);


void TestMaxBlocksize(void)
{
    cudaDeviceProp properties;
    cudaFuncAttributes attributes;
    
    // Kernel #1 : Full Occupancy on all devices
    set_func_attributes(attributes, 0, 0, 512, 10, 2048);
    
    set_G80(properties);   ASSERT_EQUAL(max_blocksize(properties, attributes), 512);
    set_GT200(properties); ASSERT_EQUAL(max_blocksize(properties, attributes), 512);
    
    // Kernel #2 : 2/3rds Occupancy on G8x and 100% on GT200
    set_func_attributes(attributes, 0, 0, 512, 16, 2048);

    set_G80(properties);   ASSERT_EQUAL(max_blocksize(properties, attributes), 512);
    set_GT200(properties); ASSERT_EQUAL(max_blocksize(properties, attributes), 512);
    
    // Kernel #3 : 50% Occupancy on G8x and 75% on GT200
    set_func_attributes(attributes, 0, 0, 512, 20, 2048);
    
    set_G80(properties);   ASSERT_EQUAL(max_blocksize(properties, attributes), 384);
    set_GT200(properties); ASSERT_EQUAL(max_blocksize(properties, attributes), 512);
    
    // Kernel #4 : 1/3rds Occupancy on G8x and 50% on GT200
    set_func_attributes(attributes, 0, 0, 384, 26, 2048);
    
    set_G80(properties);   ASSERT_EQUAL(max_blocksize(properties, attributes), 256);
    set_GT200(properties); ASSERT_EQUAL(max_blocksize(properties, attributes), 384);
    
    // Kernel #5 :100% Occupancy on G8x and GT200
    set_func_attributes(attributes, 0, 0, 512, 10, 8192);
    
    set_G80(properties);   ASSERT_EQUAL(max_blocksize(properties, attributes), 512);
    set_GT200(properties); ASSERT_EQUAL(max_blocksize(properties, attributes), 512);
}
DECLARE_UNITTEST(TestMaxBlocksize);

#endif // defined(__CUDACC__)

