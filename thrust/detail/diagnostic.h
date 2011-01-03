/*
 *  Copyright 2008-2011 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */


/*! \file diagnostic.h
 *  \brief Output system diagnostics
 *        
 */

#pragma once

#include <thrust/detail/config.h>
#include <thrust/version.h>

#include <iostream>
#include <stdio.h>

namespace thrust
{
namespace detail
{

inline 
void output_compiler_diagnostics(void)
{
#if defined(__GNUC__)
    // GCC
    std::cerr << "GCC " << __GNUC__ << "." << __GNUC_MINOR__;
#if defined(__GNUC_PATCHLEVEL__)
    std::cerr << "." << __GNUC_PATCHLEVEL__;
#endif
#elif defined(_MSC_VER)
    // Microsoft Visual C++
    std::cerr << "MSVC " << _MSC_VER;
#elif defined(__INTEL_COMPILER)
    // Intel Compiler
    std::cerr << "MSVC " << __INTEL_COMPILER;
#else // Unknown
    std::cerr << "UNKNOWN";
#endif
    std::cerr << std::endl;

#ifdef __CUDACC__
    std::cerr << "NVCC " << (CUDA_VERSION / 1000) << "." 
                         << (CUDA_VERSION % 1000) / 10 << "." 
                         << (CUDA_VERSION % 10) << std::endl;
#endif    
}

inline 
void output_thrust_diagnostics(void)
{
    std::cout << "Thrust " << THRUST_MAJOR_VERSION << "."
                           << THRUST_MINOR_VERSION << "."
                           << THRUST_SUBMINOR_VERSION << std::endl;
}


inline 
void output_device_diagnostics(void)
{
#ifdef __CUDACC__
    int deviceCount = 0;
	if (cudaGetDeviceCount(&deviceCount) != cudaSuccess)
    {
		fprintf(stderr,"cudaGetDeviceCount FAILED CUDA Driver and Runtime version may be mismatched.\n");
		fprintf(stderr,"\nFAILED\n");
	    return;
	}

    // This function call returns 0 if there are no CUDA capable devices.
    if (deviceCount == 0)
        fprintf(stderr,"There is no device supporting CUDA\n");
    
    int nGpuArchCoresPerSM[] = { -1, 8, 32, -1, -1, -1}; // buy ourselves some time

    int dev;
	int driverVersion = 0, runtimeVersion = 0;     
    for (dev = 0; dev < deviceCount; ++dev) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        if (dev == 0)
        {
            fprintf(stderr, "\n\n");
			// This function call returns 9999 for both major & minor fields, if no CUDA capable devices are present
            if (deviceProp.major == 9999 && deviceProp.minor == 9999)
                fprintf(stderr,"There is no device supporting CUDA.\n");
            else if (deviceCount == 1)
                fprintf(stderr,"There is 1 device supporting CUDA\n");
            else
                fprintf(stderr,"There are %d devices supporting CUDA\n", deviceCount);
        }
        fprintf(stderr,"\nDevice %d: \"%s\"\n", dev, deviceProp.name);

    #if CUDART_VERSION >= 2020
        // Console log
		cudaDriverGetVersion(&driverVersion);
		fprintf(stderr,"  CUDA Driver Version:                           %d.%d\n", driverVersion/1000, driverVersion%100);
		cudaRuntimeGetVersion(&runtimeVersion);
		fprintf(stderr,"  CUDA Runtime Version:                          %d.%d\n", runtimeVersion/1000, runtimeVersion%100);
    #endif
        fprintf(stderr,"  CUDA Capability Major revision number:         %d\n", deviceProp.major);
        fprintf(stderr,"  CUDA Capability Minor revision number:         %d\n", deviceProp.minor);

		fprintf(stderr,"  Total amount of global memory:                 %.0lf MBytes\n", (deviceProp.totalGlobalMem / 1048576.0));
    #if CUDART_VERSION >= 2000
        fprintf(stderr,"  Number of multiprocessors:                     %d\n", deviceProp.multiProcessorCount);
        fprintf(stderr,"  Number of cores:                               %d\n", nGpuArchCoresPerSM[deviceProp.major] * deviceProp.multiProcessorCount);
    #endif
//        fprintf(stderr,"  Total amount of constant memory:               %u bytes\n", deviceProp.totalConstMem); 
//        fprintf(stderr,"  Total amount of shared memory per block:       %u bytes\n", deviceProp.sharedMemPerBlock);
//        fprintf(stderr,"  Total number of registers available per block: %d\n", deviceProp.regsPerBlock);
//        fprintf(stderr,"  Warp size:                                     %d\n", deviceProp.warpSize);
//        fprintf(stderr,"  Maximum number of threads per block:           %d\n", deviceProp.maxThreadsPerBlock);
//        fprintf(stderr,"  Maximum sizes of each dimension of a block:    %d x %d x %d\n",
//               deviceProp.maxThreadsDim[0],
//               deviceProp.maxThreadsDim[1],
//               deviceProp.maxThreadsDim[2]);
//        fprintf(stderr,"  Maximum sizes of each dimension of a grid:     %d x %d x %d\n",
//               deviceProp.maxGridSize[0],
//               deviceProp.maxGridSize[1],
//               deviceProp.maxGridSize[2]);
//        fprintf(stderr,"  Maximum memory pitch:                          %u bytes\n", deviceProp.memPitch);
//        fprintf(stderr,"  Texture alignment:                             %u bytes\n", deviceProp.textureAlignment);
        fprintf(stderr,"  Clock rate:                                    %.2f GHz\n", deviceProp.clockRate * 1e-6f);
//    #if CUDART_VERSION >= 2000
//        fprintf(stderr,"  Concurrent copy and execution:                 %s\n", deviceProp.deviceOverlap ? "Yes" : "No");
//    #endif
    #if CUDART_VERSION >= 2020
//        fprintf(stderr,"  Run time limit on kernels:                     %s\n", deviceProp.kernelExecTimeoutEnabled ? "Yes" : "No");
//        fprintf(stderr,"  Integrated:                                    %s\n", deviceProp.integrated ? "Yes" : "No");
        fprintf(stderr,"  Support host page-locked memory mapping:       %s\n", deviceProp.canMapHostMemory ? "Yes" : "No");
        fprintf(stderr,"  Compute mode:                                  %s\n", deviceProp.computeMode == cudaComputeModeDefault ?
			                                                            "Default (multiple host threads can use this device simultaneously)" :
		                                                                deviceProp.computeMode == cudaComputeModeExclusive ?
																		"Exclusive (only one host thread at a time can use this device)" :
		                                                                deviceProp.computeMode == cudaComputeModeProhibited ?
																		"Prohibited (no host thread can use this device)" :
																		"Unknown");
    #endif
    }
#endif    
}

inline
void output_diagnostics(void)
{
    output_compiler_diagnostics();
    output_thrust_diagnostics();
    output_device_diagnostics();
}

} // end namespace detail
} // end namespace thrust

