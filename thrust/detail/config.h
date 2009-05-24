/*! \file config.h
 *  \brief Defines platform configuration.
 */

#pragma once

#ifndef __CUDACC__

// if we're not compiling with nvcc,
// #include this to define what __host__ and __device__ mean
// XXX ideally, we wouldn't require an installation of CUDA
#include <host_defines.h>

#endif // __CUDACC__

