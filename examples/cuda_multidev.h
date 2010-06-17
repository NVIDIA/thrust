#pragma once

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <stdio.h>

#define CUDA_MAX_ACTIVE_DEVICE_COUNT 64
static CUcontext contexts[CUDA_MAX_ACTIVE_DEVICE_COUNT] = {0};

inline cudaError_t cudaSetActiveDevice(int newDevice)
{
    CUresult result = CUDA_SUCCESS;
    CUcontext oldContext = NULL;
    int oldDevice = -1;
    cudaError_t error = cudaSuccess;

    error = cudaGetDevice(&oldDevice);
    if (cudaSuccess != error) {
        return error;
    }

    // save the current context to device index oldDevice
    result = cuCtxAttach(&oldContext, 0);
    if (result == CUDA_SUCCESS) {
        // make sure oldDevice only ever has 1 context associated with it
        if (contexts[oldDevice] && contexts[oldDevice] != oldContext)
        {
            fprintf(stderr, "Unexpected context for device %d.  Be afraid!\n", oldDevice);
        }
        // save this context to that device
        contexts[oldDevice] = oldContext;
        // drop oldContext's refcount (it was bumped at attach)
        result = cuCtxDetach(oldContext);
        if (CUDA_SUCCESS != result) {
            fprintf(stderr, "cuCtxDeatch failed\n");
        }
    }

    // pop the current context
    if (oldContext) {
        result = cuCtxPopCurrent(&oldContext);
        if (result != CUDA_SUCCESS) {
            fprintf(stderr, "cuCtxPopCurrent failed.\n");
            return cudaErrorUnknown;
        }
    }

    // set the runtime's active device
    error = cudaSetDevice(newDevice);
    if (cudaErrorSetOnActiveProcess == error) {
        fprintf(stderr, "cudaSetDevice failed with cudaErrorSetOnActiveProcess.\n");
    }
    if (cudaSuccess != error) {
        fprintf(stderr, "cudaSetDevice failed.\n");
        return error;
    }

    // if there's a context for this device, push it
    if (newDevice < 0 || newDevice >= CUDA_MAX_ACTIVE_DEVICE_COUNT) {
        fprintf(stderr, "Invalid device index %d \\notin [0, 64).\n", newDevice);
        return cudaErrorInvalidValue;
    }
    if (contexts[newDevice]) {
        result = cuCtxPushCurrent(contexts[newDevice]);
        if (result != CUDA_SUCCESS) {
            fprintf(stderr, "cuCtxPushCurrent failed.\n");
            return cudaErrorUnknown;
        }
    }

    return cudaSuccess;
}

inline cudaError_t cudaThreadExitActiveDevice(void)
{
    int oldDevice = -1;
    cudaError_t error = cudaSuccess;

    // get the old device index
    error = cudaGetDevice(&oldDevice);
    if (cudaSuccess != error) {
        return error;
    }

    // exit old device's context
    error = cudaThreadExit();
    if (cudaSuccess != error) {
        return error;
    }
    contexts[oldDevice] = NULL;

    return cudaSuccess;
}

