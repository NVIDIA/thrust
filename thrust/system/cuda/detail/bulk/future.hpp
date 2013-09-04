/*
 *  Copyright 2008-2013 NVIDIA Corporation
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

#pragma once

#include <thrust/system/cuda/detail/bulk/detail/config.hpp>
#include <cuda_runtime_api.h>
#include <utility>
#include <stdexcept>
#include <iostream>

BULK_NAMESPACE_PREFIX
namespace bulk
{


typedef std::runtime_error future_error;


namespace detail
{


struct future_core_access;


namespace future_detail
{


inline void throw_on_error(cudaError_t e, const char *message)
{
  if(e != cudaSuccess)
  {
    throw future_error(message);
  } // end if
} // end throw_on_error()


} // end future_detail
} // end detail


template<typename T> class future;


template<>
class future<void>
{
  public:
    ~future()
    {
      if(valid())
      {
        // swallow errors
        cudaError_t e = cudaEventDestroy(m_event);

        if(e)
        {
          std::cerr << "CUDA error after cudaEventDestroy in future dtor: " << cudaGetErrorString(e) << std::endl;
        } // end if

        if(m_owns_stream)
        {
          e = cudaStreamDestroy(m_stream);

          if(e)
          {
            std::cerr << "CUDA error after cudaStreamDestroy in future dtor: " << cudaGetErrorString(e) << std::endl;
          } // end if
        } // end if
      } // end if
    } // end ~future()

    void wait() const
    {
      // XXX need to capture the error as an exception and then throw it in .get()
      detail::future_detail::throw_on_error(cudaEventSynchronize(m_event), "cudaEventSynchronize in future::wait");

      // XXX upon c++11
      //cudaError_t status = cudaEventQuery(m_event);
      //while(status == cudaErrorNotReady)
      //{
      //  // sleep
      //  std::this_thread::sleep_for(std::chrono::milliseconds(10));
      //} // end while

      //detail::future_detail::throw_on_error(status, "cudaEventQuery in future::wait");
    } // end wait()

    bool valid() const
    {
      return m_event != 0;
    } // end valid()

    future()
      : m_stream(0), m_event(0), m_owns_stream(false)
    {}

    // simulate a move
    // XXX need to add rval_ref or something
    future(const future &other)
      : m_stream(0), m_event(0), m_owns_stream(false)
    {
      std::swap(m_stream,      const_cast<future&>(other).m_stream);
      std::swap(m_event,       const_cast<future&>(other).m_event);
      std::swap(m_owns_stream, const_cast<future&>(other).m_owns_stream);
    } // end future()

    // simulate a move
    // XXX need to add rval_ref or something
    future &operator=(const future &other)
    {
      std::swap(m_stream,      const_cast<future&>(other).m_stream);
      std::swap(m_event,       const_cast<future&>(other).m_event);
      std::swap(m_owns_stream, const_cast<future&>(other).m_owns_stream);
      return *this;
    } // end operator=()

  private:
    friend class detail::future_core_access;

    explicit future(cudaStream_t s, bool owns_stream)
      : m_stream(s),m_owns_stream(owns_stream)
    {
      detail::future_detail::throw_on_error(cudaEventCreateWithFlags(&m_event, create_flags), "cudaEventCreateWithFlags in future ctor");
      detail::future_detail::throw_on_error(cudaEventRecord(m_event, m_stream), "cudaEventRecord in future ctor");
    } // end future()

    // XXX this combination makes the constructor expensive
    //static const int create_flags = cudaEventDisableTiming | cudaEventBlockingSync;
    static const int create_flags = cudaEventDisableTiming;

    cudaStream_t m_stream;
    cudaEvent_t m_event;
    bool m_owns_stream;
}; // end future<void>


namespace detail
{


struct future_core_access
{
  inline static future<void> create(cudaStream_t s, bool owns_stream)
  {
    return future<void>(s, owns_stream);
  } // end create_in_stream()

  inline static cudaEvent_t event(const future<void> &f)
  {
    return f.m_event;
  } // end event()
}; // end future_core_access


} // end detail


} // end namespace bulk
BULK_NAMESPACE_SUFFIX

