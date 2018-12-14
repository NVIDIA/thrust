/*
 *  Copyright 2008-2018 NVIDIA Corporation
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

/// \file thrust/detail/future_error.h
/// \brief \c thrust::future error handling types and codes.

#pragma once

#include <thrust/detail/config.h>
#include <thrust/detail/cpp11_required.h>

#if THRUST_CPP_DIALECT >= 2011

#include <thrust/detail/type_traits.h>
#include <thrust/system/error_code.h>

#include <stdexcept>

THRUST_BEGIN_NS

enum class future_errc
{
  unknown_future_error
, no_state
, last_future_error
};

/// \return <tt>error_code(static_cast<int>(e), future_category())</tt>
inline error_code make_error_code(future_errc e);

/// \return <tt>error_condition(static_cast<int>(e), future_category())</tt>.
inline error_condition make_error_condition(future_errc e);

struct future_error_category : error_category
{
  future_error_category() = default;

  virtual char const* name() const
  {
    return "future";
  }

  virtual std::string message(int ev) const
  {
    switch (static_cast<future_errc>(ev))
    {
      case future_errc::no_state:
      {
        return "no_state: an operation has been performed on a moved-from or "
               "default constructed future object";
      }
      default:
      {
        return "unknown_future_error: an unknown error with a future object "
               "has occurred";
      }
    };
  }

  virtual error_condition default_error_condition(int ev) const
  {
    if (future_errc::last_future_error > static_cast<future_errc>(ev))
      return make_error_condition(static_cast<future_errc>(ev));

    return system_category().default_error_condition(ev);
  }
}; 

/// Obtains a reference to the static error category object for the errors
/// related to futures and promises. The object is required to override the
/// virtual function error_category::name() to return a pointer to the string
/// "future". It is used to identify error codes provided in the exceptions of
/// type future_error. 
inline error_category const& future_category()
{
  static const future_error_category result;
  return result;
}

/// Specialization of \p is_error_code_enum for \p future_errc.
template<> struct is_error_code_enum<future_errc> : true_type {};

/// \return <tt>error_code(static_cast<int>(e), future_category())</tt>
inline error_code make_error_code(future_errc e)
{
  return error_code(static_cast<int>(e), future_category());
}

/// \return <tt>error_condition(static_cast<int>(e), future_category())</tt>.
inline error_condition make_error_condition(future_errc e)
{
  return error_condition(static_cast<int>(e), future_category());
} 

struct future_error : std::logic_error
{
  __host__
  explicit future_error(error_code ec)
    : std::logic_error(ec.message()), ec_(ec)
  {}

  __host__
  explicit future_error(future_errc e)
    : future_error(make_error_code(e))
  {}

  __host__
  error_code const& code() const noexcept
  {
    return ec_;
  }

  __host__
  virtual ~future_error() noexcept {}

private:
  error_code ec_;
};

inline bool operator==(future_error const& lhs, future_error const& rhs) noexcept
{
  return lhs.code() == rhs.code();
}

inline bool operator<(future_error const& lhs, future_error const& rhs) noexcept
{
  return lhs.code() < rhs.code();
}

THRUST_END_NS

#endif // THRUST_CPP_DIALECT >= 2011
