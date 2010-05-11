/*
 *  Copyright 2008-2010 NVIDIA Corporation
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

#include <thrust/system/error_code.h>
#include <thrust/functional.h>
#include <cstring>

namespace thrust
{

namespace experimental
{

namespace system
{

error_category
  ::~error_category(void)
{
  ;
} // end error_category::~error_category()


error_condition error_category
  ::default_error_condition(int ev) const
{
  return error_condition(ev, *this);
} // end error_category::default_error_condition()


bool error_category
  ::equivalent(int code, const error_condition &condition) const
{
  return default_error_condition(code) == condition;
} // end error_condition::equivalent()


bool error_category
  ::equivalent(const error_code &code, int condition) const
{
  bool result = (this->operator==(code.category())) && (code.value() == condition);
  return result;
} // end error_code::equivalent()


bool error_category
  ::operator==(const error_category &rhs) const
{
  return this == &rhs;
} // end error_category::operator==()


bool error_category
  ::operator!=(const error_category &rhs) const
{
  return !this->operator==(rhs);
} // end error_category::operator!=()


bool error_category
  ::operator<(const error_category &rhs) const
{
  return thrust::less<const error_category*>()(this,&rhs);
} // end error_category::operator<()


namespace detail
{


class generic_error_category
  : public error_category
{
  public:
    inline generic_error_category(void) {}

    inline virtual const char *name(void) const
    {
      return "generic";
    }

    inline virtual std::string message(int ev) const
    {
      static const std::string unknown_err("Unknown error");

      // XXX strerror is not thread-safe:
      //     prefer strerror_r (which is not provided on windows)
      const char *c_str = std::strerror(ev);
      return c_str ? std::string(c_str) : unknown_err;
    }
}; // end generic_category_result


class system_error_category
  : public error_category
{
  public:
    inline system_error_category(void) {}

    inline virtual const char *name(void) const
    {
      return "system";
    }

    inline virtual std::string message(int ev) const
    {
      return generic_category().message(ev);
    }

    inline virtual error_condition default_error_condition(int ev) const
    {
      using namespace errc;

      switch(ev)
      {
// XXX eventually implement Windows error codes
#if THRUST_HOST_COMPILER == THRUST_HOST_COMPILER_GCC
        case EAFNOSUPPORT:    return make_error_condition(address_family_not_supported);
        case EADDRINUSE:      return make_error_condition(address_in_use);
        case EADDRNOTAVAIL:   return make_error_condition(address_not_available);
        case EISCONN:         return make_error_condition(already_connected);
        case E2BIG:           return make_error_condition(argument_list_too_long);
        case EDOM:            return make_error_condition(argument_out_of_domain);
        case EFAULT:          return make_error_condition(bad_address);
        case EBADF:           return make_error_condition(bad_file_descriptor);
        case EBADMSG:         return make_error_condition(bad_message);
        case EPIPE:           return make_error_condition(broken_pipe);
        case ECONNABORTED:    return make_error_condition(connection_aborted);
        case EALREADY:        return make_error_condition(connection_already_in_progress);
        case ECONNREFUSED:    return make_error_condition(connection_refused);
        case ECONNRESET:      return make_error_condition(connection_reset);
        case EXDEV:           return make_error_condition(cross_device_link);
        case EDESTADDRREQ:    return make_error_condition(destination_address_required);
        case EBUSY:           return make_error_condition(device_or_resource_busy);
        case ENOTEMPTY:       return make_error_condition(directory_not_empty);
        case ENOEXEC:         return make_error_condition(executable_format_error);
        case EEXIST:          return make_error_condition(file_exists);
        case EFBIG:           return make_error_condition(file_too_large);
        case ENAMETOOLONG:    return make_error_condition(filename_too_long);
        case ENOSYS:          return make_error_condition(function_not_supported);
        case EHOSTUNREACH:    return make_error_condition(host_unreachable);
        case EIDRM:           return make_error_condition(identifier_removed);
        case EILSEQ:          return make_error_condition(illegal_byte_sequence);
        case ENOTTY:          return make_error_condition(inappropriate_io_control_operation);
        case EINTR:           return make_error_condition(interrupted);
        case EINVAL:          return make_error_condition(invalid_argument);
        case ESPIPE:          return make_error_condition(invalid_seek);
        case EIO:             return make_error_condition(io_error);
        case EISDIR:          return make_error_condition(is_a_directory);
        case EMSGSIZE:        return make_error_condition(message_size);
        case ENETDOWN:        return make_error_condition(network_down);
        case ENETRESET:       return make_error_condition(network_reset);
        case ENETUNREACH:     return make_error_condition(network_unreachable);
        case ENOBUFS:         return make_error_condition(no_buffer_space);
        case ECHILD:          return make_error_condition(no_child_process);
        case ENOLINK:         return make_error_condition(no_link);
        case ENOLCK:          return make_error_condition(no_lock_available);
        case ENODATA:         return make_error_condition(no_message_available);
        case ENOMSG:          return make_error_condition(no_message);
        case ENOPROTOOPT:     return make_error_condition(no_protocol_option);
        case ENOSPC:          return make_error_condition(no_space_on_device);
        case ENOSR:           return make_error_condition(no_stream_resources);
        case ENXIO:           return make_error_condition(no_such_device_or_address);
        case ENODEV:          return make_error_condition(no_such_device);
        case ENOENT:          return make_error_condition(no_such_file_or_directory);
        case ESRCH:           return make_error_condition(no_such_process);
        case ENOTDIR:         return make_error_condition(not_a_directory);
        case ENOTSOCK:        return make_error_condition(not_a_socket);
        case ENOSTR:          return make_error_condition(not_a_stream);
        case ENOTCONN:        return make_error_condition(not_connected);
        case ENOMEM:          return make_error_condition(not_enough_memory);
        case ENOTSUP:         return make_error_condition(not_supported);
        case ECANCELED:       return make_error_condition(operation_canceled);
        case EINPROGRESS:     return make_error_condition(operation_in_progress);
        case EPERM:           return make_error_condition(operation_not_permitted);

// EOPNOTSUPP is the same as ENOTSUP on Linux
#if EOPNOTSUPP != ENOTSUP
        case EOPNOTSUPP:      return make_error_condition(operation_not_supported);
#endif

        case EWOULDBLOCK:     return make_error_condition(operation_would_block);

// EOWNERDEAD is missing on Darwin
#ifdef EOWNERDEAD
        case EOWNERDEAD:      return make_error_condition(owner_dead);
#endif // EOWNERDEAD

        case EACCES:          return make_error_condition(permission_denied);
        case EPROTO:          return make_error_condition(protocol_error);
        case EPROTONOSUPPORT: return make_error_condition(protocol_not_supported);
        case EROFS:           return make_error_condition(read_only_file_system);
        case EDEADLK:         return make_error_condition(resource_deadlock_would_occur);

// EAGAIN is the same as EWOULDBLOCK on Darwin
#if EAGAIN != EWOULDBLOCK
        case EAGAIN:          return make_error_condition(resource_unavailable_try_again);
#endif

        case ERANGE:          return make_error_condition(result_out_of_range);

// ENOTRECOVERABLE is missing on Darwin
#ifdef ENOTRECOVERABLE
        case ENOTRECOVERABLE: return make_error_condition(state_not_recoverable);
#endif

        case ETIME:           return make_error_condition(stream_timeout);
        case ETXTBSY:         return make_error_condition(text_file_busy);
        case ETIMEDOUT:       return make_error_condition(timed_out);
        case ENFILE:          return make_error_condition(too_many_files_open_in_system);
        case EMFILE:          return make_error_condition(too_many_files_open);
        case EMLINK:          return make_error_condition(too_many_links);
        case ELOOP:           return make_error_condition(too_many_symbolic_link_levels);
        case EOVERFLOW:       return make_error_condition(value_too_large);
        case EPROTOTYPE:      return make_error_condition(wrong_protocol_type);
#endif // THRUST_HOST_COMPILER == THRUST_HOST_COMPILER_GCC

        default:              return error_condition(ev,system_category());
      }
    }
}; // end system_category_result


} // end detail


const error_category &generic_category(void)
{
  static const detail::generic_error_category result;
  return result;
}


const error_category &system_category(void)
{
  static const detail::system_error_category result;
  return result;
}


} // end system

} // end experimental

} // end thrust

