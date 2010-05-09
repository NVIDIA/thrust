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


/*! \file error_code.h
 *  \brief An object used to hold error values, such as those originating from the
 *         operating system or other low-level application program interfaces.
 */

#pragma once

#include <thrust/detail/config.h>
#include <thrust/detail/type_traits.h>
#include <thrust/system/detail/posix_errno.h>
#include <errno.h>
#include <iostream>

namespace thrust
{

namespace experimental
{

namespace system
{


class error_condition;
class error_code;
class error_condition;


template<typename T> struct is_error_code_enum : public thrust::detail::false_type {};

template<typename T> struct is_error_condition_enum : public thrust::detail::false_type {};


// XXX N3000 prefers enum class errc { ... }
namespace errc
{

enum errc_t
{
  address_family_not_supported       = EAFNOSUPPORT,
  address_in_use                     = EADDRINUSE,
  address_not_available              = EADDRNOTAVAIL,
  already_connected                  = EISCONN,
  argument_list_too_long             = E2BIG,
  argument_out_of_domain             = EDOM,
  bad_address                        = EFAULT,
  bad_file_descriptor                = EBADF,
  bad_message                        = EBADMSG,
  broken_pipe                        = EPIPE,
  connection_aborted                 = ECONNABORTED,
  connection_already_in_progress     = EALREADY,
  connection_refused                 = ECONNREFUSED,
  connection_reset                   = ECONNRESET,
  cross_device_link                  = EXDEV,
  destination_address_required       = EDESTADDRREQ,
  device_or_resource_busy            = EBUSY,
  directory_not_empty                = ENOTEMPTY,
  executable_format_error            = ENOEXEC,
  file_exists                        = EEXIST,
  file_too_large                     = EFBIG,
  filename_too_long                  = ENAMETOOLONG,
  function_not_supported             = ENOSYS,
  host_unreachable                   = EHOSTUNREACH,
  identifier_removed                 = EIDRM,
  illegal_byte_sequence              = EILSEQ,
  inappropriate_io_control_operation = ENOTTY,
  interrupted                        = EINTR,
  invalid_argument                   = EINVAL,
  invalid_seek                       = ESPIPE,
  io_error                           = EIO,
  is_a_directory                     = EISDIR,
  message_size                       = EMSGSIZE,
  network_down                       = ENETDOWN,
  network_reset                      = ENETRESET,
  network_unreachable                = ENETUNREACH,
  no_buffer_space                    = ENOBUFS,
  no_child_process                   = ECHILD,
  no_link                            = ENOLINK,
  no_lock_available                  = ENOLCK,
  no_message_available               = ENODATA,
  no_message                         = ENOMSG,
  no_protocol_option                 = ENOPROTOOPT,
  no_space_on_device                 = ENOSPC,
  no_stream_resources                = ENOSR,
  no_such_device_or_address          = ENXIO,
  no_such_device                     = ENODEV,
  no_such_file_or_directory          = ENOENT,
  no_such_process                    = ESRCH,
  not_a_directory                    = ENOTDIR,
  not_a_socket                       = ENOTSOCK,
  not_a_stream                       = ENOSTR,
  not_connected                      = ENOTCONN,
  not_enough_memory                  = ENOMEM,
  not_supported                      = ENOTSUP,
  operation_canceled                 = ECANCELED,
  operation_in_progress              = EINPROGRESS,
  operation_not_permitted            = EPERM,
  operation_not_supported            = EOPNOTSUPP,
  operation_would_block              = EWOULDBLOCK,

// EOWNERDEAD is missing on Darwin
#ifdef EOWNERDEAD
  owner_dead                         = EOWNERDEAD,
#endif

  permission_denied                  = EACCES,
  protocol_error                     = EPROTO,
  protocol_not_supported             = EPROTONOSUPPORT,
  read_only_file_system              = EROFS,
  resource_deadlock_would_occur      = EDEADLK,
  resource_unavailable_try_again     = EAGAIN,
  result_out_of_range                = ERANGE,

// ENOTRECOVERABLE is missing on Darwin
#ifdef ENOTRECOVERABLE
  state_not_recoverable              = ENOTRECOVERABLE,
#endif

  stream_timeout                     = ETIME,
  text_file_busy                     = ETXTBSY,
  timed_out                          = ETIMEDOUT,
  too_many_files_open_in_system      = ENFILE,
  too_many_files_open                = EMFILE,
  too_many_links                     = EMLINK,
  too_many_symbolic_link_levels      = ELOOP,
  value_too_large                    = EOVERFLOW,
  wrong_protocol_type                = EPROTOTYPE,
}; // end errc_t

} // end namespace errc


template<> struct is_error_condition_enum<errc::errc_t> : public thrust::detail::true_type {};


// [19.5.1.1] class error_category

class error_category
{
  public:
    inline virtual ~error_category(void);

    // XXX enable upon c++0x
    // error_category(const error_category &) = delete;
    // error_category &operator=(const error_category &) = delete;

    /*! \return A string naming the error category.
     */
    inline virtual const char *name(void) const = 0;

    /*! \return \p error_condition(ev, *this).
     */
    inline virtual error_condition default_error_condition(int ev) const;

    /*! \return <tt>default_error_condition(code) == condition</tt>
     */
    inline virtual bool equivalent(int code, const error_condition &condition) const;

    /*! \return <tt>*this == code.category() && code.value() == condition</tt>
     */
    inline virtual bool equivalent(const error_code &code, int condition) const;

    /*! \return A string that describes the error condition denoted by \p ev.
     */
    virtual std::string message(int ev) const = 0;

    /*! \return <tt>*this == &rhs</tt>
     */
    inline bool operator==(const error_category &rhs) const;

    /*! \return <tt>!(*this == rhs)</tt>
     */
    inline bool operator!=(const error_category &rhs) const;

    /*! \return <tt>less<const error_category*>()(this, &rhs)</tt>
     *  \note \c less provides a total ordering for pointers.
     */
    inline bool operator<(const error_category &rhs) const;
}; // end error_category


// [19.5.1.5] error_category objects


/*! \return A reference to an object of a type derived from class \p error_category.
 *  \note The object's \p default_error_condition and \p equivalent virtual functions
 *        shall behave as specified for the class \p error_category. The object's
 *        \p name virtual function shall return a pointer to the string <tt>"generic"</tt>.
 */
inline const error_category &generic_category(void);


/*! \return A reference to an object of a type derived from class \p error_category.
 *  \note The object's \p equivalent virtual functions shall behave as specified for
 *        class \p error_category. The object's \p name virtual function shall return
 *        a pointer to the string <tt>"system"</tt>. The object's \p default_error_condition
 *        virtual function shall behave as follows:
 *
 *        If the argument <tt>ev</tt> corresponds to a POSIX <tt>errno</tt> value
 *        \c posv, the function shall return <tt>error_condition(ev,generic_category())</tt>.
 *        Otherwise, the function shall return <tt>error_condition(ev,system_category())</tt>.
 *        What constitutes correspondence for any given operating system is unspecified.
 */
inline const error_category &system_category(void);


// [19.5.2] Class error_code


class error_code
{
  public:
    // [19.5.2.2] constructors:

    /*! Effects: Constructs an object of type \p error_code.
     *  Postconditions: <tt>value() == 0</tt> and <tt>category() == &system_category()</tt>.
     */
    inline error_code(void);

    /*! Effects: Constructs an object of type \p error_code.
     *  Postconditions: <tt>value() == val</tt> and <tt>category() == &cat</tt>.
     */
    inline error_code(int val, const error_category &cat);

    /*! Effects: Constructs an object of type \p error_code.
     *  Postconditions: <tt>*this == make_error_code(e)<tt>.
     */
    template <typename ErrorCodeEnum>
      error_code(ErrorCodeEnum e,
        typename thrust::detail::enable_if<is_error_code_enum<ErrorCodeEnum>::value>::type * = 0);

    // [19.5.2.3] modifiers:

    /*! Postconditions: <tt>value() == val</tt> and <tt>category() == &cat</tt>.
     */
    inline void assign(int val, const error_category &cat);

    /*! Postconditions: <tt>*this == make_error_code(e)</tt>.
     */
    template <typename ErrorCodeEnum>
      typename thrust::detail::enable_if<is_error_code_enum<ErrorCodeEnum>::value, error_code>::type &
        operator=(ErrorCodeEnum e);

    /*! Postconditions: <tt>value() == 0</tt> and <tt>category() == system_category()</tt>.
     */
    inline void clear(void);

    // [19.5.2.4] observers:

    /*! \return An integral value of this \p error_code object.
     */
    inline int value(void) const;

    /*! \return An \p error_category describing the category of this \p error_code object.
     */
    inline const error_category &category(void) const;

    /*! \return <tt>category().default_error_condition()</tt>.
     */
    inline error_condition default_error_condition(void) const;

    /*! \return <tt>category().message(value())</tt>.
     */
    inline std::string message(void) const;

    // XXX replace the below upon c++0x
    // inline explicit operator bool (void) const;

    /*! \return <tt>value() != 0</tt>.
     */
    inline operator bool (void) const;

    /*! \cond
     */
  private:
    int m_val;
    const error_category *m_cat;
    /*! \endcond
     */
}; // end error_code


// [19.5.2.5] Class error_code non-member functions


// XXX replace errc::errc_t with errc upon c++0x
/*! \return <tt>error_code(static_cast<int>(e), generic_category())</tt>
 */
inline error_code make_error_code(errc::errc_t e);


/*! \return <tt>lhs.category() < rhs.category() || lhs.category() == rhs.category() && lhs.value() < rhs.value()</tt>.
 */
inline bool operator<(const error_code &lhs, const error_code &rhs);


/*! Effects: <tt>os << ec.category().name() << ':' << ec.value()</tt>.
 */
template <typename charT, typename traits>
  std::basic_ostream<charT,traits>&
    operator<<(std::basic_ostream<charT,traits>& os, const error_code &ec);


// [19.5.3] class error_condition


/*! The class \p error_condition describes an object used to hold values identifying
 *  error conditions.
 *
 *  \note \p error_condition values are portable abstractions, while \p error_code values
 *        are implementation specific.
 */
class error_condition
{
  public:
    // [19.5.3.2] constructors
    inline error_condition(void);

    inline error_condition(int val, const error_category &cat);

    template<typename ErrorConditionEnum>
      error_condition(ErrorConditionEnum e,
        typename thrust::detail::enable_if<is_error_condition_enum<ErrorConditionEnum>::value>::type * = 0);

    // [19.5.3.3] modifiers
    inline void assign(int val, const error_category &cat);

    template<typename ErrorConditionEnum>
      typename thrust::detail::enable_if<is_error_condition_enum<ErrorConditionEnum>::value, error_code>::type &
        operator=(ErrorConditionEnum e);

    inline void clear(void);

    // [19.5.3.4] observers
    inline int value(void) const;

    inline const error_category &category(void) const;

    inline std::string message(void) const;

    // XXX replace below with this upon c++0x
    //explicit operator bool (void) const;
    
    inline operator bool (void) const;

    /*! \cond
     */

  private:
    int m_val;
    const error_category *m_cat;

    /*! \endcond
     */
}; // end error_condition



// [19.5.3.5] Class error_condition non-member functions

// XXX replace errc::errc_t with errc upon c++0x
/*! \return <tt>error_condition(static_cast<int>(e), generic_category())</tt>.
 */
inline error_condition make_error_condition(errc::errc_t e);


/*! \return <tt>lhs.category() < rhs.category() || lhs.category() == rhs.category() && lhs.value() < rhs.value()</tt>.
 */
inline bool operator<(const error_condition &lhs, const error_condition &rhs);


// [19.5.4] Comparison operators


/*! \return <tt>lhs.category() == rhs.category() && lhs.value() == rhs.value()</tt>.
 */
inline bool operator==(const error_code &lhs, const error_code &rhs);


/*! \return <tt>lhs.category().equivalent(lhs.value(), rhs) || rhs.category().equivalent(lhs,rhs.value())</tt>.
 */
inline bool operator==(const error_code &lhs, const error_condition &rhs);


/*! \return <tt>rhs.category().equivalent(lhs.value(), lhs) || lhs.category().equivalent(rhs, lhs.value())</tt>.
 */
inline bool operator==(const error_condition &lhs, const error_code &rhs);


/*! \return <tt>lhs.category() == rhs.category() && lhs.value() == rhs.value()</tt>
 */
inline bool operator==(const error_condition &lhs, const error_condition &rhs);


/*! \return <tt>!(lhs == rhs)</tt>
 */
inline bool operator!=(const error_code &lhs, const error_code &rhs);


/*! \return <tt>!(lhs == rhs)</tt>
 */
inline bool operator!=(const error_code &lhs, const error_condition &rhs);


/*! \return <tt>!(lhs == rhs)</tt>
 */
inline bool operator!=(const error_condition &lhs, const error_code &rhs);


/*! \return <tt>!(lhs == rhs)</tt>
 */
inline bool operator!=(const error_condition &lhs, const error_condition &rhs);


} // end system

// import names into thrust::
using system::error_category;
using system::error_code;
using system::error_condition;
using system::is_error_code_enum;
using system::is_error_condition_enum;
using system::make_error_code;
using system::make_error_condition;

// XXX replace with using system::errc upon c++0x
namespace errc = system::errc;

using system::generic_category;
using system::system_category;

} // end experimental

} // end thrust

#include <thrust/system/detail/error_category.inl>
#include <thrust/system/detail/error_code.inl>
#include <thrust/system/detail/error_condition.inl>

