---
title: thrust::system::errc
parent: System Diagnostics
grand_parent: System
nav_exclude: true
has_children: true
has_toc: false
---

# Namespace `thrust::system::errc`

<code class="doxybook">
<span>namespace thrust::system::errc {</span>
<br>
<span>enum <b><a href="{{ site.baseurl }}/api/namespaces/namespacethrust_1_1system_1_1errc.html#enum-errc-t">errc&#95;t</a></b>;</span>
<span>} /* namespace thrust::system::errc */</span>
</code>

## Types

<h3 id="enum-errc-t">
Enum <code>thrust::system::errc::errc&#95;t</code>
</h3>

| Enumerator | Value | Description |
|------------|-------|-------------|
| `address_family_not_supported` | `detail::eafnosupport` |  |
| `address_in_use` | `detail::eaddrinuse` |  |
| `address_not_available` | `detail::eaddrnotavail` |  |
| `already_connected` | `detail::eisconn` |  |
| `argument_list_too_long` | `detail::e2big` |  |
| `argument_out_of_domain` | `detail::edom` |  |
| `bad_address` | `detail::efault` |  |
| `bad_file_descriptor` | `detail::ebadf` |  |
| `bad_message` | `detail::ebadmsg` |  |
| `broken_pipe` | `detail::epipe` |  |
| `connection_aborted` | `detail::econnaborted` |  |
| `connection_already_in_progress` | `detail::ealready` |  |
| `connection_refused` | `detail::econnrefused` |  |
| `connection_reset` | `detail::econnreset` |  |
| `cross_device_link` | `detail::exdev` |  |
| `destination_address_required` | `detail::edestaddrreq` |  |
| `device_or_resource_busy` | `detail::ebusy` |  |
| `directory_not_empty` | `detail::enotempty` |  |
| `executable_format_error` | `detail::enoexec` |  |
| `file_exists` | `detail::eexist` |  |
| `file_too_large` | `detail::efbig` |  |
| `filename_too_long` | `detail::enametoolong` |  |
| `function_not_supported` | `detail::enosys` |  |
| `host_unreachable` | `detail::ehostunreach` |  |
| `identifier_removed` | `detail::eidrm` |  |
| `illegal_byte_sequence` | `detail::eilseq` |  |
| `inappropriate_io_control_operation` | `detail::enotty` |  |
| `interrupted` | `detail::eintr` |  |
| `invalid_argument` | `detail::einval` |  |
| `invalid_seek` | `detail::espipe` |  |
| `io_error` | `detail::eio` |  |
| `is_a_directory` | `detail::eisdir` |  |
| `message_size` | `detail::emsgsize` |  |
| `network_down` | `detail::enetdown` |  |
| `network_reset` | `detail::enetreset` |  |
| `network_unreachable` | `detail::enetunreach` |  |
| `no_buffer_space` | `detail::enobufs` |  |
| `no_child_process` | `detail::echild` |  |
| `no_link` | `detail::enolink` |  |
| `no_lock_available` | `detail::enolck` |  |
| `no_message_available` | `detail::enodata` |  |
| `no_message` | `detail::enomsg` |  |
| `no_protocol_option` | `detail::enoprotoopt` |  |
| `no_space_on_device` | `detail::enospc` |  |
| `no_stream_resources` | `detail::enosr` |  |
| `no_such_device_or_address` | `detail::enxio` |  |
| `no_such_device` | `detail::enodev` |  |
| `no_such_file_or_directory` | `detail::enoent` |  |
| `no_such_process` | `detail::esrch` |  |
| `not_a_directory` | `detail::enotdir` |  |
| `not_a_socket` | `detail::enotsock` |  |
| `not_a_stream` | `detail::enostr` |  |
| `not_connected` | `detail::enotconn` |  |
| `not_enough_memory` | `detail::enomem` |  |
| `not_supported` | `detail::enotsup` |  |
| `operation_canceled` | `detail::ecanceled` |  |
| `operation_in_progress` | `detail::einprogress` |  |
| `operation_not_permitted` | `detail::eperm` |  |
| `operation_not_supported` | `detail::eopnotsupp` |  |
| `operation_would_block` | `detail::ewouldblock` |  |
| `owner_dead` | `detail::eownerdead` |  |
| `permission_denied` | `detail::eacces` |  |
| `protocol_error` | `detail::eproto` |  |
| `protocol_not_supported` | `detail::eprotonosupport` |  |
| `read_only_file_system` | `detail::erofs` |  |
| `resource_deadlock_would_occur` | `detail::edeadlk` |  |
| `resource_unavailable_try_again` | `detail::eagain` |  |
| `result_out_of_range` | `detail::erange` |  |
| `state_not_recoverable` | `detail::enotrecoverable` |  |
| `stream_timeout` | `detail::etime` |  |
| `text_file_busy` | `detail::etxtbsy` |  |
| `timed_out` | `detail::etimedout` |  |
| `too_many_files_open_in_system` | `detail::enfile` |  |
| `too_many_files_open` | `detail::emfile` |  |
| `too_many_links` | `detail::emlink` |  |
| `too_many_symbolic_link_levels` | `detail::eloop` |  |
| `value_too_large` | `detail::eoverflow` |  |
| `wrong_protocol_type` | `detail::eprototype` |  |

An enum containing common error codes. 


