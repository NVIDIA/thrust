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


/*! \file system_error.h
 *  \brief An exception object used to report error conditions that have an
 *         associated error code.
 */

#pragma once

#include <thrust/detail/config.h>
#include <stdexcept>
#include <string>

#include <thrust/system/system_code.h>

namespace thrust
{

namespace experimental
{

namespace system
{

// [19.5.5] Class system_error

// [19.5.5.1] Class system_error overview

/*! \brief The class \p system_error describes an exception object used to report error
 *  conditions that have an associated \p error_code. Such error conditions typically
 *  originate from the operating system or other low-level application program interfaces.
 *
 *  \note If an error represents an out-of-memory condition, implementations are encouraged
 *  to throw an exception object of type \p std::bad_alloc rather than \p system_error.
 */
class system_error
  : public std::runtime_error
{
  public:
    // [19.5.5.2] Class system_error members
    
    inline system_error(error_code ec, const std::string &what_arg);

    inline system_error(error_code ec, const char *what_arg);

    inline system_error(error_code ec);

    inline system_error(int ev, const error_category &ecat, const std::string &what_arg);

    inline system_error(int ev, const error_category &ecat, const char *what_arg);

    inline system_error(int ev, const error_category &ecat);
    
    inline const error_code &code(void) const throw();

    inline const char *what(void) const throw();

    /*! \cond
     */
  private:
    error_code          m_error_code;
    mutable std::string m_what;

    /*! \endcond
     */
}; // end system_error

} // end system

// import names into thrust::
using system::system_error;

} // end namespace experimental

} // end thrust

#include <thrust/system/detail/system_error.inl>

