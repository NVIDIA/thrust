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

/*! \file version.h
 *  \brief Compile-time macros encoding Thrust release version.
 *
 *         <thrust/version.h> is the only Thrust header that is guaranteed to
 *         change with every thrust release.
 */

#pragma once

#include <thrust/detail/config.h>

//  This is the only thrust header that is guaranteed to 
//  change with every thrust release.
//
//  THRUST_VERSION % 100 is the sub-minor version
//  THRUST_VERSION / 100 % 1000 is the minor version
//  THRUST_VERSION / 100000 is the major version

/*! \def THRUST_VERSION
 *  \brief The preprocessor macro \p THRUST_VERSION encodes the version
 *         number of the Thrust library.
 *
 *         <tt>THRUST_VERSION % 100</tt> is the sub-minor version.
 *         <tt>THRUST_VERSION / 100 % 1000</tt> is the minor version.
 *         <tt>THRUST_VERSION / 100000</tt> is the major version.
 */
#define THRUST_VERSION 100400

/*! \def THRUST_MAJOR_VERSION
 *  \brief The preprocessor macro \p THRUST_MAJOR_VERSION encodes the
 *         major version number of the Thrust library.
 */
#define THRUST_MAJOR_VERSION     (THRUST_VERSION / 100000)

/*! \def THRUST_MINOR_VERSION
 *  \brief The preprocessor macro \p THRUST_MINOR_VERSION encodes the
 *         minor version number of the Thrust library.
 */
#define THRUST_MINOR_VERSION     (THRUST_VERSION / 100 % 1000)

/*! \def THRUST_SUBMINOR_VERSION
 *  \brief The preprocessor macro \p THRUST_SUBMINOR_VERSION encodes the
 *         sub-minor version number of the Thrust library.
 */
#define THRUST_SUBMINOR_VERSION  (THRUST_VERSION % 100)

// Declare these namespaces here for the purpose of Doxygenating them

/*! \namespace thrust
 *  \brief \p thrust is the top-level namespace which contains all Thrust
 *         functions and types.
 */
namespace thrust
{

/*! \namespace deprecated
 *  \brief The \p deprecated namespace contains functionality which is scheduled
 *         to be removed in a future version of Thrust.
 *
 *         Users of functionality in namespace \p deprecated are encouraged to
 *         migrate their code to functionality offered elsewhere.
 */
namespace deprecated
{
}

/*! \namespace next
 *  \brief The \p next namespace contains functionality which is scheduled to be
 *         promoted to the top-level \p thrust namespace in a future version of
 *         Thrust.
 */
namespace next
{
}
  
}

