# Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
#
# NOTICE TO USER:
#
# This source code is subject to NVIDIA ownership rights under U.S. and
# international Copyright laws.
#
# This software and the information contained herein is being provided
# under the terms and conditions of a Source Code License Agreement.
#
# NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE
# CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR
# IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH
# REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF
# MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
# IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL,
# OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS
# OF USE, DATA OR PROFITS,  WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
# OR OTHER TORTIOUS ACTION,  ARISING OUT OF OR IN CONNECTION WITH THE USE
# OR PERFORMANCE OF THIS SOURCE CODE.
#
# U.S. Government End Users.   This source code is a "commercial item" as
# that term is defined at  48 C.F.R. 2.101 (OCT 1995), consisting  of
# "commercial computer  software"  and "commercial computer software
# documentation" as such terms are  used in 48 C.F.R. 12.212 (SEPT 1995)
# and is provided to the U.S. Government only as a commercial end item.
# Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through
# 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the
# source code with only those rights set forth herein.

# Makefile for building Thrust unit test driver

# Force C++11 mode. NVCC will ignore it if the host compiler doesn't support it.
#export CXX_STD = c++11

export VERBOSE = 1

ifndef PROFILE
  ifdef VULCAN_TOOLKIT_BASE
    include $(VULCAN_TOOLKIT_BASE)/build/getprofile.mk
    include $(VULCAN_TOOLKIT_BASE)/build/config/$(PROFILE).mk
  else
    include ../build/getprofile.mk
    include ../build/config/$(PROFILE).mk
  endif
endif

SOLNDIR := .

ifdef VULCAN_TOOLKIT_BASE
  include $(VULCAN_TOOLKIT_BASE)/build/config/DetectOS.mk
else
  include ../build/config/DetectOS.mk
endif

ifeq ($(OS),win32)
  export I_AM_SLOPPY := 1
endif

TMP_DIR      := built
TMP_PREFIX   := $(ROOTDIR)
TMP_ARCH     := $(ARCH)_$(PROFILE)_agnostic
THRUST_MKDIR := $(TMP_PREFIX)/$(TMP_DIR)/$(TMP_ARCH)/thrust/mk
THRUST_DIR   := $(ROOTDIR)/thrust

res:=$(shell $(PYTHON) ./generate_mk.py $(THRUST_MKDIR) $(THRUST_DIR))

# Use these environment variables to control what gets built:
#
#   TEST_ALL
#   TEST_UNITTESTS
#   TEST_EXAMPLES
#   TEST_BENCH
#   TEST_OTHER

ifneq ($(TEST_ALL),)
  override TEST_UNITTESTS := 1
  override TEST_EXAMPLES := 1
  override TEST_BENCH := 1
  override TEST_OTHER := 1
endif

ifeq ($(TEST_UNITTESTS)$(TEST_EXAMPLES)$(TEST_BENCH)$(TEST_OTHER),)
  override TEST_UNITTESTS := 1
  override TEST_EXAMPLES := 1
  override TEST_BENCH := 1
  override TEST_OTHER := 1
endif

ifneq ($(TEST_OTHER),)
  PROJECTS += internal/build/warningstester
endif

ifneq ($(TEST_BENCH),)
  PROJECTS += internal/benchmark/bench
endif

ifneq ($(TEST_UNITTESTS),)
  # copy existing projects
  PROJECTS_COPY := $(PROJECTS)

  # empty PROJECTS
  PROJECTS :=

  # populate PROJECTS with unit tests.
  include $(THRUST_MKDIR)/testing.mk

  # Once PROJECTS is populated with unit tests, re-add the previous projects.
  PROJECTS += $(PROJECTS_COPY)
endif

ifneq ($(TEST_EXAMPLES),)
  # Copy existing projects.
  PROJECTS_COPY := $(PROJECTS)

  # Empty PROJECTS.
  PROJECTS :=

	# Populate PROJECTS with examples.
  include $(THRUST_MKDIR)/examples.mk

  # Once PROJECTS is populated with examples, re-add the previous projects.
  PROJECTS += $(PROJECTS_COPY)
endif

ifdef VULCAN_TOOLKIT_BASE
  include $(VULCAN_TOOLKIT_BASE)/build/common.mk
else
  include ../build/common.mk
endif

# Print host compiler version.

VERSION_FLAG :=
ifeq ($(OS),$(filter $(OS),Linux Darwin))
	ifdef USEPGCXX        # PGI
		VERSION_FLAG := -V
	else
		ifdef USEXLC        # XLC
			VERSION_FLAG := -qversion
		else                # GCC, ICC or Clang AKA the sane ones.
			VERSION_FLAG := --version
		endif
	endif
else ifeq ($(OS),win32) # MSVC
	# cl.exe run without any options will print its version info and exit.
	VERSION_FLAG :=
endif

CCBIN_ENVIRONMENT :=
ifeq ($(OS), QNX)
	# QNX's GCC complains if QNX_HOST and QNX_TARGET aren't defined in the
	# environment.
	CCBIN_ENVIRONMENT := QNX_HOST=$(QNX_HOST) QNX_TARGET=$(QNX_TARGET)
endif

$(info #### CCBIN         : $(CCBIN))
$(info #### CCBIN VERSION : $(shell $(CCBIN_ENVIRONMENT) $(CCBIN) $(VERSION_FLAG)))
$(info #### CXX_STD       : $(CXX_STD))

ifeq ($(OS), win32)
  CREATE_DVS_PACKAGE = $(ZIP) -r built/CUDA-thrust-package.zip bin thrust/internal/test thrust/internal/scripts thrust/internal/benchmark thrust/*.trs $(DVS_COMMON_TEST_PACKAGE_FILES)
  APPEND_HEADERS_DVS_PACKAGE = $(ZIP) -rg built/CUDA-thrust-package.zip thrust -9 -i *.h
  APPEND_INL_DVS_PACKAGE = $(ZIP) -rg built/CUDA-thrust-package.zip thrust -9 -i *.inl
  APPEND_CUH_DVS_PACKAGE = $(ZIP) -rg built/CUDA-thrust-package.zip thrust -9 -i *.cuh
  MAKE_DVS_PACKAGE = $(CREATE_DVS_PACKAGE) && $(APPEND_HEADERS_DVS_PACKAGE) && $(APPEND_INL_DVS_PACKAGE) && $(APPEND_CUH_DVS_PACKAGE)
else
  CREATE_DVS_PACKAGE = tar -cv -f built/CUDA-thrust-package.tar bin thrust/internal/test thrust/internal/scripts thrust/internal/benchmark thrust/*.trs $(DVS_COMMON_TEST_PACKAGE_FILES)
  APPEND_HEADERS_DVS_PACKAGE = find thrust -name "*.h" | xargs tar rvf built/CUDA-thrust-package.tar
  APPEND_INL_DVS_PACKAGE = find thrust -name "*.inl" | xargs tar rvf built/CUDA-thrust-package.tar
  APPEND_CUH_DVS_PACKAGE = find thrust -name "*.cuh" | xargs tar rvf built/CUDA-thrust-package.tar
  COMPRESS_DVS_PACKAGE = bzip2 built/CUDA-thrust-package.tar
  MAKE_DVS_PACKAGE = $(CREATE_DVS_PACKAGE) && $(APPEND_HEADERS_DVS_PACKAGE) && $(APPEND_INL_DVS_PACKAGE) && $(APPEND_CUH_DVS_PACKAGE) && $(COMPRESS_DVS_PACKAGE)
endif

DVS_OPTIONS :=

ifneq ($(TARGET_ARCH),$(HOST_ARCH))
  DVS_OPTIONS += TARGET_ARCH=$(TARGET_ARCH)
endif
ifeq ($(TARGET_ARCH),ARMv7)
  DVS_OPTIONS += ABITYPE=$(ABITYPE)
endif

THRUST_DVS_BUILD = release

pack:
	cd .. && $(MAKE_DVS_PACKAGE)

dvs:
	$(MAKE) $(DVS_OPTIONS) -s -C ../cuda $(THRUST_DVS_BUILD)
	$(MAKE) $(DVS_OPTIONS) $(THRUST_DVS_BUILD) THRUST_DVS=1
	cd .. && $(MAKE_DVS_PACKAGE)

# XXX Deprecated, remove.
dvs_nightly: dvs

dvs_release:
	$(MAKE) dvs THRUST_DVS_BUILD=release

dvs_debug:
	$(MAKE) dvs THRUST_DVS_BUILD=debug

include $(THRUST_MKDIR)/dependencies.mk

