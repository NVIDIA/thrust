# Copyright 2010-2020 NVIDIA Corporation.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#		http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Makefile for building Thrust unit test driver

# Force C++11 mode. NVCC will ignore it if the host compiler doesn't support it.
export CXX_STD := c++11

export CCCL_ENABLE_DEPRECATIONS := 1

export VERBOSE := 1

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

ifeq ($(OS), win32)
  CREATE_DVS_PACKAGE = $(ZIP) -r built/CUDA-thrust-package.zip bin thrust/internal/test thrust/internal/scripts thrust/internal/benchmark $(DVS_COMMON_TEST_PACKAGE_FILES)
  APPEND_H_DVS_PACKAGE = $(ZIP) -rg built/CUDA-thrust-package.zip thrust -9 -i *.h
  APPEND_INL_DVS_PACKAGE = $(ZIP) -rg built/CUDA-thrust-package.zip thrust -9 -i *.inl
  APPEND_CUH_DVS_PACKAGE = $(ZIP) -rg built/CUDA-thrust-package.zip thrust -9 -i *.cuh
  MAKE_DVS_PACKAGE = $(CREATE_DVS_PACKAGE) && $(APPEND_H_DVS_PACKAGE) && $(APPEND_INL_DVS_PACKAGE) && $(APPEND_CUH_DVS_PACKAGE)
else
  TAR_FILES = bin thrust/internal/test thrust/internal/scripts thrust/internal/benchmark $(DVS_COMMON_TEST_PACKAGE_FILES)
  TAR_FILES += `find -L thrust \( -name "*.cuh" -o -name "*.h" -o -name "*.inl" \)`
  MAKE_DVS_PACKAGE = tar -I bzip2 -chvf built/CUDA-thrust-package.tar.bz2 $(TAR_FILES)
endif

COPY_CUB_FOR_PACKAGING = rm -rf cub && cp -rp ../cub/cub cub

DVS_OPTIONS :=

ifneq ($(TARGET_ARCH),$(HOST_ARCH))
  DVS_OPTIONS += TARGET_ARCH=$(TARGET_ARCH)
endif
ifeq ($(TARGET_ARCH),ARMv7)
  DVS_OPTIONS += ABITYPE=$(ABITYPE)
endif

THRUST_DVS_BUILD = release

pack:
	$(COPY_CUB_FOR_PACKAGING)
	cd .. && $(MAKE_DVS_PACKAGE)

dvs:
	$(COPY_CUB_FOR_PACKAGING)
# Build the CUDA Runtime in GVS, because GVS has no CUDA Runtime component.
# This is a temporary workaround until the Tegra team adds a CUDA Runtime
# component, which they have promised to do.
ifdef GVS
	$(MAKE) $(DVS_OPTIONS) -s -C ../cuda $(THRUST_DVS_BUILD)
endif
	$(MAKE) $(DVS_OPTIONS) $(THRUST_DVS_BUILD) THRUST_DVS=1
	cd .. && $(MAKE_DVS_PACKAGE)

dvs_release:
	$(MAKE) dvs THRUST_DVS_BUILD=release

dvs_debug:
	$(MAKE) dvs THRUST_DVS_BUILD=debug

include $(THRUST_MKDIR)/dependencies.mk

