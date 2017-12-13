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


ifndef PROFILE
ifdef VULCAN_TOOLKIT_BASE
include $(VULCAN_TOOLKIT_BASE)/build/getprofile.mk
include $(VULCAN_TOOLKIT_BASE)/build/config/$(PROFILE).mk
else
include ../build/getprofile.mk
include ../build/config/$(PROFILE).mk
endif
endif

SOLNDIR  := .

# Possible bug when compiling Thrust v.1.7.0 with VC8 so use at least VC9
#ifndef USEVC10
#export USEVC9=	1
#endif

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
# TODO: Refactor //sw/gpgpu/build and devise a solution in a form of
#       include mk file that defines BUILT_ROOTDIR
res:=$(shell $(PYTHON) generate_mk.py $(THRUST_MKDIR) $(THRUST_DIR))

## Generate makefiles
#

# Use these environment variables to control what gets built
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

ifneq ($(TEST_EXAMPLES_CUDA)$(TEST_EXAMPLES_THRUST),)
  override TEST_EXAMPLES=1
endif

ifeq ($(TEST_UNITTESTS)$(TEST_EXAMPLES)$(TEST_BENCH)$(TEST_OTHER),)
  override TEST_UNITTESTS := 1
  override TEST_EXAMPLES := 1
  override TEST_BENCH := 1
  override TEST_OTHER := 1
endif

filter_substr = $(foreach v,$2,$(if $(findstring $1,$v),$v))
filterout_substr =  $(foreach v,$2,$(if $(findstring $1,$v),,$v))


ifneq ($(TEST_UNITTESTS),)
  # copy existing projects
  PROJECTS_COPY := $(PROJECTS)
  # empty PROJECTS
  PROJECTS :=
  # populate PROJECTS with unit tests
  include $(THRUST_MKDIR)/testing.mk

  ifdef ERIS_TEST_LEVELS

    ERIS_PROJECTS :=
    # an empty list for L0
    ifneq ($(findstring L0,$(ERIS_TEST_LEVELS)),)
    endif

    # list of test for L1
    ifneq ($(findstring L1,$(ERIS_TEST_LEVELS)),)
      ERIS_PROJECTS += $(filter %testframework,$(PROJECTS))
      ERIS_PROJECTS += $(filter %thrust.test.adjacent_difference,$(PROJECTS))
      ERIS_PROJECTS += $(filter %thrust.test.cuda.merge_sort,$(PROJECTS))
      ERIS_PROJECTS += $(filter %thrust.test.cuda.pinned_allocator,$(PROJECTS))
      ERIS_PROJECTS += $(filter %thrust.test.cuda.radix_sort_by_key,$(PROJECTS))
      ERIS_PROJECTS += $(filter %thrust.test.cuda.radix_sort,$(PROJECTS))
      ERIS_PROJECTS += $(filter %thrust.test.cuda.reduce_intervals,$(PROJECTS))
      ERIS_PROJECTS += $(filter %thrust.test.binary_search,$(PROJECTS))
      ERIS_PROJECTS += $(filter %thrust.test.binary_search_descending,$(PROJECTS))
      ERIS_PROJECTS += $(filter %thrust.test.binary_search_vector,$(PROJECTS))
      ERIS_PROJECTS += $(filter %thrust.test.binary_search_vector_descending,$(PROJECTS))
      ERIS_PROJECTS += $(filter %thrust.test.copy,$(PROJECTS))
      ERIS_PROJECTS += $(filter %thrust.test.count,$(PROJECTS))
      ERIS_PROJECTS += $(filter %thrust.test.equal,$(PROJECTS))
      ERIS_PROJECTS += $(filter %thrust.test.fill,$(PROJECTS))
      ERIS_PROJECTS += $(filter %thrust.test.find,$(PROJECTS))
      ERIS_PROJECTS += $(filter %thrust.test.for_each,$(PROJECTS))
      ERIS_PROJECTS += $(filter %thrust.test.gather,$(PROJECTS))
      ERIS_PROJECTS += $(filter %thrust.test.generate,$(PROJECTS))
      ERIS_PROJECTS += $(filter %thrust.test.inner_product,$(PROJECTS))
      ERIS_PROJECTS += $(filter %thrust.test.is_partitioned,$(PROJECTS))
      ERIS_PROJECTS += $(filter %thrust.test.is_sorted,$(PROJECTS))
      ERIS_PROJECTS += $(filter %thrust.test.is_sorted_until,$(PROJECTS))
      ERIS_PROJECTS += $(filter %thrust.test.max_element,$(PROJECTS))
      ERIS_PROJECTS += $(filter %thrust.test.merge_by_key,$(PROJECTS))
      ERIS_PROJECTS += $(filter %thrust.test.merge,$(PROJECTS))
      ERIS_PROJECTS += $(filter %thrust.test.min_element,$(PROJECTS))
      ERIS_PROJECTS += $(filter %thrust.test.minmax_element,$(PROJECTS))
      ERIS_PROJECTS += $(filter %thrust.test.mismatch,$(PROJECTS))
      ERIS_PROJECTS += $(filter %thrust.test.partition,$(PROJECTS))
      ERIS_PROJECTS += $(filter %thrust.test.partition_point,$(PROJECTS))
      ERIS_PROJECTS += $(filter %thrust.test.permutation_iterator,$(PROJECTS))
      ERIS_PROJECTS += $(filter %thrust.test.reduce_by_key,$(PROJECTS))
      ERIS_PROJECTS += $(filter %thrust.test.reduce,$(PROJECTS))
      ERIS_PROJECTS += $(filter %thrust.test.remove,$(PROJECTS))
      ERIS_PROJECTS += $(filter %thrust.test.replace,$(PROJECTS))
      ERIS_PROJECTS += $(filter %thrust.test.reverse,$(PROJECTS))
      ERIS_PROJECTS += $(filter %thrust.test.reverse_iterator,$(PROJECTS))
      ERIS_PROJECTS += $(filter %thrust.test.scan_by_key,$(PROJECTS))
      ERIS_PROJECTS += $(filter %thrust.test.scan,$(PROJECTS))
      ERIS_PROJECTS += $(filter %thrust.test.scatter,$(PROJECTS))
      ERIS_PROJECTS += $(filter %thrust.test.sequence,$(PROJECTS))
      ERIS_PROJECTS += $(filter %thrust.test.set_difference_by_key,$(PROJECTS))
      ERIS_PROJECTS += $(filter %thrust.test.set_difference_by_key_descending,$(PROJECTS))
      ERIS_PROJECTS += $(filter %thrust.test.set_difference,$(PROJECTS))
      ERIS_PROJECTS += $(filter %thrust.test.set_difference_descending,$(PROJECTS))
      ERIS_PROJECTS += $(filter %thrust.test.set_intersection_by_key,$(PROJECTS))
      ERIS_PROJECTS += $(filter %thrust.test.set_intersection_by_key_descending,$(PROJECTS))
      ERIS_PROJECTS += $(filter %thrust.test.set_intersection,$(PROJECTS))
      ERIS_PROJECTS += $(filter %thrust.test.set_intersection_descending,$(PROJECTS))
      ERIS_PROJECTS += $(filter %thrust.test.set_symmetric_difference_by_key,$(PROJECTS))
      ERIS_PROJECTS += $(filter %thrust.test.set_symmetric_difference_by_key_descending,$(PROJECTS))
      ERIS_PROJECTS += $(filter %thrust.test.set_symmetric_difference,$(PROJECTS))
      ERIS_PROJECTS += $(filter %thrust.test.set_symmetric_difference_descending,$(PROJECTS))
      ERIS_PROJECTS += $(filter %thrust.test.set_union_by_key,$(PROJECTS))
      ERIS_PROJECTS += $(filter %thrust.test.set_union_by_key_descending,$(PROJECTS))
      ERIS_PROJECTS += $(filter %thrust.test.set_union,$(PROJECTS))
      ERIS_PROJECTS += $(filter %thrust.test.set_union_descending,$(PROJECTS))
      ERIS_PROJECTS += $(filter %thrust.test.sort_by_key,$(PROJECTS))
      ERIS_PROJECTS += $(filter %thrust.test.sort,$(PROJECTS))
      ERIS_PROJECTS += $(filter %thrust.test.stable_sort_by_key,$(PROJECTS))
      ERIS_PROJECTS += $(filter %thrust.test.stable_sort,$(PROJECTS))
      ERIS_PROJECTS += $(filter %thrust.test.swap_ranges,$(PROJECTS))
      ERIS_PROJECTS += $(filter %thrust.test.tabulate,$(PROJECTS))
      ERIS_PROJECTS += $(filter %thrust.test.transform,$(PROJECTS))
      ERIS_PROJECTS += $(filter %thrust.test.transform_reduce,$(PROJECTS))
      ERIS_PROJECTS += $(filter %thrust.test.transform_scan,$(PROJECTS))
      ERIS_PROJECTS += $(filter %thrust.test.uninitialized_copy,$(PROJECTS))
      ERIS_PROJECTS += $(filter %thrust.test.unique_by_key,$(PROJECTS))
      ERIS_PROJECTS += $(filter %thrust.test.unique,$(PROJECTS))
      ERIS_PROJECTS += $(filter %thrust.test.vector_insert,$(PROJECTS))
    endif
    
	# a full unit test suite for L2
    ifneq ($(findstring L2,$(ERIS_TEST_LEVELS)),)
      ERIS_PROJECTS := $(PROJECTS)
    endif

    PROJECTS := $(ERIS_PROJECTS)
     
  endif # ERIS_TEST_LEVELS

  ifdef THRUST_DVS
    ifndef THRUST_DVS_NIGHTLY
      PRJ := $(filter %testframework,$(PROJECTS))
      PRJ += $(filter %test.adjacent_difference,$(PROJECTS))
      PRJ += $(filter %test.cuda.arch,$(PROJECTS))
      PRJ += $(filter %test.cuda.radix_sort,$(PROJECTS))
      PRJ += $(filter %test.cuda.radix_sort_by_key,$(PROJECTS))
      PRJ += $(filter %test.binary_search_vector,$(PROJECTS))
      PRJ += $(filter %test.copy,$(PROJECTS))
      PRJ += $(filter %test.count,$(PROJECTS))
      PRJ += $(filter %test.fill,$(PROJECTS))
      PRJ += $(filter %test.for_each,$(PROJECTS))
      PRJ += $(filter %test.gather,$(PROJECTS))
      PRJ += $(filter %test.generate,$(PROJECTS))
      PRJ += $(filter %test.inner_product,$(PROJECTS))
      PRJ += $(filter %test.logical,$(PROJECTS))
      PRJ += $(filter %test.max_element,$(PROJECTS))
      PRJ += $(filter %test.merge,$(PROJECTS))
      PRJ += $(filter %test.merge_key_value,$(PROJECTS))
      PRJ += $(filter %test.min_element,$(PROJECTS))
      PRJ += $(filter %test.minmax_element,$(PROJECTS))
      PRJ += $(filter %test.partition,$(PROJECTS))
      PRJ += $(filter %test.partition_point,$(PROJECTS))
      PRJ += $(filter %test.reduce,$(PROJECTS))
      PRJ += $(filter %test.reduce_by_key,$(PROJECTS))
      PRJ += $(filter %test.remove,$(PROJECTS))
      PRJ += $(filter %test.replace,$(PROJECTS))
      PRJ += $(filter %test.reverse,$(PROJECTS))
      PRJ += $(filter %test.set_intersection,$(PROJECTS))
      PRJ += $(filter %test.set_symmetric_difference,$(PROJECTS))
      PRJ += $(filter %test.set_union,$(PROJECTS))
      PRJ += $(filter %test.transform,$(PROJECTS))
      PRJ += $(filter %test.transform_scan,$(PROJECTS))
      PRJ += $(filter %test.type_traits,$(PROJECTS))
      PRJ += $(filter %test.unique,$(PROJECTS))
      PRJ += $(filter %test.unique_by_key,$(PROJECTS))
      PRJ += $(filter %test.vector_cpp_subset,$(PROJECTS))
      PROJECTS := $(PRJ)
    endif
  endif  # THRUST_DVS

  # once PROJECTS is populated with unit tests extend it it with previous projects
  PROJECTS += $(PROJECTS_COPY)

  # Filter out tests that are known to fail to compile
  ifeq ($(TARGET_OS), QNX)
    PROJECTS := $(filter-out %thrust.test.complex_transform, $(PROJECTS))
  endif
endif

ifneq ($(TEST_OTHER),)
  PROJECTS += internal/build/warningstester
endif

ifneq ($(TEST_BENCH),)
  PROJECTS += internal/benchmark/bench
endif

ifneq ($(TEST_EXAMPLES),)
  PROJECTS_COPY := $(PROJECTS)
  PROJECTS :=
  include $(THRUST_MKDIR)/examples.mk

  EXAMPLES_CUDA   := $(call filter_substr,example.cuda,$(PROJECTS))
  EXAMPLES_THRUST := $(call filterout_substr,example.cuda,$(PROJECTS))

  ifneq ($(TEST_EXAMPLES_CUDA),)
    PROJECTS := $(PROJECTS_COPY) $(EXAMPLES_CUDA)
  else ifneq ($(TEST_EXAMPLES_THRUST),)
    PROJECTS := $(PROJECTS_COPY) $(EXAMPLES_THRUST)
  else
    PROJECTS := $(PROJECTS_COPY) $(EXAMPLES_CUDA) $(EXAMPLES_THRUST)
  endif

  # custom_temporary_allocation only works with gcc version 4.4 and higher
  ifneq ($(OS), win32)
    ifneq ($(shell expr "`$(CC) -dumpversion`" \< "4.4"), 0)
      PROJECTS := $(filter-out %example.cuda.custom_temporary_allocation, $(PROJECTS))
    endif
  endif

  # fallback_allocator TDRs on windows, thrust_nightly doesn't have a per-OS waive mechanism at the moment
  # so don't build it
  ifeq ($(OS), win32)
      PROJECTS := $(filter-out %example.cuda.fallback_allocator, $(PROJECTS))
  endif
endif

ifneq ($(OPENMP),)
  PROJECTS += internal/build/unittesterOMP
endif

ifdef ERIS_TEST_LEVELS
  PROJECTS += internal/build/eris_testsuites
endif

ifdef VULCAN_TOOLKIT_BASE
include $(VULCAN_TOOLKIT_BASE)/build/common.mk
else
include ../build/common.mk
endif

.PHONY: docs copy_doc
docs:
	$(MAKE) -f internal/doc/pdf.mk ROOTDIR=$(ROOTDIR) docs

copy_docs:
	$(MAKE) -f internal/doc/pdf.mk ROOTDIR=$(ROOTDIR) copy_docs

docs.clean:
	$(MAKE) -f internal/doc/pdf.mk ROOTDIR=$(ROOTDIR) clean

ifeq ($(OS), win32)
MAKE_DVS_PACKAGE = $(ZIP) -r built/CUDA-thrust-package.zip bin thrust/internal/test $(DVS_COMMON_TEST_PACKAGE_FILES)
else
MAKE_DVS_PACKAGE = tar -cvj -f built/CUDA-thrust-package.tar.bz2 bin thrust/internal/test $(DVS_COMMON_TEST_PACKAGE_FILES)
endif

DVS_OPTIONS :=

ifneq ($(TARGET_ARCH),$(HOST_ARCH))
  DVS_OPTIONS += TARGET_ARCH=$(TARGET_ARCH)
endif
ifeq ($(TARGET_ARCH),ARMv7)
  DVS_OPTIONS += ABITYPE=$(ABITYPE)
endif

THRUST_DVS_BUILD = release

dvs:
	$(MAKE) $(DVS_OPTIONS) -s -C ../cuda $(THRUST_DVS_BUILD)
	$(MAKE) $(DVS_OPTIONS) $(THRUST_DVS_BUILD) THRUST_DVS=1
	cd .. && $(MAKE_DVS_PACKAGE)

dvs_release:
	$(MAKE) dvs THRUST_DVS_BUILD=release

dvs_nightly dvs_nightly_release:
	$(MAKE) dvs_release THRUST_DVS_NIGHTLY=1

dvs_debug:
	$(MAKE) dvs THRUST_DVS_BUILD=debug

dvs_nightly_debug:
	$(MAKE) dvs_debug THRUST_DVS_NIGHTLY=1



include $(THRUST_MKDIR)/dependencies.mk

ifdef ERIS_TEST_LEVELS
DEPS := $(filter-out eris_testsuites,$(notdir $(PROJECTS)))
eris_testsuites: $(DEPS)
endif

