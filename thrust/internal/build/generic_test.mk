# Generic project mk that is included by unit tests mk
EXECUTABLE := $(TEST_NAME)
BUILD_SRC  := $(ROOTDIR)/thrust/$(TEST_SRC)

ifdef VULCAN
  INCLUDES_ABSPATH += $(VULCAN_TOOLKIT_BASE)/thrust/testing
else
  INCLUDES_ABSPATH += $(ROOTDIR)/thrust/testing
endif

PROJ_LIBRARIES += testframework

THRUST_TEST := 1

include $(ROOTDIR)/thrust/internal/build/common_detect.mk

TEST_MAKEFILE := $(join $(dir $(BUILD_SRC)), $(basename $(notdir $(BUILD_SRC))).mk)
ifneq ("$(wildcard $(TEST_MAKEFILE))","") # Check if the file exists.
  include $(TEST_MAKEFILE)
endif

include $(ROOTDIR)/thrust/internal/build/common_build.mk

