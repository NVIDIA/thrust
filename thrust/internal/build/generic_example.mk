# Generic project mk that is included by examples mk
EXECUTABLE := $(EXAMPLE_NAME)
BUILD_SRC  := $(ROOTDIR)/thrust/$(EXAMPLE_SRC)

include $(ROOTDIR)/thrust/internal/build/common_detect.mk

EXAMPLE_MAKEFILE := $(join $(dir $(BUILD_SRC)), $(basename $(notdir $(BUILD_SRC))).mk)
ifneq ("$(wildcard $(EXAMPLE_MAKEFILE))","") # Check if the file exists.
  include $(EXAMPLE_MAKEFILE)
endif

include $(ROOTDIR)/thrust/internal/build/common_build.mk

