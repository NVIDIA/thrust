# Generic project mk that is included by unit tests mk
#  TEST_NAME : the name of the test
#  TEST_SRC  : path to the source code relative to thrust
#  TEST_EXT  : extension of the test source code, could be .cu  or .cpp
#  TEST_DIR  : path to source code relative to path where unit test mk is located
EXECUTABLE        := $(TEST_NAME)
BUILD_SRC         := $(ROOTDIR)/thrust/$(TEST_SRC)
BUILD_SRC_FLAGS   := $(TEST_FLAGS)

ifdef VULCAN
INCLUDES_ABSPATH += $(VULCAN_TOOLKIT_BASE)/thrust/testing
else
INCLUDES_ABSPATH += $(ROOTDIR)/thrust/testing
endif

PROJ_LIBRARIES += testframework

THRUST_TEST := 1
include $(ROOTDIR)/thrust/internal/build/common_build.mk
