# Generic project mk that is included by examples mk
#  EXAMPLE_NAME : the name of the example
#  EXAMPLE_SRC  : path to the source code relative to thrust
#  EXAMPLE_EXT  : extension of the example source code, could be .cu  or .cpp
#  EXAMPLE_DIR  : path to source code relative to path where example mk is located
EXECUTABLE         := $(EXAMPLE_NAME)
BUILD_SRC          := $(ROOTDIR)/thrust/$(EXAMPLE_SRC)
BUILD_SRC_FLAGS    := $(EXAMPLE_FLAGS)

include $(ROOTDIR)/thrust/internal/build/common_build.mk
