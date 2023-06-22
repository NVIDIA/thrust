ifeq ($(OS),$(filter $(OS),Linux Darwin))
  ifndef USEPGCXX
    CUDACC_FLAGS += -Xcompiler "-Wall -Wextra -Werror"

    ifdef USEXLC
      CXX_STD := c++14

      # GCC does not warn about unused parameters in uninstantiated
      # template functions, but xlC does. This causes xlC to choke on the
      # OMP backend, which is mostly #ifdef'd out when you aren't using it.
      CUDACC_FLAGS += -Xcompiler "-Wno-unused-parameter"

      # xlC is unreasonable about unused functions in a translation unit
      # when this warning is enabled; this includes warning on most functions
      # that are defined as static inline in cuda_fp16.h. Disable this warning
      # entirely under xlC.
      CUDACC_FLAGS += -Xcompiler "-Wno-unused-function"
    else # GCC, ICC or Clang AKA the sane ones.
      # XXX Enable -Wcast-align.
      CUDACC_FLAGS += -Xcompiler "-Winit-self -Woverloaded-virtual -Wno-cast-align -Wcast-qual -Wno-long-long -Wno-variadic-macros -Wno-unused-function"

      ifdef USE_CLANGLLVM
        IS_CLANG := 1
      endif

      ifeq ($(ABITYPE), androideabi)
        ifneq ($(findstring clang, $(BASE_COMPILER)),)
          IS_CLANG := 1
        endif
      endif

      ifeq ($(OS), Darwin)
        IS_CLANG := 1
      endif

      ifdef IS_CLANG
        CXX_STD := c++14

        ifdef USE_CLANGLLVM
          CLANG_VERSION = $(shell $(USE_CLANGLLVM) --version 2>/dev/null | head -1 | sed -e 's/.*\([0-9]\)\.\([0-9]\)\(\.[0-9]\).*/\1\2/g')
        else
          CLANG_VERSION = $(shell $(CCBIN) --version 2>/dev/null | head -1 | sed -e 's/.*\([0-9]\)\.\([0-9]\)\(\.[0-9]\).*/\1\2/g')
        endif

        # GCC does not warn about unused parameters in uninstantiated
        # template functions, but Clang does. This causes Clang to choke on the
        # OMP backend, which is mostly #ifdef'd out when you aren't using it.
        CUDACC_FLAGS += -Xcompiler "-Wno-unused-parameter"

        # -Wunneeded-internal-declaration misfires in the unit test framework
        # on older versions of Clang.
        CUDACC_FLAGS += -Xcompiler "-Wno-unneeded-internal-declaration"

        ifeq ($(shell if test $(CLANG_VERSION) -ge 60; then echo true; fi),true)
          # Clang complains about name mangling changes due to `noexcept`
          # becoming part of the type system; we don't care.
          CUDACC_FLAGS += -Xcompiler "-Wno-noexcept-type"
        endif
      else # GCC
        ifdef CCBIN
          CCBIN_ENVIRONMENT :=
          ifeq ($(OS), QNX)
            # QNX's GCC complains if QNX_HOST and QNX_TARGET aren't defined in the
            # environment.
            CCBIN_ENVIRONMENT := QNX_HOST=$(QNX_HOST) QNX_TARGET=$(QNX_TARGET)
          endif

          # Newer versions of GCC only print the major number with the
          # -dumpversion flag, but they print all three with -dumpfullversion.
          GCC_VERSION = $(shell $(CCBIN_ENVIRONMENT) $(CCBIN) -dumpfullversion 2>/dev/null | sed -e 's/\([0-9]\)\.\([0-9]\)\(\.[0-9]\)\?/\1\2/g')

          ifeq ($(GCC_VERSION),)
            # Older versions of GCC (~4.4 and older) seem to print three version
            # numbers (major, minor and patch) with the -dumpversion flag; newer
            # versions only print one or two numbers.
            GCC_VERSION = $(shell $(CCBIN_ENVIRONMENT) $(CCBIN) -dumpversion | sed -e 's/\([0-9]\)\.\([0-9]\)\(\.[0-9]\)\?/\1\2/g')
          endif

          ifeq ($(shell if test $(GCC_VERSION) -ge 50; then echo true; fi),true)
            CXX_STD := c++14
          else
            CUDACC_FLAGS += -DTHRUST_IGNORE_DEPRECATED_CPP_DIALECT
          endif

          ifeq ($(shell if test $(GCC_VERSION) -ge 73; then echo true; fi),true)
            # GCC 7.3 complains about name mangling changes due to `noexcept`
            # becoming part of the type system; we don't care.
            CUDACC_FLAGS += -Xcompiler "-Wno-noexcept-type"
          endif
          ifeq ($(shell if test $(GCC_VERSION) -ge 80; then echo true; fi),true)
            # GCC 8.x has a new warning that tries to diagnose technical misuses of
            # memcpy and memmove. We need to resolve it better than this, but for the
            # time being, we'll downgrade it from an error to a warning.
            CUDACC_FLAGS += -Xcompiler "-Wno-error=class-memaccess"
          endif
        else
          $(error CCBIN is not defined.)
        endif
      endif
    endif
  else
    CXX_STD := c++14
  endif
else ifeq ($(OS),win32)
  CXX_STD := c++14

  # XXX Enable /Wall
  CUDACC_FLAGS += -Xcompiler "/WX"

  # Disabled loss-of-data conversion warnings.
  # XXX Re-enable.
  CUDACC_FLAGS += -Xcompiler "/wd4244 /wd4267"

  # Suppress numeric conversion-to-bool warnings.
  # XXX Re-enable.
  CUDACC_FLAGS += -Xcompiler "/wd4800"

  # Disable warning about applying unary - to unsigned type.
  CUDACC_FLAGS += -Xcompiler "/wd4146"

  # Warning about declspec(allocator) on inappropriate function types
  CUDACC_FLAGS += -Xcompiler "/wd4494"

  # Allow tests to have lots and lots of sections in each translation unit:
  CUDACC_FLAGS += -Xcompiler "/bigobj"
endif

# Promote all NVCC warnings into errors
CUDACC_FLAGS += -Werror all-warnings

# Print warning numbers with cudafe diagnostics
CUDACC_FLAGS += -Xcudafe --display_error_number

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

