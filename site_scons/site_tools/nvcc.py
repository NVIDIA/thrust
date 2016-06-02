"""SCons.Tool.nvcc

Tool-specific initialization for NVIDIA CUDA Compiler.

There normally shouldn't be any need to import this module directly.
It will usually be imported through the generic SCons.Tool.Tool()
selection method.

"""

import SCons.Tool
import SCons.Scanner.C
import SCons.Defaults
import os
import platform


def get_cuda_paths(env):
  """Determines CUDA {bin,lib,include} paths
  
  returns (bin_path,lib_path,inc_path)
  """

  cuda_path = env['cuda_path']

  bin_path = cuda_path + '/bin'
  lib_path = cuda_path + '/lib'
  inc_path = cuda_path + '/include'
   
  # fix up the name of the lib directory on 64b platforms
  if platform.machine()[-2:] == '64':
    if os.name == 'posix' and platform.system() != 'Darwin':
      lib_path += '64'
    elif os.name == 'nt':
      lib_path += '/x64'

  # override with environment variables
  if 'CUDA_BIN_PATH' in os.environ:
    bin_path = os.path.abspath(os.environ['CUDA_BIN_PATH'])
  if 'CUDA_LIB_PATH' in os.environ:
    lib_path = os.path.abspath(os.environ['CUDA_LIB_PATH'])
  if 'CUDA_INC_PATH' in os.environ:
    inc_path = os.path.abspath(os.environ['CUDA_INC_PATH'])

  return (bin_path,lib_path,inc_path)


CUDASuffixes = ['.cu']

# make a CUDAScanner for finding #includes
# cuda uses the c preprocessor, so we can use the CScanner
CUDAScanner = SCons.Scanner.C.CScanner()

def add_common_nvcc_variables(env):
  """
  Add underlying common "NVIDIA CUDA compiler" variables that
  are used by multiple builders.
  """

  # "NVCC common command line"
  if not env.has_key('_NVCCCOMCOM'):
    # nvcc needs '-I' prepended before each include path, regardless of platform
    env['_NVCC_CPPPATH'] = '${_concat("-I ", CPPPATH, "", __env__)}'

    # prepend -Xcompiler before each flag which needs it; some do not
    disallowed_flags = ['-std=c++03']

    need_no_prefix = ['-std=c++03', '-std=c++11']
    def flags_which_need_no_prefix(flags):
        # first filter out flags which nvcc doesn't allow
        flags = [flag for flag in flags if flag not in disallowed_flags]
        result = [flag for flag in flags if flag in need_no_prefix]
        return result

    def flags_which_need_prefix(flags):
        # first filter out flags which nvcc doesn't allow
        flags = [flag for flag in flags if flag not in disallowed_flags]
        result = [flag for flag in flags if flag not in need_no_prefix]
        return result

    env['_NVCC_BARE_FLAG_FILTER'] = flags_which_need_no_prefix
    env['_NVCC_PREFIXED_FLAG_FILTER'] = flags_which_need_prefix

    env['_NVCC_BARE_CFLAGS']       = '${_concat("",            CFLAGS, "", __env__, _NVCC_BARE_FLAG_FILTER)}'
    env['_NVCC_PREFIXED_CFLAGS']   = '${_concat("-Xcompiler ", CFLAGS, "", __env__, _NVCC_PREFIXED_FLAG_FILTER)}'
    env['_NVCC_CFLAGS']            = '$_NVCC_BARE_CFLAGS $_NVCC_PREFIXED_CFLAGS'

    env['_NVCC_BARE_SHCFLAGS']     = '${_concat("",            SHCFLAGS, "", __env__, _NVCC_BARE_FLAG_FILTER)}'
    env['_NVCC_PREFIXED_SHCFLAGS'] = '${_concat("-Xcompiler ", SHCFLAGS, "", __env__, _NVCC_PREFIXED_FLAG_FILTER)}'
    env['_NVCC_SHCFLAGS']          = '$_NVCC_BARE_SHCFLAGS $_NVCC_PREFIXED_SHCFLAGS'

    env['_NVCC_BARE_CCFLAGS']      = '${_concat("",            CCFLAGS, "", __env__, _NVCC_BARE_FLAG_FILTER)}'
    env['_NVCC_PREFIXED_CCFLAGS']  = '${_concat("-Xcompiler ", CCFLAGS, "", __env__, _NVCC_PREFIXED_FLAG_FILTER)}'
    env['_NVCC_CCFLAGS']           = '$_NVCC_BARE_CCFLAGS $_NVCC_PREFIXED_CCFLAGS'

    env['_NVCC_BARE_SHCCFLAGS']     = '${_concat("",            SHCCFLAGS, "", __env__, _NVCC_BARE_FLAG_FILTER)}'
    env['_NVCC_PREFIXED_SHCCFLAGS'] = '${_concat("-Xcompiler ", SHCCFLAGS, "", __env__, _NVCC_PREFIXED_FLAG_FILTER)}'
    env['_NVCC_SHCCFLAGS']          = '$_NVCC_BARE_SHCCFLAGS $_NVCC_PREFIXED_SHCCFLAGS'

    env['_NVCC_BARE_CPPFLAGS']      = '${_concat("",            CPPFLAGS, "", __env__, _NVCC_BARE_FLAG_FILTER)}'
    env['_NVCC_PREFIXED_CPPFLAGS']  = '${_concat("-Xcompiler ", CPPFLAGS, "", __env__, _NVCC_PREFIXED_FLAG_FILTER)}'
    env['_NVCC_CPPFLAGS']           = '$_NVCC_BARE_CPPFLAGS $_NVCC_PREFIXED_CPPFLAGS'

    # assemble the common command line
    env['_NVCCCOMCOM'] = '$_NVCC_CPPFLAGS $_CPPDEFFLAGS $_NVCC_CPPPATH'

def generate(env):
  """
  Add Builders and construction variables for CUDA compilers to an Environment.
  """

  # create a builder that makes PTX files from .cu files
  ptx_builder = SCons.Builder.Builder(action = '$NVCC -ptx $NVCCFLAGS $_NVCC_CFLAGS $_NVCC_CCFLAGS $_NVCCCOMCOM $SOURCES -o $TARGET',
                                      emitter = {},
                                      suffix = '.ptx',
                                      src_suffix = CUDASuffixes)
  env['BUILDERS']['PTXFile'] = ptx_builder

  # create builders that make static & shared objects from .cu files
  static_obj, shared_obj = SCons.Tool.createObjBuilders(env)

  for suffix in CUDASuffixes:
    # Add this suffix to the list of things buildable by Object
    static_obj.add_action('$CUDAFILESUFFIX', '$NVCCCOM')
    shared_obj.add_action('$CUDAFILESUFFIX', '$SHNVCCCOM')
    static_obj.add_emitter(suffix, SCons.Defaults.StaticObjectEmitter)
    shared_obj.add_emitter(suffix, SCons.Defaults.SharedObjectEmitter)

    # Add this suffix to the list of things scannable
    SCons.Tool.SourceFileScanner.add_scanner(suffix, CUDAScanner)

  add_common_nvcc_variables(env)

  # set the "CUDA Compiler Command" environment variable
  # windows is picky about getting the full filename of the executable
  if os.name == 'nt':
    env['NVCC'] = 'nvcc.exe'
    env['SHNVCC'] = 'nvcc.exe'
  else:
    env['NVCC'] = 'nvcc'
    env['SHNVCC'] = 'nvcc'
  
  # set the include path, and pass both c compiler flags and c++ compiler flags
  env['NVCCFLAGS'] = SCons.Util.CLVar('')
  env['SHNVCCFLAGS'] = SCons.Util.CLVar('') + ' -shared'
  
  # 'NVCC Command'
  env['NVCCCOM']   = '$NVCC -o $TARGET -c $NVCCFLAGS $_NVCC_CFLAGS $_NVCC_CCFLAGS $_NVCCCOMCOM $SOURCES'
  env['SHNVCCCOM'] = '$SHNVCC -o $TARGET -c $SHNVCCFLAGS $_NVCC_SHCFLAGS $_NVCC_SHCCFLAGS $_NVCCCOMCOM $SOURCES'
  
  # the suffix of CUDA source files is '.cu'
  env['CUDAFILESUFFIX'] = '.cu'

  # XXX add code to generate builders for other miscellaneous
  # CUDA files here, such as .gpu, etc.

  (bin_path,lib_path,inc_path) = get_cuda_paths(env)
    
  env.PrependENVPath('PATH', bin_path)

def exists(env):
  return env.Detect('nvcc')
