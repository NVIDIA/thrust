"""SCons.Tool.clang

Tool-specific initialization for Clang as CUDA Compiler.

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

  returns (cuda_path,bin_path,lib_path,inc_path)
  """

  cuda_path = env['cuda_path']

  # determine defaults
  if os.name == 'posix':
    bin_path = cuda_path + '/bin'
    lib_path = cuda_path + '/lib'
    inc_path = cuda_path + '/include'
  else:
    raise ValueError, 'Error: unknown OS.  Where is CUDA installed?'

  if platform.machine()[-2:] == '64':
    lib_path += '64'

  # override with environment variables
  if 'CUDA_BIN_PATH' in os.environ:
    bin_path = os.path.abspath(os.environ['CUDA_BIN_PATH'])
  if 'CUDA_LIB_PATH' in os.environ:
    lib_path = os.path.abspath(os.environ['CUDA_LIB_PATH'])
  if 'CUDA_INC_PATH' in os.environ:
    inc_path = os.path.abspath(os.environ['CUDA_INC_PATH'])

  return (cuda_path,bin_path,lib_path,inc_path)


CUDASuffixes = ['.cu']

# make a CUDAScanner for finding #includes
# cuda uses the c preprocessor, so we can use the CScanner
CUDAScanner = SCons.Scanner.C.CScanner()

def add_common_clang_variables(env):
  """
  Add underlying common clang variables that
  are used by multiple builders.
  """

  # "CLANG common command line"
  if not env.has_key('_CLANGCOMCOM'):
    # clang needs '-I' prepended before each include path, regardless of platform
    env['_CLANG_CPPPATH'] = '${_concat("-I ", CPPPATH, "", __env__)}'
    env['_CLANG_CFLAGS']       = '${_concat("",            CFLAGS, "", __env__)}'
    env['_CLANG_SHCFLAGS']     = '${_concat("",            SHCFLAGS, "", __env__)}'
    env['_CLANG_CCFLAGS']      = '${_concat("",            CCFLAGS, "", __env__)}'
    env['_CLANG_SHCCFLAGS']     = '${_concat("",            SHCCFLAGS, "", __env__)}'
    env['_CLANG_CPPFLAGS']      = '${_concat("",            CPPFLAGS, "", __env__)}'

    # assemble the common command line
    env['_CLANGCOMCOM'] = '$_CLANG_CPPFLAGS $_CPPDEFFLAGS $_CLANG_CPPPATH'

def generate(env):
  """
  Add Builders and construction variables for CUDA compilers to an Environment.
  """

  # create a builder that makes PTX files from .cu files
  ptx_builder = SCons.Builder.Builder(action = '$CLANG -S --cuda-path=$cuda_path --cuda-device-only $CLANGFLAGS $_CLANG_CFLAGS $_CLANG_CCFLAGS $_CLANGCOMCOM $SOURCES -o $TARGET',
                                      emitter = {},
                                      suffix = '.ptx',
                                      src_suffix = CUDASuffixes)
  env['BUILDERS']['PTXFile'] = ptx_builder

  # create builders that make static & shared objects from .cu files
  static_obj, shared_obj = SCons.Tool.createObjBuilders(env)

  for suffix in CUDASuffixes:
    # Add this suffix to the list of things buildable by Object
    static_obj.add_action('$CUDAFILESUFFIX', '$CLANGCOM')
    shared_obj.add_action('$CUDAFILESUFFIX', '$SHCLANGCOM')
    static_obj.add_emitter(suffix, SCons.Defaults.StaticObjectEmitter)
    shared_obj.add_emitter(suffix, SCons.Defaults.SharedObjectEmitter)

    # Add this suffix to the list of things scannable
    SCons.Tool.SourceFileScanner.add_scanner(suffix, CUDAScanner)

  add_common_clang_variables(env)

  (cuda_path, bin_path,lib_path,inc_path) = get_cuda_paths(env)

  # set the "CUDA Compiler Command" environment variable
  # windows is picky about getting the full filename of the executable
  env['CLANG'] = 'clang++'
  env['SHCLANG'] = 'clang++'

  # set the include path, and pass both c compiler flags and c++ compiler flags
  env['CLANGFLAGS'] = SCons.Util.CLVar('')
  env['SHCLANGFLAGS'] = SCons.Util.CLVar('') + ' -shared'

  # 'CLANG Command'
  env['CLANGCOM']   = '$CLANG -o $TARGET --cuda-path=$cuda_path -c $CLANGFLAGS $_CLANG_CFLAGS $_CLANG_CCFLAGS $_CLANGCOMCOM $SOURCES'
  env['SHCLANGCOM'] = '$SHCLANG -o $TARGET --cuda-path=$cuda_path -c $SHCLANGFLAGS $_CLANG_SHCFLAGS $_CLANG_SHCCFLAGS $_CLANGCOMCOM $SOURCES'

  # the suffix of CUDA source files is '.cu'
  env['CUDAFILESUFFIX'] = '.cu'

  env.PrependENVPath('PATH', bin_path)
  if 'CLANG_PATH' in os.environ:
    env.PrependENVPath('PATH', os.path.abspath(os.environ['CLANG_PATH']))

def exists(env):
  return env.Detect('clang++')
