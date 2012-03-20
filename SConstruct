"""Exports a SCons construction environment 'env' with configuration common to all build projects"""
EnsureSConsVersion(1,2)

import os
import platform
import glob


def RecursiveGlob(env, pattern, directory = Dir('.'), exclude = '\B'):
  """Recursively globs a directory and its children, returning a list of sources.
  Allows exclusion of directories given a regular expression.
  """
  directory = Dir(directory)

  result = directory.glob(pattern)

  for n in directory.glob('*'):
    # only recurse into directories which aren't in the blacklist
    import re
    if n.isdir() and not re.match(exclude, directory.rel_path(n)):
      result.extend(RecursiveGlob(env, pattern, n, exclude))
  return result


# map features to the list of compiler switches implementing them
gnu_compiler_flags = {
  'warn_all'           : ['-Wall'],
  'warnings_as_errors' : ['-Werror'],
  'release'            : ['-O2'],
  'debug'              : ['-g'],
  'exception_handling' : [],
  'cpp'                : [],
  'omp'                : ['-fopenmp'],
  'tbb'                : [],
  'cuda'               : [],
  'workarounds'        : []
}

msvc_compiler_flags = {
  'warn_all'           : ['/Wall'],
  'warnings_as_errors' : ['/WX'],
  'release'            : ['/Ox'],
  'debug'              : ['/Zi', '-D_DEBUG', '/MTd'],
  'exception_handling' : ['/EHsc'],
  'cpp'                : [],
  'omp'                : ['/openmp'],
  'tbb'                : [],
  'cuda'               : [],

  # avoid min/max problems due to windows.h
  # suppress warnings due to "decorated name length exceeded"
  'workarounds'        : ['/DNOMINMAX', '/wd4503']
}

compiler_to_flags = {
  'g++' : gnu_compiler_flags,
  'cl'  : msvc_compiler_flags
}

gnu_linker_flags = {
  'debug'       : [],
  'release'     : [],
  'workarounds' : []
}

msvc_linker_flags = {
  'debug'       : ['/debug'],
  'release'     : [],
  'workarounds' : []
}

linker_to_flags = {
  'gcc'  : gnu_linker_flags,
  'link' : msvc_linker_flags
}


def cuda_installation():
  """Returns the details of CUDA's installation
  returns (bin_path,lib_path,inc_path,library_name)
  """

  # determine defaults
  if os.name == 'nt':
    bin_path = 'C:/CUDA/bin'
    lib_path = 'C:/CUDA/lib'
    inc_path = 'C:/CUDA/include'
  elif os.name == 'posix':
    bin_path = '/usr/local/cuda/bin'
    lib_path = '/usr/local/cuda/lib'
    inc_path = '/usr/local/cuda/include'
  else:
    raise ValueError, 'Error: unknown OS.  Where is nvcc installed?'
   
  if platform.machine()[-2:] == '64':
    lib_path += '64'

  # override with environement variables
  if 'CUDA_BIN_PATH' in os.environ:
    bin_path = os.path.abspath(os.environ['CUDA_BIN_PATH'])
  if 'CUDA_LIB_PATH' in os.environ:
    lib_path = os.path.abspath(os.environ['CUDA_LIB_PATH'])
  if 'CUDA_INC_PATH' in os.environ:
    inc_path = os.path.abspath(os.environ['CUDA_INC_PATH'])

  return (bin_path,lib_path,inc_path,'cudart')


def omp_installation(CXX):
  """Returns the details of OpenMP's installation
  returns (bin_path,lib_path,inc_path,library_name)
  """

  bin_path = ''
  lib_path = ''
  inc_path = ''

  # the name of the library is compiler-dependent
  library_name = ''
  if CXX == 'g++':
    library_name = 'gomp'
  elif CXX == 'cl':
    library_name = 'VCOMP'
  else:
    raise ValueError, "Unknown compiler. What is the name of the OpenMP library?"

  return (bin_path,lib_path,inc_path,library_name)


def tbb_installation():
  """Returns the details of TBB's installation
  returns (bin_path,lib_path,inc_path,library_name)
  """

  # determine defaults
  if os.name == 'nt':
    raise ValueError, 'Where is TBB installed?'
  else:
    bin_path = ''
    lib_path = ''
    inc_path = ''

  return (bin_path,lib_path,inc_path,'tbb')


def inc_paths():
  """Returns a list of include paths needed by the compiler"""
  thrust_inc_path = Dir('.')
  cuda_inc_path = cuda_installation()[2]
  tbb_inc_path  = tbb_installation()[2]

  # note that the thrust path comes before the cuda path, which
  # may itself contain a different version of thrust
  return [thrust_inc_path, cuda_inc_path, tbb_inc_path]
  

def lib_paths():
  """Returns a list of lib paths needed by the linker"""
  cuda_lib_path = cuda_installation()[1]
  tbb_lib_path  = tbb_installation()[0]

  return [cuda_lib_path, tbb_lib_path]


def libs(CCX, host_backend, device_backend):
  """Returns a list of libraries to link against"""
  result = []

  # when compiling with g++, link against the standard library
  # we don't have to do this with cl
  if CCX == 'g++':
    result.append('stdc++')

  # link against backend-specific runtimes
  if host_backend == 'cuda' or device_backend == 'cuda':
    result.append(cuda_installation()[3])

  if host_backend == 'omp' or device_backend == 'omp':
    result.append(omp_installation(CCX)[3])

  if host_backend == 'tbb' or device_backend == 'omp':
    result.append(tbb_installation()[3])

  return result


def linker_flags(LINK, mode, platform, device_backend):
  """Returns a list of command line flags needed by the linker"""
  result = []

  flags = linker_to_flags[LINK]

  # debug/release
  result.extend(flags[mode])

  # unconditional workarounds
  result.extend(flags['workarounds'])

  # conditional workarounds

  # on darwin, we need to tell the linker to build 32b code for cuda
  if platform == 'darwin' and device_backend == 'cuda':
    result.append('-m32')

  return result

  
def macros(mode, host_backend, device_backend):
  """Returns a list of preprocessor macros needed by the compiler"""
  result = []

  # backend defines
  result.append('-DTHRUST_HOST_SYSTEM=THRUST_HOST_SYSTEM_' + host_backend.upper())
  result.append('-DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_' + device_backend.upper())

  if mode == 'debug':
    # turn on thrust debug mode
    result.append('-DTHRUST_DEBUG')

  return result


def cc_compiler_flags(CXX, mode, host_backend, device_backend, warn_all, warnings_as_errors):
  """Returns a list of command line flags needed by the c or c++ compiler"""
  # start with all platform-independent preprocessor macros
  result = macros(mode, host_backend, device_backend)

  flags = compiler_to_flags[CXX]

  # continue with unconditional flags

  # exception handling
  result.extend(flags['exception_handling'])

  # finish with conditional flags

  # debug/release
  result.extend(flags[mode])

  # enable host_backend code generation
  result.extend(flags[host_backend])

  # enable device_backend code generation
  result.extend(flags[device_backend])

  # Wall
  if warn_all:
    result.extend(flags['warn_all'])

  # Werror 
  if warnings_as_errors:
    result.extend(flags['warnings_as_errors'])

  # workarounds
  result.extend(flags['workarounds'])

  return result


def nv_compiler_flags(mode, device_backend, arch):
  """Returns a list of command line flags specific to nvcc"""
  result = ['-arch=' + arch]
  if mode == 'debug':
    # turn on debug mode
    # XXX make this work when we've debugged nvcc -G
    #result.append('-G')
    pass
  if device_backend != 'cuda':
    result.append("--x=c++")
  return result


def command_line_variables():
  # allow the user discretion to select the MSVC version
  vars = Variables()
  if os.name == 'nt':
    vars.Add(EnumVariable('MSVC_VERSION', 'MS Visual C++ version', None, allowed_values=('8.0', '9.0', '10.0')))
  
  # add a variable to handle the device backend
  vars.Add(EnumVariable('device_backend', 'The parallel device backend to target', 'cuda',
                        allowed_values = ('cuda', 'omp', 'tbb')))
  
  # add a variable to handle the host backend
  vars.Add(EnumVariable('host_backend', 'The host backend to target', 'cpp',
                        allowed_values = ('cpp', 'omp', 'tbb')))
  
  # add a variable to handle release/debug mode
  vars.Add(EnumVariable('mode', 'Release versus debug mode', 'release',
                        allowed_values = ('release', 'debug')))
  
  # add a variable to handle compute capability
  vars.Add(EnumVariable('arch', 'Compute capability code generation', 'sm_10',
                        allowed_values = ('sm_10', 'sm_11', 'sm_12', 'sm_20', 'sm_21')))
  
  # add a variable to handle warnings
  # only enable Wall by default on compilers other than cl
  vars.Add(BoolVariable('Wall', 'Enable all compilation warnings', os.name != 'nt'))
  
  # add a variable to treat warnings as errors
  vars.Add(BoolVariable('Werror', 'Treat warnings as errors', 0))

  return vars


# create an Environment
vars = command_line_variables()
env = Environment(variables = vars, tools = ['default', 'packaging'])
Help(vars.GenerateHelpText(env))

# enable nvcc
env.Tool('nvcc', toolpath = ['build'])

# enable RecursiveGlob
env.AddMethod(RecursiveGlob)

# import the LD_LIBRARY_PATH so we can run commands which
# depend on shared libraries (e.g., cudart)
# we don't need to do this on windows
if env['PLATFORM'] == 'posix':
  env['ENV']['LD_LIBRARY_PATH'] = os.environ['LD_LIBRARY_PATH']
elif env['PLATFORM'] == 'darwin':
  env['ENV']['DYLD_LIBRARY_PATH'] = os.environ['DYLD_LIBRARY_PATH']

# populate the environment
env.Append(CPPPATH = inc_paths())

env.Append(CCFLAGS = cc_compiler_flags(env.subst('$CXX'), env['mode'], env['host_backend'], env['device_backend'], env['Wall'], env['Werror']))

env.Append(NVCCFLAGS = nv_compiler_flags(env['mode'], env['device_backend'], env['arch']))

env.Append(LINKFLAGS = linker_flags(env.subst('$LINK'), env['mode'], env['PLATFORM'], env['device_backend']))

env.Append(LIBPATH = lib_paths())

env.Append(LIBS = libs(env.subst('$CXX'), env['host_backend'], env['device_backend']))

# make the build environment available to SConscripts
Export('env')

SConscript('SConscript')
SConscript('examples/SConscript')
SConscript('testing/SConscript')
SConscript('performance/SConscript')

