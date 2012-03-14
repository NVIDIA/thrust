EnsureSConsVersion(1,2)

import os
import platform


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


def omp_installation(CC):
  """Returns the details of OpenMP's installation
  returns (bin_path,lib_path,inc_path,library_name)
  """

  bin_path = ''
  lib_path = ''
  inc_path = ''

  # the name of the library is compiler-dependent
  library_name = ''
  if CC == 'gcc':
    library_name = 'gomp'
  elif CC == 'cl':
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
  thrust_inc_path = '.'
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


def libs(CC, CCX, host_backend, device_backend):
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
    result.append(omp_installation(CC)[3])

  if host_backend == 'tbb' or device_backend == 'omp':
    result.append(tbb_installation()[3])

  return result

  
def macros(mode, host_backend, device_backend):
  """Returns a list of preprocessor macros needed by the compiler"""
  result = []

  # backend defines
  result.append('-DTHRUST_HOST_SYSTEM=THRUST_HOST_SYSTEM_' + host_backend.upper())
  result.append('-DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_' + device_backend.upper())

  # turn on thrust debug mode
  result.append('-DTHRUST_DEBUG')

  return result


def command_line_variables():
  # allow the user discretion to select the MSVC version
  vars = Variables()
  if os.name == 'nt':
    vars.Add(EnumVariable('MSVC_VERSION', 'MS Visual C++ version', None, allowed_values=('8.0', '9.0', '10.0')))
  
  # add a variable to handle the device backend
  device_backend_variable = EnumVariable('device_backend', 'The parallel device backend to target', 'cuda',
                                         allowed_values = ('cuda', 'omp', 'tbb'))
  vars.Add(device_backend_variable)
  
  # add a variable to handle the host backend
  host_backend_variable = EnumVariable('host_backend', 'The host backend to target', 'cpp',
                                       allowed_values = ('cpp', 'omp', 'tbb'))
  vars.Add(host_backend_variable)
  
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
env = Environment(variables = vars)
Help(vars.GenerateHelpText(env))

# enable nvcc
env.Tool('nvcc', toolpath = ['build'])

# populate the environment
env.Append(CPPPATH = inc_paths())

env.Append(CXXFLAGS = macros(env['mode'], env['host_backend'], env['device_backend']))

env.Append(LIBPATH = lib_paths())

env.Append(LIBS = libs(env.subst('$CC'), env.subst('$CXX'), env['host_backend'], env['device_backend']))

# make the build environment available to SConscripts
Export('env')

