"""Exports a SCons construction environment 'env' with configuration common to all build projects"""
EnsureSConsVersion(1,2)

import os
import platform
import glob
import itertools


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


def tbb_installation(env):
  """Returns the details of TBB's installation
  returns (bin_path,lib_path,inc_path,library_name)
  """

  # determine defaults
  if os.name == 'nt':
    try:
      # we assume that TBBROOT exists in the environment
      root = env['ENV']['TBBROOT']

      # choose bitness
      bitness = 'ia32'
      if platform.machine()[-2:] == '64':
        bitness = 'intel64'

      # choose msvc version
      msvc_version = 'vc' + str(int(float(env['MSVC_VERSION'])))
      
      # assemble paths
      bin_path = os.path.join(root, 'bin', bitness, msvc_version)
      lib_path = os.path.join(root, 'lib', bitness, msvc_version)
      inc_path = os.path.join(root, 'include')
        
    except:
      raise ValueError, 'Where is TBB installed?'
  else:
    bin_path = ''
    lib_path = ''
    inc_path = ''

  return (bin_path,lib_path,inc_path,'tbb')


def inc_paths(env):
  """Returns a list of include paths needed by the compiler"""
  thrust_inc_path = Dir('.')
  cuda_inc_path = cuda_installation()[2]
  tbb_inc_path  = tbb_installation(env)[2]

  # note that the thrust path comes before the cuda path, which
  # may itself contain a different version of thrust
  return [thrust_inc_path, cuda_inc_path, tbb_inc_path]
  

def lib_paths(env):
  """Returns a list of lib paths needed by the linker"""
  cuda_lib_path = cuda_installation()[1]
  tbb_lib_path  = tbb_installation(env)[1]

  return [cuda_lib_path, tbb_lib_path]


def libs(env, CCX, host_backend, device_backend):
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

  if host_backend == 'tbb' or device_backend == 'tbb':
    result.append(tbb_installation(env)[3])

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
  result = []
  for machine_arch in arch:
    # transform arch_XX to compute_XX
    virtual_arch = machine_arch.replace('sm','compute')
    result.append('-gencode="arch={0},code={1}"'.format(virtual_arch, virtual_arch))
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
  
  # add a variable to handle the host backend
  vars.Add(ListVariable('host_backend', 'The host backend to target', 'cpp',
                        ['cpp', 'omp', 'tbb']))
  
  # add a variable to handle the device backend
  vars.Add(ListVariable('device_backend', 'The parallel device backend to target', 'cuda',
                        ['cuda', 'omp', 'tbb', 'cpp']))
  
  # add a variable to handle release/debug mode
  vars.Add(EnumVariable('mode', 'Release versus debug mode', 'release',
                        allowed_values = ('release', 'debug')))
  
  # add a variable to handle compute capability
  vars.Add(ListVariable('arch', 'Compute capability code generation', 'sm_10',
                        ['sm_10', 'sm_11', 'sm_12', 'sm_13', 'sm_20', 'sm_21']))
  
  # add a variable to handle warnings
  # only enable Wall by default on compilers other than cl
  vars.Add(BoolVariable('Wall', 'Enable all compilation warnings', os.name != 'nt'))
  
  # add a variable to treat warnings as errors
  vars.Add(BoolVariable('Werror', 'Treat warnings as errors', 0))

  return vars


# create a master Environment
vars = command_line_variables()

master_env = Environment(variables = vars, tools = ['default', 'nvcc', 'zip'])

# XXX it might be a better idea to harvest help text from subsidiary
#     SConscripts and only add their help text if one of their targets
#     is scheduled to be built
Help(vars.GenerateHelpText(master_env))

# enable RecursiveGlob
master_env.AddMethod(RecursiveGlob)

# add CUDA's lib dir to LD_LIBRARY_PATH so that we can execute commands
# which depend on shared libraries (e.g., cudart)
# we don't need to do this on windows
if master_env['PLATFORM'] == 'posix':
  master_env['ENV'].setdefault('LD_LIBRARY_PATH', []).append(cuda_installation()[1])
elif master_env['PLATFORM'] == 'darwin':
  master_env['ENV'].setdefault('DYLD_LIBRARY_PATH', []).append(cuda_installation()[1])
elif master_env['PLATFORM'] == 'win32':
  master_env['ENV']['TBBROOT'] = os.environ['TBBROOT']
  master_env['ENV']['PATH'] += ';' + tbb_installation(master_env)[0]

# get the list of requested backends
host_backends = master_env.subst('$host_backend').split()
device_backends = master_env.subst('$device_backend').split()

for (host,device) in itertools.product(host_backends, device_backends):
  # clone the master environment for this config
  env = master_env.Clone()

  # populate the environment
  env.Append(CPPPATH = inc_paths(env))
  
  env.Append(CCFLAGS = cc_compiler_flags(env.subst('$CXX'), env['mode'], host, device, env['Wall'], env['Werror']))
  
  env.Append(NVCCFLAGS = nv_compiler_flags(env['mode'], device, env['arch']))
  
  env.Append(LINKFLAGS = linker_flags(env.subst('$LINK'), env['mode'], env['PLATFORM'], device))
  
  env.Append(LIBPATH = lib_paths(env))
  
  env.Append(LIBS = libs(env, env.subst('$CXX'), host, device))
  
  # assemble the name of this configuration's targets directory
  targets_dir = 'targets/{0}_host_{1}_device_{2}'.format(host, device, env['mode'])

  # allow subsidiary SConscripts to peek at the backends
  env['host_backend'] = host
  env['device_backend'] = device
  
  # invoke each SConscript with a variant directory
  env.SConscript('examples/SConscript',    exports='env', variant_dir = 'examples/'    + targets_dir, duplicate = 0)
  env.SConscript('testing/SConscript',     exports='env', variant_dir = 'testing/'     + targets_dir, duplicate = 0)
  env.SConscript('performance/SConscript', exports='env', variant_dir = 'performance/' + targets_dir, duplicate = 0)

env = master_env
master_env.SConscript('SConscript', exports='env', variant_dir = 'targets', duplicate = False)

