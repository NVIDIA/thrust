import os
import inspect
import platform


def is_64bit():
  """ is this a 64-bit system? """
  return platform.machine()[-2:] == '64'
  #return platform.machine() == 'x86_64':
  #return platform.machine() == 'AMD64':


def getTools():
  result = []
  if os.name == 'nt':
    result = ['default', 'msvc']
  elif os.name == 'posix':
    result = ['default', 'gcc']
  else:
    result = ['default']
  return result;


OldEnvironment = Environment;


# this dictionary maps the name of a compiler program to a dictionary mapping the name of
# a compiler switch of interest to the specific switch implementing the feature
gCompilerOptions = {
    'gcc' : {'optimization' : '-O2', 'debug' : '-g',  'exception_handling' : '',      'omp' : '-fopenmp'},
    'g++' : {'optimization' : '-O2', 'debug' : '-g',  'exception_handling' : '',      'omp' : '-fopenmp'},
    'cl'  : {'optimization' : '/Ox', 'debug' : '/Zi', 'exception_handling' : '/EHsc', 'omp' : '/openmp'}
  }


# this dictionary maps the name of a linker program to a dictionary mapping the name of
# a linker switch of interest to the specific switch implementing the feature
gLinkerOptions = {
    'gcc' : {'debug' : ''},
    'g++' : {'debug' : ''},
    'link'  : {'debug' : '/debug'}
  }


def getBackendDefine(backend):
  result = ''
  if backend == 'cuda':
    result = '-DTHRUST_DEVICE_BACKEND=THRUST_DEVICE_BACKEND_CUDA'
  elif backend == 'omp':
    result = '-DTHRUST_DEVICE_BACKEND=THRUST_DEVICE_BACKEND_OMP'
  return result


def getCFLAGS(mode, backend, CC):
  result = []
  if mode == 'release' or mode == 'emurelease':
    # turn on optimization
    result.append(gCompilerOptions[CC]['optimization'])
  elif mode == 'debug' or mode == 'emudebug':
    # turn on debug mode
    result.append(gCompilerOptions[CC]['debug'])
  # force 32b code on darwin
  if platform.platform()[:6] == 'Darwin':
    result.append('-m32')

  # get the define for the device backend
  result.append(getBackendDefine(backend))

  # generate omp code
  if backend == 'omp':
    result.append(gCompilerOptions[CC]['omp'])

  return result


def getCXXFLAGS(mode, backend, CXX):
  result = []
  if mode == 'release' or mode == 'emurelease':
    # turn on optimization
    result.append(gCompilerOptions[CXX]['optimization'])
  elif mode == 'debug' or mode == 'emudebug':
    # turn on debug mode
    result.append(gCompilerOptions[CXX]['debug'])
  # enable exception handling
  result.append(gCompilerOptions[CXX]['exception_handling'])
  # force 32b code on darwin
  if platform.platform()[:6] == 'Darwin':
    result.append('-m32')

  # get the define for the device backend
  result.append(getBackendDefine(backend))

  return result


def getNVCCFLAGS(mode, backend):
  result = []
  if mode == 'emurelease' or mode == 'emudebug':
    # turn on emulation
    result.append('-deviceemu')

  # get the define for the device backend
  result.append(getBackendDefine(backend))

  return result


# XXX this should actually be based on LINK,
#     but that's apparently a dynamic variable which
#     is harder to figure out
def getLINKFLAGS(mode, backend, CXX):
  result = []
  if mode == 'debug':
    # turn on debug mode
    result.append(gLinkerOptions[CXX]['debug'])
  # force 32b code on darwin
  if platform.platform()[:6] == 'Darwin':
    result.append('-m32')

  # XXX make this portable
  if backend == 'ocelot':
    result.append(os.popen('OcelotConfig -l').read().split())

  return result


def Environment():
  # allow the user discretion to choose the MSVC version
  vars = Variables()
  if os.name == 'nt':
    vars.Add(EnumVariable('MSVC_VERSION', 'MS Visual C++ version', None, allowed_values=('8.0', '9.0', '10.0')))

  env = OldEnvironment(tools = getTools(), variables = vars)

  # scons has problems with finding the proper LIBPATH with Visual Studio Express 2008
  # help it out
  # XXX we might be able to ditch this WAR
  if os.name == 'nt':
    if is_64bit():
      env.Append(LIBPATH = ['C:/Program Files/Microsoft Visual Studio 8/VC/lib/amd64'])
    else:
      env.Append(LIBPATH = ['C:/Program Files/Microsoft SDKs/Windows/v6.0A/Lib'])
      env.Append(LIBPATH = ['C:/Program Files/Microsoft Visual Studio 9.0/VC/lib'])

  # get the absolute path to the directory containing
  # this source file
  thisFile = inspect.getabsfile(Environment)
  thisDir = os.path.dirname(thisFile)

  # enable nvcc
  env.Tool('nvcc', toolpath = [os.path.join(thisDir)])

  mode = 'release'
  if ARGUMENTS.get('mode'):
    mode = ARGUMENTS['mode']

  backend = 'cuda'
  if ARGUMENTS.get('backend'):
    backend = ARGUMENTS['backend']

  # get C compiler switches
  env.Append(CFLAGS = getCFLAGS(mode, backend, env.subst('$CC')))

  # get CXX compiler switches
  env.Append(CXXFLAGS = getCXXFLAGS(mode, backend, env.subst('$CXX')))

  # get NVCC compiler switches
  env.Append(NVCCFLAGS = getNVCCFLAGS(mode, backend))

  # get linker switches
  env.Append(LINKFLAGS = getLINKFLAGS(mode, backend, env.subst('$LINK')))
   
  # set CUDA lib & include path
  if is_64bit():
      lib_folder = 'lib64'
  else:
      lib_folder = 'lib'

  if os.name == 'posix':
    env.Append(LIBPATH = ['/usr/local/cuda/' + lib_folder])
    env.Append(CPPPATH = ['/usr/local/cuda/include'])
  elif os.name == 'nt':
    env.Append(LIBPATH = ['C:/CUDA/' + lib_folder])
    env.Append(CPPPATH = ['C:/CUDA/include'])
  else:
    raise ValueError, "Unknown OS. What are the CUDA include & library paths?"

  # set Ocelot lib path
  if backend == 'ocelot':
    if os.name == 'posix':
      env.Append(LIBPATH = ['/usr/local/lib'])
    else:
      raise ValueError, "Unknown OS.  What is the Ocelot library path?"

  # add CUDA runtime library
  # XXX ideally this gets handled in nvcc.py if possible
  env.Append(LIBS = 'cudart')

  # link against omp if necessary
  if backend == 'omp':
    if os.name == 'posix':
      env.Append(LIBS = ['gomp'])
    elif os.name == 'nt':
      env.Append(LIBS = ['VCOMP'])
    else:
      raise ValueError, "Unknown OS.  What is the name of the OpenMP library?"

  # set thrust include path
  env.Append(CPPPATH = os.path.dirname(thisDir))

  # import the LD_LIBRARY_PATH so we can run commands which depend
  # on shared libraries
  # XXX we should probably just copy the entire environment
  if os.name == 'posix':
    env['ENV']['LD_LIBRARY_PATH'] = os.environ['LD_LIBRARY_PATH']

  return env


