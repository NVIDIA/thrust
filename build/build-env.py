import os
import inspect

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
    'gcc' : {'optimization' : '-O3', 'debug' : '-g',  'exception_handling' : ''},
    'g++' : {'optimization' : '-O3', 'debug' : '-g',  'exception_handling' : ''},
    'cl'  : {'optimization' : '/Ox', 'debug' : '/Zi', 'exception_handling' : '/EHsc'}
  }

# this dictionary maps the name of a linker program to a dictionary mapping the name of
# a linker switch of interest to the specific switch implementing the feature
gLinkerOptions = {
    'gcc' : {'debug' : ''},
    'g++' : {'debug' : ''},
    'link'  : {'debug' : '/debug'}
  }

def getCFLAGS(mode, CC):
  result = []
  if mode == 'release' or mode == 'emurelease':
    # turn on optimization
    result.append(gCompilerOptions[CC]['optimization'])
  elif mode == 'debug' or mode == 'emudebug':
    # turn on debug mode
    result.append(gCompilerOptions[CC]['debug'])
  return result

def getCXXFLAGS(mode, CXX):
  result = []
  if mode == 'release' or mode == 'emurelease':
    # turn on optimization
    result.append(gCompilerOptions[CXX]['optimization'])
  elif mode == 'debug' or mode == 'emudebug':
    # turn on debug mode
    result.append(gCompilerOptions[CXX]['debug'])
  # enable exception handling
  result.append(gCompilerOptions[CXX]['exception_handling'])
  return result

def getNVCCFLAGS(mode):
  result = []
  if mode == 'emurelease' or mode == 'emudebug':
    # turn on emulation
    result.append('-deviceemu')
  return result

# XXX this should actually be based on LINK,
#     but that's apparently a dynamic variable which
#     is harder to figure out
def getLINKFLAGS(mode, CXX):
  result = []
  if mode == 'debug':
    # turn on debug mode
    result = gLinkerOptions[CXX]['debug']
  return result

def Environment():
  env = OldEnvironment(tools = getTools())

  # scons has problems with finding the proper LIBPATH with Visual Studio Express 2008
  # help it out
  if os.name == 'nt':
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

  # get C compiler switches
  env.Append(CFLAGS = getCFLAGS(mode, env.subst('$CC')))

  # get CXX compiler switches
  env.Append(CXXFLAGS = getCXXFLAGS(mode, env.subst('$CXX')))

  # get NVCC compiler switches
  env.Append(NVCCFLAGS = getNVCCFLAGS(mode))

  # get linker switches
  env.Append(LINKFLAGS = getLINKFLAGS(mode, env.subst('$LINK')))

  # set CUDA lib & include path
  if os.name == 'posix':
    env.Append(LIBPATH = ['/usr/local/cuda/lib'])
    env.Append(CPPPATH = ['/usr/local/cuda/include'])
  elif os.name == 'nt':
    env.Append(LIBPATH = ['C:/CUDA/lib'])
    env.Append(CPPPATH = ['C:/CUDA/include'])
  else:
    raise ValueError, "Unknown OS. What are the CUDA include & library paths?"

  # add CUDA runtime library
  # XXX ideally this gets handled in nvcc.py if possible
  env.Append(LIBS = 'cudart')

  # set komrade include path
  env.Append(CPPPATH = os.path.dirname(thisDir))

  # import the LD_LIBRARY_PATH so we can run commands which depend
  # on shared libraries
  # XXX we should probably just copy the entire environment
  if os.name == 'posix':
    env['ENV']['LD_LIBRARY_PATH'] = os.environ['LD_LIBRARY_PATH']

  return env

