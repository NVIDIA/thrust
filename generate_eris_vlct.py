#!/usr/bin/env python
# Generate a .vlct file for ERIS testing
# Usage: python generate_eris_vlct.py BINPATH  L{0,1,2}
#   The program globs executables and constructs a test_projects_L{0,1,2}.vlct file
#   The program is called from the Makefile once all the tests are built if ERIS_TEST_LEVELS is set
# NOTE: L{0,1,2} parameter in principle is not required, because the .vlct file is generated at the end of the building process.
#       Thus a single name for all test, such as eris_tests.vlct will suffice.
#       However, ERIS requires that .vlct files have unique names, ergo the L{0,1,2} suffix in the base name.
#
import sys, os, glob, re, platform

thrust_tests_vlct_template = """
{
  # Descriptive name for the testsuite (required).
  "name"      : "Thrust %(LEVEL)s Test suite",
  # Testsuite owner's email (required).
  "owner"     : "mrepasy@nvidia.com",
  # Define paths containing shared libraries required by the tests. Use envvar VULCAN_SHAREDLIB_DIR to refer 
  # to the platform specific portion of the path (e.g. bin/ for windows, lib64/ for 64-bit
  # Linux, etc.)
  "dllpath"   : [ "${VULCAN_INSTALL_DIR}/cuda/${INSTALL_TARGET_DIR}/${SHAREDLIB_DIR}",
                  "${VULCAN_INSTALL_DIR}/cuda/_internal/driver",
                  "${VULCAN_INSTALL_DIR}/PGI/17.1/linux86-64/17.1/lib"
                ],
  # Default working directory for test runs (optional). The directory can be a an absolute
  # or relative path. A relative path is relative to this file's location. Variables can
  # be used in the path using the ${var} syntax.
  "cwd"       : "${VULCAN_TESTSUITE_DIR}",
  # Timeout for entire testsuite, in seconds (optional). If not timeout is specified the
  # default timeout value of 900 seconds will be used.
  "timeout" : "%(TIMEOUT)s",
  # Default timeout for individual tests, in seconds (optional).
  "testtimeout" : "900",
  # The tests in the testsuite (required).
  "tests" : [
    %(THRUST_EXEC)s
  ]
}
"""

thrust_exec_template = """
    {
      "exe" : "%(test_exe)s",
      "attributes": [%(attributes)s]
      %(post)s
    }%(test_end)s
    """
thrust_exec_attributes = {
       'thrust.example.custom_temporary_allocation':
       """ 
         { "filter" : { "os" : "SLES11SP4, SLES11SP3, Mac" }},
         "result=skip",
         "comment=only works with gcc version 4.4 and higher on Linux & Mac"
       """,
       'thrust.example.fallback_allocator':
       """ 
         { "filter" : { "os" : "Windows" }},
         "result=skip",
         "comment=The fallback_allocator building from the makefile removed"
       """,
        }

thrust_skip_gold_verify = [
    "thrust.example.discrete_voronoi",
    "thrust.example.sorting_aos_vs_soa",
    "thrust.example.cuda.simple_cuda_streams",
    "thrust.example.cuda.fallback_allocator",
    ]


def Glob(pattern, directory,exclude='\b'):
    src = glob.glob(os.path.join(directory,pattern))
    p = re.compile(exclude)
    src = [s for s in src if not p.match(s)]
    return src

def build_vlct(name,binpath,use_post=True):
    system = platform.system();
    win32 = system == "Windows" or system[0:6] == "CYGWIN";
    if win32:
        execs=Glob(name+".exe", binpath)
    else:
        execs=Glob(name, binpath)

    exec_vlct = ""
    for e in execs:
        test_exe  = os.path.basename(e);
        test_name = os.path.splitext(test_exe)[0] if win32 else test_exe
        attributes = ""
        post = ""

        if test_name in thrust_exec_attributes:
          attributes = thrust_exec_attributes[test_name];
        if use_post and (not test_name in thrust_skip_gold_verify):
            post = ""","post": "${DIFF} STDOUT %s.gold" """ % test_name

        test_end = "" if e == execs[-1] else ","

        exec_vlct += thrust_exec_template % {
                "test_exe":test_exe,
                "post":post,
                "attributes":attributes,
                "test_end":test_end}
    return exec_vlct


binpath=sys.argv[1]
level=sys.argv[2]

if level == "L2":
    timeout = "12000"
elif level == "L1":
    timeout = "10200"
else:
    timeout = "3600"

THRUST_EXAMPLES = build_vlct("thrust.example.*",binpath);
THRUST_TESTS    = build_vlct("thrust.test.*",   binpath,use_post=False);

THRUST_EXEC = THRUST_EXAMPLES + THRUST_TESTS;

thrust_tests_vlct = thrust_tests_vlct_template % {"THRUST_EXEC":THRUST_EXEC,"LEVEL":level,"TIMEOUT":timeout}

#print thrust_tests_vlct

test_fn = "thrust_tests_%s.vlct" % level
f = open(os.path.join(binpath,test_fn),"w")
f.write(thrust_tests_vlct)
f.close()


