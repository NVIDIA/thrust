def product(*iterables):
    """compute the cartesian product of a list of iterables
    >>> for i in product(['a','b','c'],[1,2]):
    ...     print i
    ... 
    ['a', 1]
    ['a', 2]
    ['b', 1]
    ['b', 2]
    ['c', 1]
    ['c', 2]
    """

    if iterables:
        for head in iterables[0]:
            for remainder in product(*iterables[1:]):
                yield [head] + remainder
    else:
        yield []


####
# Function generators
def make_test_function_template(INITIALIZE, TIME, FINALIZE):
    import string
    import os

    function_template_file = os.path.join( os.path.split(__file__)[0], 'test_function_template.cxx')

    # test_function_template has locations for $PREAMBLE $INITIALIZE etc.
    test_template = string.Template(open(function_template_file).read())

    sections = {'INITIALIZE' : INITIALIZE,
                'TIME' : TIME,
                'FINALIZE' : FINALIZE}

    # skeleton has supplied definitions for $INCLUDE and $PREAMBLE
    # and has locations for $InputType and $InputSize etc.
    skeleton = test_template.safe_substitute(sections)
    
    return string.Template(skeleton)

def make_test_function(fname, TestVariablePairs, ftemplate):
    VariableDescription = '\n'.join(['RECORD_VARIABLE("%s","%s");' % pair for pair in TestVariablePairs])

    fmap = dict(TestVariablePairs)               
    fmap['DESCRIPTION'] = VariableDescription
    fmap['FUNCTION']    = fname
            
    return ftemplate.substitute(fmap)

def generate_functions(pname, TestVariables, INITIALIZE, TIME, FINALIZE):
    ftemplate = make_test_function_template(INITIALIZE, TIME, FINALIZE)

    TestVariableNames  = [ pair[0] for pair in TestVariables]
    TestVariableRanges = [ pair[1] for pair in TestVariables]

    for n,values in enumerate(product(*TestVariableRanges)):
        converted_values = []
        for v in values:
            v = str(v)
            v = v.replace(" ","_")  # C++ tokens we don't want
            v = v.replace(".","_")
            v = v.replace("<","_")
            v = v.replace(">","_")
            v = v.replace(",","_")
            v = v.replace(":","_")
            converted_values.append(v)

        fname = '_'.join( [pname] + converted_values )
        TestVariablePairs = zip(TestVariableNames, values)
        yield (fname, make_test_function(fname, TestVariablePairs, ftemplate))


####
# Program generators
def make_test_program(pname, functions, PREAMBLE = ""):
    parts = []
    parts.append("#include <build/perftest.h>")

    parts.append(PREAMBLE)

    for fname,fcode in functions:
        parts.append(fcode)

    #TODO output TestVariables in <testsuite> somewhere

    parts.append("int main(int argc, char **argv)")
    parts.append("{")
    parts.append("PROCESS_ARGUMENTS(argc, argv);")
    parts.append("BEGIN_TESTSUITE(\"" + pname + "\");")
    parts.append("RECORD_PLATFORM_INFO();")
    for fname,fcode in functions:
        parts.append(fname + "();")
    parts.append("END_TESTSUITE();")
    parts.append("}")
    parts.append("\n")

    return "\n".join(parts)

def generate_program(pname, TestVariables, PREAMBLE, INITIALIZE, TIME, FINALIZE):
    functions = list(generate_functions(pname, TestVariables, INITIALIZE, TIME, FINALIZE))
    return make_test_program(pname, functions, PREAMBLE)


###
# Test Input File -> Test Program
def process_test_file(filename):
    import os
    pname = os.path.splitext(os.path.split(filename)[1])[0]
    
    test_env_file = os.path.join( os.path.split(__file__)[0], 'test_env.py')

    # XXX why does execfile() not give us the right namespace?
    exec open(test_env_file)
    exec open(filename)

    return generate_program(pname, TestVariables, PREAMBLE, INITIALIZE, TIME, FINALIZE)


def compile_test(input_name, output_name):
    """Compiles a .test file into a .cu file"""
    open(output_name, 'w').write( process_test_file(input_name) )



##
# Simple Driver script
if __name__ == '__main__':
    import os, sys

    if len(sys.argv) not in [2,3]:
        print "usage: %s test_input.py [test_output.cu]" % (sys.argv[0],)
        os.exit()
    
    input_name = sys.argv[1]

    if len(sys.argv) == 2:
        # reduce.test -> reduce.cu
        output_name = os.path.splitext(os.path.split(filename)[1])[0] + '.cu'
    else:
        output_name = sys.argv[2]
        
    # process_test_file returns a string containing 
    # the whole test program (i.e. the text of a .cu file)
    compile_test(input_name, output_name)

    # this is just for show, scons integration would do this differently
    #import subprocess
    #subprocess.call('scons')
    #subprocess.call('./' + pname)
    #print "collecting data..."
    #output = subprocess.Popen(['./' + pname], stdout=subprocess.PIPE).communicate()[0]
    #print output


