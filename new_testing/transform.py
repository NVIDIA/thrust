# get some standard definitions
from build import *

# Test generators are functions starting with the name Test*

# We could support classes to, which would allow commonly related 
# test generators could subclass one another to further reduce
#  redundancy
# 
#   import bar
#   class TestFooSomeQuality (TestBarSomeQuality):
#   ...


# here we're using Python's builtin printf-style string 
# interpolation, but we could use a more powerful template library
# such as string.Template
def TestTransformUnarySemantics():
    for space in ['thrust::host_space_tag', 'thrust::device_space_tag']:
        yield "TestTransformUnarySemantics<%s>();" % (space,)

# test generators could also accept arguments from the
# main test driver (fast vs. slow, large vs. small, etc.)
# maybe a dictionary of options, alternatively they could
# inspect some singleton


def TestTransformUnaryGenerality():
    value_types = IntegerTypes
    containers  = ['std::list', 'std::vector', 'thrust::host_vector', 'thrust::device_vector']
    containers  = [('%s<%s>' % tuple(pair)) for pair in product(containers,value_types)]
    operators   = ['thrust::negate']

    # we could expand this to consider varying sizes, or make use of 
    # a general scheme to prepare the inputs and outputs
    template = """$container input1;
               $container output1;
               input1.push_back(1);
               input1.push_back(3);
               input1.push_back(5);
               input1.push_back(7);
               output1.push_back(0);
               output1.push_back(0);
               output1.push_back(0);
               output1.push_back(0);
               
               typedef $container::value_type input_type;

               TestTransformUnaryGenerality(input1.begin(), input1.begin(), output1.begin(), $operator<input_type>());
               """

    import string
    template = string.Template(template)

    # for illustration purposes, I don't think we'd actually want to consider
    # more than one operator or generate that many tests

    for (container,operator) in product(containers, operators):
        yield template.safe_substitute({'container' : container, 'operator' : operator})

# We could also support one-off, standalone tests
#def TestTransformSimple:
#    return """
#           thrust::vector<float> x(2);
#           thrust::vector<float> y(2);
#           thrust::vector<float> z(2);
#           x[0] = 1;
#           x[1] = 2;
#           z[0] = -1;
#           z[1] = -2;
#           thrust::transform(x.begin(), x.end(), y.begin(), thrust::negate<float>());
#           ASSERT_EQUAL(x, z);
#           """

