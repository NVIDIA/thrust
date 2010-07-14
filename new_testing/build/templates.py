import string

StandardTypes      = ['int', 'float']
StandardContainers = ['std::list', 'std::vector', 'thrust::host_vector', 'thrust::device_vector']

def initialize_container(typename, varname, values):
    if isinstance(values, list):
        num_values = len(values)
        value_list = ', '.join([str(value) for value in values])
        template = string.Template("{ $typename temp[$num_values] = {$value_list}; $varname.assign(temp, temp + $num_values); }\n")
        return template.safe_substitute(locals())
    elif isinstance(values, int):
        return "%s.resize(%d,%s);\n" % (varname, values)
    else:
        raise TypeError('unknown value type')

    # could do (N,"random")
    #          (N,V)

def make_container(container, typename, varname, values):
    if isinstance(values, list):
        pass
        
    if container == "std::list":
        pass
    
StandardInputs = []    

# ordered integer sequences
StandardInputs.extend([range(N) for N in range(14)])

# reverse ordered sequences
StandardInputs.extend([range(N,-1,-1) for N in range(1,14)])

# random permutations 
# generated with 
# >>> import numpy; 
# >>> for n in range(3,14): print "StandardInputs.append(%s)" % (list(numpy.random.permutation(n)),)
StandardInputs.append([1, 0, 2])
StandardInputs.append([2, 0, 1, 3])
StandardInputs.append([1, 4, 0, 2, 3])
StandardInputs.append([4, 2, 3, 1, 5, 0])
StandardInputs.append([4, 5, 3, 6, 1, 2, 0])
StandardInputs.append([5, 3, 7, 2, 4, 0, 6, 1])
StandardInputs.append([3, 1, 5, 8, 7, 4, 0, 6, 2])
StandardInputs.append([4, 3, 8, 9, 6, 7, 0, 1, 2, 5])
StandardInputs.append([6, 10, 0, 1, 4, 5, 3, 2, 8, 7, 9])
StandardInputs.append([11, 10, 1, 0, 3, 8, 7, 5, 4, 9, 2, 6])
StandardInputs.append([10, 11, 5, 3, 12, 6, 1, 2, 4, 9, 0, 8, 7])


def unary_transformation(function, containers=StandardContainers, value_types=StandardTypes):
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

    template = string.Template(template)

    # for illustration purposes, I don't think we'd actually want to consider
    # more than one operator or generate that many tests

    for (container,operator) in product(containers, operators):
        yield template.safe_substitute({'container' : container, 'operator' : operator})


