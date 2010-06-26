# common definitions used by Python test generators

IntegerTypes       = ['char', 'short', 'int', 'long', 'long long']
FloatingPointTypes = ['float', 'double']

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

