StandardTypes = ['char', 'unsigned char', 'short', 'unsigned short', 'int', 'unsigned int', 'long', 'unsigned long', 'float']
SignedIntegerTypes = ['char', 'short', 'int', 'long']
FloatingPointTypes = ['float','double']

StandardSizes = [2**k for k in range(4, 24)]

TestVariables = []

PREAMBLE = ""
INITIALIZE = ""
TIME = ""
FINALIZE = ""

