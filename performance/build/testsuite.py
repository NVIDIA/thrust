"""functions that generate reports and figures using the .xml output from the performance tests"""

__all__ = ['TestSuite', 'parse_testsuite_xml']

class TestSuite:
    def __init__(self, name, platform, tests):
        self.name = name
        self.platform = platform
        self.tests = tests

    def __repr__(self):
        import pprint
        return 'TestSuite' + pprint.pformat( (self.name, self.platform, self.tests) ) 

class Test:
    def __init__(self, name, variables, results):
        self.name = name
        self.variables = variables
        self.results = results

    def __repr__(self):
        return 'Test' + repr( (self.name, self.variables, self.results) )

def scalar_element(element):
    value = element.get('value')

    try:
        return int(value)
    except:
        try:
            return float(value)
        except:
            return value

def parse_testsuite_platform(et):
    testsuite_platform = {}

    platform_element = et.find('platform')
    device_element = platform_element.find('device')

    device = {}
    device['name'] = device_element.get('name')
    for property_element in device_element.findall('property'):
        device[property_element.get('name')] = scalar_element(property_element)

    testsuite_platform['device'] = device

    return testsuite_platform

def parse_testsuite_tests(et):
    testsuite_tests = {}

    for test_element in et.findall('test'):
        # test name
        test_name = test_element.get('name')

        # test variables: name -> value
        test_variables = {}
        for variable_element in test_element.findall('variable'):
            test_variables[variable_element.get('name')] = scalar_element(variable_element)

        # test results: name -> (value, units)
        test_results = {}
        for result_element in test_element.findall('result'):
            # TODO make this a thing that can be converted to its first element when treated like a number
            test_results[result_element.get('name')] = scalar_element(result_element)
        
        testsuite_tests[test_name] = Test(test_name, test_variables, test_results)

    return testsuite_tests

def parse_testsuite_xml(filename):
    import xml.etree.ElementTree as ET

    et = ET.parse(filename)
    
    testsuite_name = et.getroot().get('name')
    testsuite_platform = parse_testsuite_platform(et)
    testsuite_tests = parse_testsuite_tests(et)
    
    return TestSuite(testsuite_name, testsuite_platform, testsuite_tests)


