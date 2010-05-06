#include "unittest/testframework.h"
#include "unittest/exceptions.h"

#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>
#include <algorithm>
#include <string>


void UnitTestDriver::register_test(UnitTest * test)
{
    if( UnitTestDriver::s_driver().test_map.count(test->name) )
        std::cout << "[WARNING] Test name \"" << test->name << " already encountered " << std::endl;
    UnitTestDriver::s_driver().test_map[test->name] = test;
}

UnitTest::UnitTest(const char * _name) : name(_name)
{
  UnitTestDriver::s_driver().register_test(this);
}


void process_args(int argc, char ** argv,
                  ArgumentSet& args,
                  ArgumentMap& kwargs)

{
    for(int i = 1; i < argc; i++)
    {
        std::string arg(argv[i]);

        // look for --key or --key=value arguments 
        if (arg.substr(0,2) == "--")
        {   
            std::string::size_type n = arg.find('=',2);

            if (n == std::string::npos)
                kwargs[arg.substr(2)] = std::string();              // (key,"")
            else
                kwargs[arg.substr(2, n - 2)] = arg.substr(n + 1);   // (key,value)
        }
        else
        {
            args.insert(arg);
        }
    }
}

void usage(int argc, char** argv)
{
    std::cout << "Usage:\n";
    std::cout << "\t" << argv[0] << "\n";
    std::cout << "\t" << argv[0] << " TestName1 [TestName2 ...] \n";
    std::cout << "\t" << argv[0] << " PartialTestName1* [PartialTestName2* ...] \n";
    std::cout << "\t" << argv[0] << " --device=1\n";
    std::cout << "\t" << argv[0] << " --verbose or --concise\n";
    std::cout << "\t" << argv[0] << " --list\n";
    std::cout << "\t" << argv[0] << " --help\n";
}

void list_devices(void)
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0)
        std::cout << "There is no device supporting CUDA" << std::endl;

    int selected_device;
    cudaGetDevice(&selected_device);

    for (int dev = 0; dev < deviceCount; ++dev) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        if (dev == 0)
        {
            if (deviceProp.major == 9999 && deviceProp.minor == 9999)
                std::cout << "There is no device supporting CUDA." << std::endl;
            else if (deviceCount == 1)
                std::cout << "There is 1 device supporting CUDA" << std:: endl;
            else
                std::cout << "There are " << deviceCount <<  " devices supporting CUDA" << std:: endl;
        }

        std::cout << "\nDevice " << dev << ": \"" << deviceProp.name << "\"";
        if(dev == selected_device)
            std::cout << "  [SELECTED]";
        std::cout << std::endl;

        std::cout << "  Major revision number:                         " << deviceProp.major << std::endl;
        std::cout << "  Minor revision number:                         " << deviceProp.minor << std::endl;
        std::cout << "  Total amount of global memory:                 " << deviceProp.totalGlobalMem << " bytes" << std::endl;
    }
    std::cout << std::endl;
}


struct TestResult
{
    TestStatus  status;
    std::string name;
    std::string message;
    
    TestResult(const TestStatus status, const UnitTest& u)
        : status(status), name(u.name)
    { }

    TestResult(const TestStatus status, const UnitTest& u, const unittest::UnitTestException& e)
        : status(status), name(u.name), message(e.message)
    { }

    bool operator<(const TestResult& tr) const
    {
        if (status < tr.status)
            return true;
        else if (tr.status < status)
            return false;
        else
            return name < tr.name;
    }
};

void record_result(const TestResult& test_result, std::vector< TestResult >& test_results)
{
    test_results.push_back(test_result);
}

void report_results(std::vector< TestResult >& test_results)
{
    std::cout << std::endl;

    std::string hline = "================================================================";
  
    std::sort(test_results.begin(), test_results.end());

    size_t num_failures = 0;
    size_t num_known_failures = 0;
    size_t num_errors = 0;

    for(size_t i = 0; i < test_results.size(); i++)
    {
        const TestResult& tr = test_results[i];

        if (tr.status != Pass)
        {
            std::cout << hline << std::endl;
        
            switch(tr.status)
            {
                case Failure:
                    std::cout << "FAILURE";       num_failures++;       break;
                case KnownFailure:
                    std::cout << "KNOWN FAILURE"; num_known_failures++; break;
                case Error:
                    std::cout << "ERROR";         num_errors++;         break;
                default:
                    break;
            }

            std::cout << ": " << tr.name << std::endl << tr.message << std::endl;
        }
    }

    std::cout << hline << std::endl;

    std::cout << "Totals: ";
    std::cout << num_failures << " failures, ";
    std::cout << num_known_failures << " known failures and ";
    std::cout << num_errors << " errors" << std::endl;
}


void UnitTestDriver::list_tests(void)
{
    for(TestMap::iterator iter = test_map.begin(); iter != test_map.end(); iter++)
        std::cout << iter->second->name << std::endl;
}


bool UnitTestDriver::run_tests(std::vector<UnitTest *>& tests_to_run, const ArgumentMap& kwargs)
{
    bool verbose = kwargs.count("verbose");
    bool concise = kwargs.count("concise");
    
    std::vector< TestResult > test_results;

    if (verbose && concise)
    {
        std::cout << "--verbose and --concise cannot be used together" << std::endl;
        exit(EXIT_FAILURE);
    }

    if (!concise)
        std::cout << "Running " << tests_to_run.size() << " unit tests." << std::endl;

    // Check error status before running any tests
    cudaError_t error = cudaGetLastError();
    if(error)
    {
        if (!concise)
        {
            std::cout << "[ERROR] CUDA Error detected before running tests: [";
            std::cout << std::string(cudaGetErrorString(error));
            std::cout << "]" << std::endl;
        }

        return false;
    } 


    for(size_t i = 0; i < tests_to_run.size(); i++){
        UnitTest& test = *tests_to_run[i];

        if (verbose)
            std::cout << "Running " << test.name << "..." << std::flush;

        try
        {
            // run the test
            test.run();

            // test passed
            record_result(TestResult(Pass, test), test_results);
        } 
        catch (unittest::UnitTestFailure& f)
        {
            record_result(TestResult(Failure, test, f), test_results);
        }
        catch (unittest::UnitTestKnownFailure& f)
        {
            record_result(TestResult(KnownFailure, test, f), test_results);
        }
        catch (unittest::UnitTestError& e)
        {
            record_result(TestResult(Error, test, e), test_results);
        }


        // immediate report
        if (!concise)
        {
            if (verbose)
            {
                switch(test_results.back().status)
                {
                    case Pass:
                        std::cout << "\r[PASS]             "; break;
                    case Failure:
                        std::cout << "\r[FAILURE]          "; break;
                    case KnownFailure:
                        std::cout << "\r[KNOWN FAILURE]    "; break;
                    case Error:
                        std::cout << "\r[ERROR]            "; break;
                    default:
                        break;
                }

                std::cout << " " << test.name << std::endl;
            }
            else
            {
                switch(test_results.back().status)
                {
                    case Pass:
                        std::cout << "."; break;
                    case Failure:
                        std::cout << "F"; break;
                    case KnownFailure:
                        std::cout << "K"; break;
                    case Error:
                        std::cout << "E"; break;
                    default:
                        break;
                }
            }
        }

        
        error = cudaGetLastError();
        if(error)
        {
            if (!concise)
            {
                std::cout << "\t[ERROR] CUDA Error detected after running " << test.name << ": [";
                std::cout << std::string(cudaGetErrorString(error));
                std::cout << "]" << std::endl;
            }
            return false;
        }

        std::cout.flush();
    }

    // summary report
    if (!concise)
        report_results(test_results);


    // if any failures or errors return false
    for(size_t i = 0; i < test_results.size(); i++)
        if (test_results[i].status != Pass && test_results[i].status != KnownFailure)
            return false;

    // all tests pass or are known failures
    return true;
}


bool UnitTestDriver::run_tests(const ArgumentSet& args, const ArgumentMap& kwargs)
{
    if (args.empty())
    {
        // run all tests
        std::vector<UnitTest *> tests_to_run;

        for(TestMap::iterator iter = test_map.begin(); iter != test_map.end(); iter++)
            tests_to_run.push_back(iter->second);

        return run_tests(tests_to_run, kwargs);
    }
    else
    {
        // all non-keyword arguments are assumed to be test names or partial test names

        typedef TestMap::iterator               TestMapIterator;

        // vector to accumulate tests
        std::vector<UnitTest *> tests_to_run;

        for(ArgumentSet::const_iterator iter = args.begin(); iter != args.end(); iter++)
        {
            const std::string& arg = *iter;

            size_t len = arg.size();
            size_t matches = 0;

            if (arg[len-1] == '*')
            {
                // wildcard search
                std::string search = arg.substr(0,len-1);

                TestMapIterator lb = test_map.lower_bound(search);
                while(lb != test_map.end())
                {
                    if (search != lb->first.substr(0,len-1))
                        break;

                    tests_to_run.push_back(lb->second); 
                    lb++;
                    matches++;
                }
            }
            else
            {
                // non-wildcard search
                TestMapIterator lb = test_map.find(arg);

                if (lb != test_map.end())
                {
                    tests_to_run.push_back(lb->second); 
                    matches++;
                }
            }


            if (matches == 0)
                std::cout << "[WARNING] found no test names matching the pattern: " << arg << std::endl;
        }

        return run_tests(tests_to_run, kwargs);
    }
}

UnitTestDriver &
UnitTestDriver::s_driver()
{
  static UnitTestDriver s_instance;
  return s_instance;
}

int main(int argc, char **argv)
{
    ArgumentSet args;
    ArgumentMap kwargs;

    process_args(argc, argv, args, kwargs);
    
    if(kwargs.count("help"))
    {
        usage(argc, argv);
        return 0;
    }

    if(kwargs.count("list"))
    {
        UnitTestDriver::s_driver().list_tests();
        return 0;
    }

    if(kwargs.count("device"))
    {
        int device_id  = kwargs.count("device") ? atoi(kwargs["device"].c_str()) :  0;
        cudaSetDevice(device_id);
    }
        
    if(kwargs.count("verbose"))
        list_devices();

    bool passed = UnitTestDriver::s_driver().run_tests(args, kwargs);

    if (kwargs.count("concise"))
        std::cout << ((passed) ? "PASSED" : "FAILED") << std::endl;
   
    return (passed) ? EXIT_SUCCESS : EXIT_FAILURE;
}
