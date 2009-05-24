#include "thrusttest/testframework.h"
#include "thrusttest/exceptions.h"

#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>

void UnitTestDriver::register_test(UnitTest *test)
{
    UnitTestDriver::s_driver()._test_list.push_back(test);
}

UnitTest::UnitTest(const char * _name) : name(_name)
{
  UnitTestDriver::s_driver().register_test(this);
}

bool UnitTestDriver::run_tests(const std::vector<UnitTest *> &tests_to_run, const bool verbose)
{
    bool any_failed = false;

    std::cout << "Running " << tests_to_run.size() << " unit tests." << std::endl;

    
    std::vector< std::pair<UnitTest *,thrusttest::UnitTestFailure> >      test_failures;
    std::vector< std::pair<UnitTest *,thrusttest::UnitTestKnownFailure> > test_known_failures;
    std::vector< std::pair<UnitTest *,thrusttest::UnitTestError>   >      test_errors;
    std::vector< UnitTest * >                                              test_exceptions;
    
    cudaError_t error = cudaGetLastError();
    if(error){
        std::cerr << "[ERROR] CUDA Error detected before running tests: [";
        std::cerr << std::string(cudaGetErrorString(error));
        std::cerr << "]" << std::endl;
        exit(EXIT_FAILURE);
    } 


    for(size_t i = 0; i < tests_to_run.size(); i++){
        UnitTest * test = tests_to_run[i];

        try {
            test->run();
            if (verbose)
                std::cout << "[PASS] " << test->name << std::endl;
            else
                std::cout << ".";
        } 
        catch (thrusttest::UnitTestFailure& f){
            any_failed = true;
            if (verbose)
                std::cout << "[FAILURE] " << test->name << std::endl;
            else
                std::cout << "F";
            test_failures.push_back(std::make_pair(test,f));
        }
        catch (thrusttest::UnitTestKnownFailure& f){
            if (verbose)
                std::cout << "[KNOWN FAILURE] " << test->name << std::endl;
            else
                std::cout << "K";
            test_known_failures.push_back(std::make_pair(test,f));
        } 
        catch (thrusttest::UnitTestError& e){
            any_failed = true;
            if (verbose)
                std::cout << "[ERROR] " << test->name << std::endl;
            else
                std::cout << "E";
            test_errors.push_back(std::make_pair(test,e));
        } 
        catch (...){
            any_failed = true;
            if (verbose)
                std::cout << "[UNKNOWN EXCEPTION] " << test->name << std::endl;
            else
                std::cout << "U";
            test_exceptions.push_back(test);
        }
        
        error = cudaGetLastError();
        if(error){
            std::cerr << "[ERROR] CUDA Error detected after running " << test->name << ": [";
            std::cerr << std::string(cudaGetErrorString(error));
            std::cerr << "]" << std::endl;
            exit(EXIT_FAILURE);
        }

        std::cout.flush();
    }
    std::cout << std::endl;

    std::string hline = "================================================================";

    for(size_t i = 0; i < test_failures.size(); i++){
        std::cout << hline << std::endl;
        std::cout << "FAILURE: " << test_failures[i].first->name << std::endl;
        std::cout << test_failures[i].second << std::endl;
    }
    for(size_t i = 0; i < test_known_failures.size(); i++){
        std::cout << hline << std::endl;
        std::cout << "KNOWN FAILURE: " << test_known_failures[i].first->name << std::endl;
        std::cout << test_known_failures[i].second << std::endl;
    }
    for(size_t i = 0; i < test_errors.size(); i++){
        std::cout << hline << std::endl;
        std::cout << "ERROR: " << test_errors[i].first->name << std::endl;
        std::cout << test_errors[i].second << std::endl;
    }
    for(size_t i = 0; i < test_exceptions.size(); i++){
        std::cout << hline << std::endl;
        std::cout << "UNKNOWN EXCEPTION: " << test_exceptions[i]->name << std::endl;
    }

    std::cout << hline << std::endl;

    std::cout << "Totals: ";
    std::cout << test_failures.size() << " failures, ";
    std::cout << test_known_failures.size() << " known failures, ";
    std::cout << test_errors.size() << " errors and ";
    std::cout << test_exceptions.size() << " unknown exceptions." << std::endl;

    return any_failed;
}

bool UnitTestDriver::run_all_tests(const bool verbose)
{
  return run_tests(_test_list, verbose);
}

bool 
UnitTestDriver::run_tests(const std::vector<std::string> &tests, const bool verbose)
{
  int i, j;
  std::vector<UnitTest *> tests_to_run;

  for (j=0; j < tests.size(); j++) {

    bool found = false;
    for (i = 0; !found && i < _test_list.size(); i++)
      if (tests[j] == _test_list[i]->name) {

        tests_to_run.push_back(_test_list[i]);
        found = true;
      }

    if (!found) {
      printf("[WARNING] UnitTestDriver::run_tests - test %s not found\n", tests[j].c_str());
    }
  }

  return run_tests(tests_to_run, verbose);
}

UnitTestDriver &
UnitTestDriver::s_driver()
{
  static UnitTestDriver s_instance;
  return s_instance;
}



