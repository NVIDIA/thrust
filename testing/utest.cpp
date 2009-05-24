#include "thrusttest/testframework.h"
#include <cstdlib>


int main(int argc, char **argv)
{
    bool verbose = false;

    std::vector<std::string> tests;

    for(int i = 1; i < argc; i++){
        if(std::string(argv[i]) == "-v")
            verbose = true;
        else
            tests.push_back(argv[i]);
    }

  if (tests.empty()){
      if (!UnitTestDriver::s_driver().run_all_tests(verbose))
          exit(-1);
  }
  else {
      if (!UnitTestDriver::s_driver().run_tests(tests, verbose))
          exit(-1);
  }

  return 0;
}
