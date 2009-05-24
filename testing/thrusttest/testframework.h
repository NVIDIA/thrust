#pragma once

#include <string>
#include <vector>
#include <iostream>

#include <stdio.h>

#include "meta.h"
#include "util.h"

// define some common lists of types
typedef thrusttest::type_list<int,
                               unsigned int,
                               float> ThirtyTwoBitTypes;

typedef thrusttest::type_list<long long,
                               unsigned long long,
                               double> SixtyFourBitTypes;

typedef thrusttest::type_list<char,
                               signed char,
                               unsigned char,
                               short,
                               unsigned short,
                               int,
                               unsigned int,
                               long,
                               unsigned long,
                               long long,
                               unsigned long long> IntegralTypes;

typedef thrusttest::type_list<signed char,
                               signed short,
                               signed int,
                               signed long,
                               signed long long> SignedIntegralTypes;

typedef thrusttest::type_list<unsigned char,
                               unsigned short,
                               unsigned int,
                               unsigned long,
                               unsigned long long> UnsignedIntegralTypes;

typedef thrusttest::type_list<char,
                               signed char,
                               unsigned char> ByteTypes;

typedef thrusttest::type_list<char,
                               signed char,
                               unsigned char,
                               short,
                               unsigned short> SmallIntegralTypes;

typedef thrusttest::type_list<long long,
                               unsigned long long> LargeIntegralTypes;

inline std::string base_class_name(const char *name)
{
  std::string result = name;

  // chop everything including and after first "<"
  return result.replace(result.find_first_of("<"),
                        result.size(),
                        "");
}

class UnitTest {
    public:
        std::string name;
        UnitTest() {}
        UnitTest(const char * name);
        virtual void run() {};
};

class UnitTestDriver {
  std::vector<UnitTest *> _test_list;

  bool run_tests(const std::vector<UnitTest *> &tests, const bool verbose);

public:
    
  void register_test(UnitTest *);
  bool run_all_tests(const bool verbose);
  bool run_tests(const std::vector<std::string> &tests, const bool verbose);

  static UnitTestDriver &s_driver();
};


// Macro to create a single unittest
#define DECLARE_UNITTEST(TEST)                                   \
class TEST##UnitTest : public UnitTest {                         \
    public:                                                      \
    TEST##UnitTest() : UnitTest(#TEST) {}                        \
    void run(){                                                  \
            TEST();                                              \
    }                                                            \
};                                                               \
TEST##UnitTest TEST##Instance

// Macro to create host and device versions of a
// unit test for a couple data types
#define DECLARE_VECTOR_UNITTEST(VTEST)                                                                            \
void VTEST##Host(void)   {  VTEST< thrust::host_vector<short> >();   VTEST< thrust::host_vector<int> >();   }    \
void VTEST##Device(void) {  VTEST< thrust::device_vector<short> >(); VTEST< thrust::device_vector<int> >(); }    \
DECLARE_UNITTEST(VTEST##Host);                                                                                    \
DECLARE_UNITTEST(VTEST##Device);

// Macro to create instances of a test for several 
// data types and array sizes
#define DECLARE_VARIABLE_UNITTEST(TEST)                          \
class TEST##UnitTest : public UnitTest {                         \
    public:                                                      \
    TEST##UnitTest() : UnitTest(#TEST) {}                        \
    void run(){                                                  \
        size_t sizes[] =   {0, 1, 2, 5, 13, 127,                 \
            251, 256, 259, 1023,                                 \
            1024, 1025, 10000, 12345,                            \
            (1<<16) - 3, 1<<16, (1<<16) + 3};                    \
        size_t num_sizes = sizeof(sizes) / sizeof(size_t);       \
        for(int i = 0; i != num_sizes; ++i)                      \
        {                                                        \
            TEST<char>(sizes[i]);                                \
            TEST<unsigned char>(sizes[i]);                       \
            TEST<short>(sizes[i]);                               \
            TEST<unsigned short>(sizes[i]);                      \
            TEST<int>(sizes[i]);                                 \
            TEST<unsigned int>(sizes[i]);                        \
            TEST<float>(sizes[i]);                               \
        }                                                        \
    }                                                            \
};                                                               \
TEST##UnitTest TEST##Instance

template<template <typename> class TestName, typename TypeList>
  class VariableUnitTest : public UnitTest
{
  public:
    VariableUnitTest()
      : UnitTest(base_class_name(thrusttest::type_name<TestName<int> >()).c_str()) {}

    void run()
    {
        size_t sizes[] =
           {0, 1, 2, 5, 13, 127,          
            251, 256, 259, 1023,                          
            1024, 1025, 10000, 12345,                     
            (1<<16) - 3, 1<<16, (1<<16) + 3};             
        size_t num_sizes = sizeof(sizes) / sizeof(size_t);
        for(int i = 0; i != num_sizes; ++i)               
        {                                                 
            // get the first type in the list
            typedef typename thrusttest::get_type<TypeList,0>::type first_type;

            thrusttest::for_each_type<TypeList,TestName,first_type,0> loop;

            // loop over the types
            loop(sizes[i]);
        }                                                 
    }
}; // end VariableUnitTest

template<template <typename> class TestName,
         typename TypeList,
         template <typename, typename> class Vector,
         template <typename> class Alloc>
  struct VectorUnitTest
    : public UnitTest
{
  VectorUnitTest()
    : UnitTest("dummy_name")
  {
    std::string test_name = base_class_name(thrusttest::type_name<TestName< Vector<int, Alloc<int> > > >());
    std::string vector_name = base_class_name(thrusttest::type_name<Vector<int, Alloc<int> > >());

    std::string my_name = test_name + "<" + vector_name + ">";

    UnitTest::name = my_name.c_str();
  }

  void run()
  {
    // zip up the type list with Alloc
    typedef typename thrusttest::transform1<TypeList, Alloc>::type AllocList;

    // zip up the type list & alloc list with Vector
    typedef typename thrusttest::transform2<TypeList, AllocList, Vector>::type VectorList;

    // get the first type in the list
    typedef typename thrusttest::get_type<VectorList,0>::type first_type;

    thrusttest::for_each_type<VectorList,TestName,first_type,0> loop;

    // loop over the types
    loop(0);
  }
}; // end VectorUnitTest

