#include <unittest/unittest.h>
#include <string>
#include <thrust/detail/preprocessor.h>

void test_stringize()
{
    ASSERT_EQUAL(
        std::string(THRUST_PP_STRINGIZE(int))
      , "int"
    );

    ASSERT_EQUAL(
        std::string(THRUST_PP_STRINGIZE(hello world))
      , "hello world"
    );

    ASSERT_EQUAL(
        std::string(THRUST_PP_STRINGIZE(hello  world))
      , "hello world"
    );

    ASSERT_EQUAL(
        std::string(THRUST_PP_STRINGIZE( hello  world))
      , "hello world"
    );

    ASSERT_EQUAL(
        std::string(THRUST_PP_STRINGIZE(hello  world ))
      , "hello world"
    );

    ASSERT_EQUAL(
        std::string(THRUST_PP_STRINGIZE( hello  world ))
      , "hello world"
    );

    ASSERT_EQUAL(
        std::string(THRUST_PP_STRINGIZE(hello
                                        world))
      , "hello world"
    );

    ASSERT_EQUAL(
        std::string(THRUST_PP_STRINGIZE("hello world"))
      , "\"hello world\""
    );

    ASSERT_EQUAL(
        std::string(THRUST_PP_STRINGIZE('hello world'))
      , "'hello world'"
    );

    ASSERT_EQUAL(
        std::string(THRUST_PP_STRINGIZE($%!&<->))
      , "$%!&<->"
    );

    ASSERT_EQUAL(
        std::string(THRUST_PP_STRINGIZE($%!&""<->))
      , "$%!&\"\"<->"
    );

    ASSERT_EQUAL(
        std::string(THRUST_PP_STRINGIZE(THRUST_PP_STRINGIZE))
      , "THRUST_PP_STRINGIZE"
    );

    ASSERT_EQUAL(
        std::string(THRUST_PP_STRINGIZE(THRUST_PP_STRINGIZE(int)))
      , "\"int\""
    ); 
}
DECLARE_UNITTEST(test_stringize);

