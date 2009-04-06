#include <komradetest/unittest.h>
#include <komrade/device_vector.h>
#include <komrade/device_ptr.h>

void TestDevicePointerManipulation(void)
{
    typedef int T;

    komrade::device_vector<int> data(5);

    komrade::device_ptr<int> begin(&data[0]);
    komrade::device_ptr<int> end(&data[5]);

    ASSERT_EQUAL(end - begin, 5);

    begin++;
    begin--;
    
    ASSERT_EQUAL(end - begin, 5);

    begin += 1;
    begin -= 1;
    
    ASSERT_EQUAL(end - begin, 5);

    begin = begin + (int) 1;
    begin = begin - (int) 1;

    ASSERT_EQUAL(end - begin, 5);

    begin = begin + (unsigned int) 1;
    begin = begin - (unsigned int) 1;
    
    ASSERT_EQUAL(end - begin, 5);
    
    begin = begin + (size_t) 1;
    begin = begin - (size_t) 1;

    ASSERT_EQUAL(end - begin, 5);

    begin = begin + (ptrdiff_t) 1;
    begin = begin - (ptrdiff_t) 1;

    ASSERT_EQUAL(end - begin, 5);

    begin = begin + (komrade::device_ptr<int>::difference_type) 1;
    begin = begin - (komrade::device_ptr<int>::difference_type) 1;

    ASSERT_EQUAL(end - begin, 5);
}
DECLARE_UNITTEST(TestDevicePointerManipulation);

void TestMakeDevicePointer(void)
{
    typedef int T;

    T *raw_ptr = 0;

    komrade::device_ptr<T> p0 = komrade::device_pointer_cast(raw_ptr);

    ASSERT_EQUAL(komrade::raw_pointer_cast(p0), raw_ptr);

    komrade::device_ptr<T> p1 = komrade::device_pointer_cast(p0);

    ASSERT_EQUAL(p0, p1);
}
DECLARE_UNITTEST(TestMakeDevicePointer);

