#include <thrust/version.h>
#include <iostream>

int main(void)
{
    int major = THRUST_MAJOR_VERSION;
    int minor = THRUST_MINOR_VERSION;
    int subminor = THRUST_SUBMINOR_VERSION;

    std::cout << "Thrust v" << major << "." << minor << "." << subminor << std::endl;

    return 0;
}

