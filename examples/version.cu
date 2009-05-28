#include <thrust/version.h>
#include <iostream>

int main(void)
{
    int major = THRUST_VERSION / 100000;
    int minor = (THRUST_VERSION / 100) % 1000;

    std::cout << "Thrust v" << major << "." << minor << std::endl;

    return 0;
}

