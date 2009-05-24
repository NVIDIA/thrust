#include <thrust/version.h>
#include <iostream>

int main(void)
{
    int major = KOMRADE_VERSION / 100000;
    int minor = (KOMRADE_VERSION / 100) % 1000;

    std::cout << "Komrade v" << major << "." << minor << std::endl;

    return 0;
}

