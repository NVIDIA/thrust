#include <komrade/host_vector.h>
#include <komrade/device_vector.h>
#include <komrade/copy.h>
#include <komrade/fill.h>
#include <komrade/range.h>

#include <iostream>

int main(void)
{
    // initialize all ten integers of a device_vector to 1
    komrade::device_vector<int> D(10, 1);

    // set the first seven elements of a vector to 9
    komrade::fill(D.begin(), D.begin() + 7, 9);

    // initialize a host_vector with the first five elements of D
    komrade::host_vector<int> H(D.begin(), D.begin() + 5);

    // set the elements of H to 0, 1, 2, 3, ...
    komrade::range(H.begin(), H.end());

    // copy all of H back to the beginning of D
    komrade::copy(H.begin(), H.end(), D.begin());

    // print D
    for(int i = 0; i < D.size(); i++)
        std::cout << "D[" << i << "] = " << D[i] << std::endl;

    return 0;
}
