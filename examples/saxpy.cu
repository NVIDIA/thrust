#include <komrade/transform.h>
#include <komrade/device_vector.h>
#include <komrade/host_vector.h>
#include <komrade/functional.h>
#include <iostream>
#include <iterator>
#include <algorithm>

struct saxpy_functor
{
    const float a;

    saxpy_functor(float _a) : a(_a) {}

    __host__ __device__
        float operator()(const float& x, const float& y) const { 
            return a * x + y;
        }
};

void saxpy_fast(float A, komrade::device_vector<float>& X, komrade::device_vector<float>& Y)
{
    // Y <- A * X + Y
    komrade::transform(X.begin(), X.end(), Y.begin(), Y.begin(), saxpy_functor(A));
}

void saxpy_slow(float A, komrade::device_vector<float>& X, komrade::device_vector<float>& Y)
{
    komrade::device_vector<float> temp(X.size());
   
    // temp <- A
    komrade::fill(temp.begin(), temp.end(), A);
    
    // temp <- A * X
    komrade::transform(X.begin(), X.end(), temp.begin(), temp.begin(), komrade::multiplies<float>());

    // Y <- A * X + Y
    komrade::transform(temp.begin(), temp.end(), Y.begin(), Y.begin(), komrade::plus<float>());
}

int main(void)
{
    // initialize host arrays
    float x[4] = {1.0, 1.0, 1.0, 1.0};
    float y[4] = {1.0, 2.0, 3.0, 4.0};

    {
        // transfer to device
        komrade::device_vector<float> X(x, x + 4);
        komrade::device_vector<float> Y(y, y + 4);

        // slow method
        saxpy_slow(2.0, X, Y);
    }

    {
        // transfer to device
        komrade::device_vector<float> X(x, x + 4);
        komrade::device_vector<float> Y(y, y + 4);

        // fast method
        saxpy_fast(2.0, X, Y);
    }
    
    return 0;
}

