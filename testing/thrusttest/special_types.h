#pragma once

#include <iostream>
#include <utility>

namespace thrusttest
{

template <typename T1, typename T2>
class test_pair
{
    public:
    T1 first;
    T2 second;

    __host__ __device__
    bool operator==(const test_pair& tp) const {
        return (first == tp.first) && (second == tp.second);
    }
    
    __host__ __device__
    test_pair & operator=(const test_pair& tp)  {
        first = tp.first;
        second = tp.second;
        return *this;
    }

    __host__ __device__
    test_pair & operator=(const volatile test_pair& tp)  {
        first = tp.first;
        second = tp.second;
        return *this;
    }

    __host__ __device__
    volatile test_pair & operator=(const test_pair& tp) volatile  {
        first = tp.first;
        second = tp.second;
        return *this;
    }

    __host__ __device__
    volatile test_pair & operator=(const volatile test_pair& tp) volatile  {
        first = tp.first;
        second = tp.second;
        return *this;
    }

    __host__ __device__
    test_pair operator+(const test_pair& tp) const {
        test_pair temp;
        temp.first = first + tp.first;
        temp.second = second + tp.second;
        return temp;
    }

    // lexicographical comparison
    __host__ __device__
    bool operator<(const test_pair& tp) const {
        if (first < tp.first)
            return true;
        else if (tp.first < first)
            return false;
        else if (second < tp.second)
            return true;
        else
            return false;
    }

    template <typename X, typename Y>
    friend std::ostream& operator<<(std::ostream& out, const test_pair<X,Y> &tp);
};

template <typename T1, typename T2>
std::ostream& operator<<(std::ostream &o, const test_pair<T1, T2> &tp)
{
   return o << "(" << tp.first << "," << tp.second << ")";
}

} // end namespace thrusttest
