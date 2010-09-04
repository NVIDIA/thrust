#pragma once

template <typename T, unsigned int N>
struct FixedVector
{
    T data[N];
    
    __host__ __device__
    FixedVector() : data() { }

    __host__ __device__
    FixedVector(T init)
    {
        for(unsigned int i = 0; i < N; i++)
            data[i] = init;
    }

    __host__ __device__
    FixedVector operator+(const FixedVector& bs) const
    {
        FixedVector output;
        for(unsigned int i = 0; i < N; i++)
            output.data[i] = data[i] + bs.data[i];
        return output;
    }
    
    __host__ __device__
    bool operator<(const FixedVector& bs) const
    {
        for(unsigned int i = 0; i < N; i++)
        {
            if(data[i] < bs.data[i])
                return true;
            else if(bs.data[i] < data[i])
                return false;
        }
        return false;
    }

    __host__ __device__
    bool operator==(const FixedVector& bs) const
    {
        for(unsigned int i = 0; i < N; i++)
        {
            if(!(data[i] == bs.data[i]))
                return false;
        }
        return true;                
    }
};


