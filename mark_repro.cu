#include <thrust/iterator/transform_iterator.h>

int main()
{
    char str[100];

    auto comp = [=] (char v)
    {
        return (v == ' ') ? 0 : 1;
    };

    thrust::make_transform_iterator(str, comp);
}
