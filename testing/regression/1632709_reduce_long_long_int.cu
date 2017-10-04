#include <thrust/reduce.h> 
#include <thrust/iterator/constant_iterator.h> 
 
int main()
{ 
  long long int n = 10000000000ULL; 
  long long int s = 
  thrust::reduce(thrust::constant_iterator<long long int>(1LL),
                 thrust::constant_iterator<long long int>(1LL)+n); 
  std::cout << "long long: " << n << ' ' << s << std::endl; 
}
 
