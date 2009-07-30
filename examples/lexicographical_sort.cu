#include <thrust/host_vector.h>
#include <thrust/generate.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <iostream>
#include <cstdlib>

// This example shows how to perform a lexicographical sort on multiple keys.
//
// http://en.wikipedia.org/wiki/Lexicographical_order

template <typename KeyVector, typename PermutationVector>
void update_permutation(KeyVector& keys, PermutationVector& permutation)
{
    // temporary storage for keys
    KeyVector temp(keys.size());

    // permute the keys with the current reordering
    thrust::gather(temp.begin(), temp.end(), permutation.begin(), keys.begin());

    // stable_sort the permuted keys and update the permutation
    thrust::stable_sort_by_key(temp.begin(), temp.end(), permutation.begin());
}


// random digit: integer in [0,10)
int rand10()
{
    return rand() % 10;
}


int main(void)
{
    size_t N = 20;

    // generate three arrays of random values
    thrust::host_vector<int> upper(N);
    thrust::host_vector<int> middle(N);
    thrust::host_vector<int> lower(N);
    thrust::generate(upper.begin(),  upper.end(),  rand10);
    thrust::generate(middle.begin(), middle.end(), rand10);
    thrust::generate(lower.begin(),  lower.end(),  rand10);
    
    std::cout << "Unsorted Keys" << std::endl;
    for(size_t i = 0; i < N; i++)
    {
        std::cout << "(" << upper[i] << "," << middle[i] << "," << lower[i] << ")" << std::endl;
    }

    // initialize permutation to [0, 1, 2, ... ,N-1]
    thrust::host_vector<int> permutation(N);
    thrust::sequence(permutation.begin(), permutation.end());

    // sort from least significant key to most significant keys
    update_permutation(lower,  permutation);
    update_permutation(middle, permutation);
    update_permutation(upper,  permutation);

    // Note: keys have not been modified
    // Note: permutation now maps unsorted keys to sorted order
    
    std::cout << "Sorted Keys" << std::endl;
    for(size_t i = 0; i < N; i++)
    {
        // p is the index in the *unsorted* arrays of the i-th sorted element
        int p = permutation[i];
        std::cout << "(" << upper[p] << "," << middle[p] << "," << lower[p] << ")" << std::endl;
    }

    return 0;
}
