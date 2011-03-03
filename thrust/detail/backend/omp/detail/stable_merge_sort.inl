/*
 *  Copyright 2008-2011 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

// don't attempt to #include this file without omp support
#if (THRUST_DEVICE_COMPILER_IS_OMP_CAPABLE == THRUST_TRUE)
#include <omp.h>
#endif // omp support

//#include <thrust/detail/host/sort.h>
#include <algorithm>

#include <thrust/iterator/detail/forced_iterator.h> // XXX remove this we we have a proper OMP sort
#include <thrust/iterator/iterator_traits.h>
#include <thrust/detail/device/omp/dispatch/sort.h>


namespace thrust
{
namespace detail
{
namespace device
{
namespace omp
{
namespace detail
{

template<typename RandomAccessIterator,
         typename StrictWeakOrdering>
void stable_merge_sort(RandomAccessIterator first,
                       RandomAccessIterator last,
                       StrictWeakOrdering comp)
{
    // we're attempting to launch an omp kernel, assert we're compiling with omp support
    // ========================================================================
    // X Note to the user: If you've found this line due to a compiler error, X
    // X you need to OpenMP support in your compiler.                         X
    // ========================================================================
    THRUST_STATIC_ASSERT( (depend_on_instantiation<RandomAccessIterator,
                          (THRUST_DEVICE_COMPILER_IS_OMP_CAPABLE == THRUST_TRUE)>::value) );

// do not attempt to compile the body of this function, which calls omp functions, without
// support from the compiler
// XXX implement the body of this function in another file to eliminate this ugliness
#if (THRUST_DEVICE_COMPILER_IS_OMP_CAPABLE == THRUST_TRUE)

    unsigned int P = omp_get_max_threads();

    // tiles processed by each processor
    unsigned int keycount  = last - first;
    unsigned int blocksize = (keycount + P - 1) / P;
    std::vector<unsigned int> begin(P), end(P);

    omp_set_num_threads(P);

    #pragma omp parallel
    {
        int p_i = omp_get_thread_num();

        begin[p_i] = blocksize * p_i;
        end[p_i]   = std::min(begin[p_i]+blocksize, keycount);

        //fprintf(stderr, "Range %d = [%d, %d)\n", p_i, begin[p_i], end[p_i]);

        // Every thread sorts its own tile
        std::stable_sort(thrust::raw_pointer_cast(&*first) + begin[p_i],
                         thrust::raw_pointer_cast(&*first) + end[p_i],
                         comp);

        #pragma omp barrier

        unsigned int nseg=P, h=2;

        // keep track of which sub-range we're processing
        unsigned int a=p_i, b=p_i, c=p_i+1;

        while( nseg>1 )
        {
            if( c>=P )  c=P-1;

            if( (p_i%h)==0 && c>b )
            {
                // Every thread sorts its own tile
                std::inplace_merge(thrust::raw_pointer_cast(&*first) + begin[a],
                                   thrust::raw_pointer_cast(&*first) + end[b],
                                   thrust::raw_pointer_cast(&*first) + end[c],
                                   comp);
                b  = c;
                c += h;
            }

            nseg = (nseg+1)/2;
            h *= 2;
            #pragma omp barrier
        }
    }
#endif // THRUST_DEVICE_COMPILER_IS_OMP_CAPABLE
}

} // end namespace detail
} // end namespace omp
} // end namespace device
} // end namespace detail
} // end namespace thrust

