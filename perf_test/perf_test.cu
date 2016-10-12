#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/version.h>

#include <string>
#include <iostream>
#include <cassert>
#include <map>

#include "device_timer.h"
#include "random.h"
#include "demangle.hpp"

// Algos
#include "adjacent_difference.h"
#include "binary_search.h"
#include "copy.h"
#include "count.h"
#include "equal.h"
#include "extrema.h"
#include "fill.h"
#include "find.h"
#include "for_each.h"
#include "gather.h"
#include "generate.h"
#include "inner_product.h"
#include "logical.h"
#include "merge.h"
#include "mismatch.h"
#include "partition.h"
#include "reduce.h"
#include "remove.h"
#include "replace.h"
#include "reverse.h"
#include "scan.h"
#include "scatter.h"
#include "sequence.h"
#include "set_operations.h"
#include "set_operations_by_key.h"
#include "sort.h"
#include "swap.h"
#include "transform.h"
#include "transform_reduce.h"
#include "transform_scan.h"
#include "uninitialized_copy.h"
#include "uninitialized_fill.h"
#include "unique.h"

#if THRUST_VERSION >= 100700
#include "tabulate.h"
#endif

struct caching_device_allocator
{
  typedef char  value_type;
  typedef char *allocator_pointer;
  typedef std::multimap<std::ptrdiff_t, void *> free_blocks_type;
  typedef std::map<void *, std::ptrdiff_t>      allocated_blocks_type;

  free_blocks_type      free_blocks;
  allocated_blocks_type allocated_blocks;

  void free_all()
  {
    // deallocate all outstanding blocks in both lists
    for (free_blocks_type::iterator i = free_blocks.begin();
         i != free_blocks.end();
         ++i)
    {
      cudaError_t status = cudaFree(i->second);
      assert(cudaSuccess == status);
    }

    for (allocated_blocks_type::iterator i = allocated_blocks.begin();
         i != allocated_blocks.end();
         ++i)
    {
      cudaError_t status = cudaFree(i->first);
      assert(cudaSuccess == status);
    }
  }

  caching_device_allocator() {}

  ~caching_device_allocator()
  {
    // free all allocations when cached_allocator goes out of scope
    free_all();
  }

  char *allocate(std::ptrdiff_t num_bytes)
  {
    void *result = 0;

    // search the cache for a free block
    free_blocks_type::iterator free_block = free_blocks.find(num_bytes);

    if (free_block != free_blocks.end())
    {
      // get the pointer
      result = free_block->second;

      // erase from the free_blocks map
      free_blocks.erase(free_block);
    }
    else
    {
      // no allocation of the right size exists
      // create a new one with m_base_allocator
      // allocate memory and convert to raw pointer
      cudaError_t status = cudaMalloc(&result, num_bytes);
      assert(cudaSuccess == status);
    }

    // insert the allocated pointer into the allocated_blocks map
    allocated_blocks.insert(std::make_pair(result, num_bytes));

    return (char*)result;
  }

  void deallocate(char *ptr, size_t n)
  {
    // erase the allocated block from the allocated blocks map
    allocated_blocks_type::iterator iter      = allocated_blocks.find(ptr);
    std::ptrdiff_t                  num_bytes = iter->second;
    allocated_blocks.erase(iter);

    // insert the block into the free blocks map
    free_blocks.insert(std::make_pair(num_bytes, ptr));
  }
};


template<typename T>
std::string name_of_type()
{
  return std::string(demangle(typeid(T).name()));
}


template <typename Test>
void report(const Test& test, double time)
{
  std::string test_name = name_of_type<Test>();

  if (test_name.find("<") != std::string::npos)
  {
    test_name.resize(test_name.find("<"));
  }

  std::cout << test_name << ", " << time << ", " << std::endl;
}

__THRUST_DEFINE_HAS_MEMBER_FUNCTION(has_reset, reset);


template <typename Test>
typename thrust::detail::enable_if<
  has_reset<Test, void(void)>::value
>::type
  benchmark(Test& test, size_t iterations = 20)
{
  // run one iteration (warm up)
  for (int i = 0; i < 3; ++i)
  {
    test();

    test.reset();
  }
  
  thrust::host_vector<double> times(iterations);

  // the test has a reset function so we have to
  // be careful not to include the time it takes

  for (size_t i = 0; i < iterations; i++)
  {
    cudaDeviceSynchronize();
    device_timer timer;

    test();
    cudaDeviceSynchronize();
    
    times[i] = timer.elapsed_seconds();

    test.reset();
  }

  double mean = thrust::reduce(times.begin(), times.end()) / times.size();

  report(test, mean);
};


template <typename Test>
typename thrust::detail::disable_if<
  has_reset<Test, void(void)>::value
>::type
  benchmark(Test& test, size_t iterations = 20)
{
  // run one iteration (warm up)
  for (int i = 0; i < 3; ++i)
  {
    test();
  }

  // the test doesn't have a reset function so we can
  // just take the average time

  cudaDeviceSynchronize();
  device_timer timer;

  for (size_t i = 0; i < iterations; i++)
  {
    test();
  }
  cudaDeviceSynchronize();
    
  double time = timer.elapsed_seconds()/ iterations;

  report(test, time);
};

template <class Ty, class P>
void doit(P p, size_t N, size_t seed)
{
  typedef thrust::device_vector<Ty>       Vector;
  typedef thrust::host_vector<Ty>         hVector;
  typedef testing::random_integers<Ty>    RandomIntegers;
  typedef testing::random_integers<bool> RandomBooleans;


  RandomIntegers A_(N, 1235630645667);
  RandomIntegers B_(N, 234339572634);
  RandomIntegers C_(N, 345);
  RandomBooleans D(N, 456);
  Vector         T(N, 1);
  Vector         F(N, 0);
  Vector         S(N); thrust::sequence(S.begin(), S.end());
  Vector         U1(2*N, 0);
  Vector         U2(2*N, 0);


  hVector hA(N);
  hVector hB(N);
  hVector hC(N);

  srand48(seed);
  for (int i = 0; i < N; ++i)
  {
    hA[i] = drand48()*N;
    hB[i] = drand48()*N;
    hC[i] = drand48()*N;
  }
  
  Vector A = hA;
  Vector B = hB;
  Vector C = hC;


#ifndef _ALL
  { ComparisonSort<P,Vector>              temp(p,A);             benchmark(temp); }
  { ComparisonSortByKey<P,Vector>         temp(p,A,B);           benchmark(temp); }


#else

  thrust::identity<Ty> I;
  { AdjacentDifference<P,Vector>          temp(p,A,B);           benchmark(temp); } // adjacent_difference
  { LowerBound<P,Vector>                  temp(p,A,B,C);         benchmark(temp); } // binary_search
  { UpperBound<P,Vector>                  temp(p,A,B,C);         benchmark(temp); }
  { BinarySearch<P,Vector>                temp(p,A,B,C);         benchmark(temp); }
  { Copy<P,Vector>                        temp(p,A,B);           benchmark(temp); } // copy
  { CopyN<P,Vector>                       temp(p,A,B);           benchmark(temp); }
  { CopyIf<P,Vector>                      temp(p,A,D,B);         benchmark(temp); }
  { Count<P,Vector>                       temp(p,D);             benchmark(temp); } // count
  { CountIf<P,Vector>                     temp(p,D);             benchmark(temp); }
  { Equal<P,Vector>                       temp(p,A,A);           benchmark(temp); } // equal
  { MinElement<P,Vector>                  temp(p,A);             benchmark(temp); } // extrema
  { MaxElement<P,Vector>                  temp(p,A);             benchmark(temp); }
  { MinMaxElement<P,Vector>               temp(p,A);             benchmark(temp); }
  { Fill<P,Vector>                        temp(p,A);             benchmark(temp); } // fill
  { FillN<P,Vector>                       temp(p,A);             benchmark(temp); }
  { Find<P,Vector>                        temp(p,F,1);           benchmark(temp); } // find
  { FindIf<P,Vector>                      temp(p,F);             benchmark(temp); }
  { FindIfNot<P,Vector>                   temp(p,T);             benchmark(temp); }
  { ForEach<P,Vector>                     temp(p,A);             benchmark(temp); } // for_each
  { Gather<P,Vector>                      temp(p,S,A,B);         benchmark(temp); } // gather
  { GatherIf<P,Vector>                    temp(p,S,D,A,B);       benchmark(temp); }
  { Generate<P,Vector>                    temp(p,A);             benchmark(temp); } // generate
  { GenerateN<P,Vector>                   temp(p,A);             benchmark(temp); }
  { InnerProduct<P,Vector>                temp(p,A,B);           benchmark(temp); } // inner_product
  { AllOf<P,Vector>                       temp(p,T);             benchmark(temp); } // logical
  { AnyOf<P,Vector>                       temp(p,F);             benchmark(temp); }
  { NoneOf<P,Vector>                      temp(p,F);             benchmark(temp); }
  { Merge<P,Vector>                       temp(p,A,B,U1);        benchmark(temp); } // merge
  { Mismatch<P,Vector>                    temp(p,A,A);           benchmark(temp); } // mismatch
  { Partition<P,Vector>                   temp(p,A);             benchmark(temp); } // partition
  { PartitionCopy<P,Vector>               temp(p,D,A,B);         benchmark(temp); }
  { StablePartition<P,Vector>             temp(p,A);             benchmark(temp); }
  { StablePartitionCopy<P,Vector>         temp(p,D,A,B);         benchmark(temp); }
  { IsPartitioned<P,Vector>               temp(p,T);             benchmark(temp); }
  { PartitionPoint<P,Vector>              temp(p,T);             benchmark(temp); }
  { Reduce<P,Vector>                      temp(p,A);             benchmark(temp); } // reduce
  { ReduceByKey<P, Vector>                temp(p,D,A,B,C);       benchmark(temp); }
  { Remove<P,Vector>                      temp(p,D,0);           benchmark(temp); } // remove
  { RemoveCopy<P,Vector>                  temp(p,D,A,0);         benchmark(temp); }
  { RemoveIf<P,Vector>                    temp(p,A,D);           benchmark(temp); }
  { RemoveCopyIf<P,Vector>                temp(p,A,D,B);         benchmark(temp); }
  { Replace<P,Vector>                     temp(p,D,0,2);         benchmark(temp); } // replace
  { ReplaceCopy<P,Vector>                 temp(p,D,A,0,2);       benchmark(temp); }
  { ReplaceIf<P,Vector>                   temp(p,A,D,I,0);       benchmark(temp); }
  { ReplaceCopyIf<P,Vector>               temp(p,A,D,B,I,0);     benchmark(temp); }
  { Reverse<P,Vector>                     temp(p,A);             benchmark(temp); }
  { ReverseCopy<P,Vector>                 temp(p,A,B);           benchmark(temp); }
  { InclusiveScan<P,Vector>               temp(p,A,B);           benchmark(temp); }
  { ExclusiveScan<P,Vector>               temp(p,A,B);           benchmark(temp); }
  { InclusiveScanByKey<P,Vector>          temp(p,D,A,B);         benchmark(temp); }
  { ExclusiveScanByKey<P,Vector>          temp(p,D,A,B);         benchmark(temp); }
  { Scatter<P,Vector>                     temp(p,A,S,B);         benchmark(temp); } // scatter
  { ScatterIf<P,Vector>                   temp(p,A,S,D,B);       benchmark(temp); }
  { Sequence<P,Vector>                    temp(p,A);             benchmark(temp); } // sequence
  { SetDifference<P,Vector>               temp(p,A,B,U1);        benchmark(temp); } // set_operations
  { SetIntersection<P,Vector>             temp(p,A,B,U1);        benchmark(temp); }
  { SetSymmetricDifference<P,Vector>      temp(p,A,B,U1);        benchmark(temp); }
  { SetUnion<P,Vector>                    temp(p,A,B,U1);        benchmark(temp); }
  { Sort<P,Vector>                        temp(p,A);             benchmark(temp); } // sort
  { SortByKey<P,Vector>                   temp(p,A,B);           benchmark(temp); }
  { StableSort<P,Vector>                  temp(p,A);             benchmark(temp); }
  { StableSortByKey<P,Vector>             temp(p,A,B);           benchmark(temp); }
  { ComparisonSort<P,Vector>              temp(p,A);             benchmark(temp); }
  { ComparisonSortByKey<P,Vector>         temp(p,A,B);           benchmark(temp); }
  { IsSorted<P,Vector>                    temp(p,S);             benchmark(temp); }
  { IsSortedUntil<P,Vector>               temp(p,S);             benchmark(temp); }
  { SwapRanges<P,Vector>                  temp(p,A,B);           benchmark(temp); } // swap
  { UnaryTransform<P,Vector>              temp(p,A,B);           benchmark(temp); } // transform
  { BinaryTransform<P,Vector>             temp(p,A,B,C);         benchmark(temp); }
  { UnaryTransformIf<P,Vector>            temp(p,A,D,B);         benchmark(temp); }
  { BinaryTransformIf<P,Vector>           temp(p,A,B,D,C);       benchmark(temp); }
  { TransformReduce<P,Vector>             temp(p,A);             benchmark(temp); } // transform_reduce
  { TransformInclusiveScan<P,Vector>      temp(p,A,B);           benchmark(temp); } // transform_scan
  { TransformExclusiveScan<P,Vector>      temp(p,A,B);           benchmark(temp); }
  { UninitializedCopy<P,Vector>           temp(p,A,B);           benchmark(temp); } // uninitialized_copy
  { UninitializedFill<P,Vector>           temp(p,A);             benchmark(temp); } // fill
  { UninitializedFillN<P,Vector>          temp(p,A);             benchmark(temp); }
  { Unique<P,Vector>                      temp(p,D);             benchmark(temp); } // unique
  { UniqueCopy<P,Vector>                  temp(p,D,A);           benchmark(temp); }
  { UniqueByKey<P,Vector>                 temp(p,D,A);           benchmark(temp); }
  { UniqueByKeyCopy<P,Vector>             temp(p,D,A,B,C);       benchmark(temp); }
  { MergeByKey<P,Vector>                  temp(p,A,B,C,D,U1,U2); benchmark(temp); } // merge_by_key
  { SetDifferenceByKey<P,Vector>          temp(p,A,B,C,D,U1,U2); benchmark(temp); } // set_operations by_key
  { SetIntersectionByKey<P,Vector>        temp(p,A,B,C,U1,U2);   benchmark(temp); }
  { SetSymmetricDifferenceByKey<P,Vector> temp(p,A,B,C,D,U1,U2); benchmark(temp); }
  { SetUnionByKey<P,Vector>               temp(p,A,B,C,D,U1,U2); benchmark(temp); }
  { Tabulate<P,Vector>                    temp(p,A);             benchmark(temp); } // tabulate

#endif
  // host<->device copy

}


int main(int argc, char **argv)
{
  size_t N = 16 << 20;
  if(argc > 1)
  {
    N = atoi(argv[1]);
  } else if(argc > 2)
  {
    std::cerr << "usage: driver [datasize]" << std::endl;
    exit(-1);
  }


  std::cerr << "N= " << N << std::endl;

  size_t seed = (size_t)main;
  seed = 12345;

#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA_BULK
#define _CUDA cuda_bulk
#else
#define _CUDA cuda
#endif

#ifdef USE_CUDA_MALLOC
#define _PAR par
#else
  caching_device_allocator alloc;
#define _PAR par(alloc)
#endif

  {
    std::cout << "Ty = usigned int" << std::endl;
    std::cout << "-----------------" << std::endl;
    typedef unsigned int Ty;


    doit<Ty>(thrust::_CUDA::_PAR, N, seed);
  }
  {
    std::cout << std::endl;
    std::cout << "Ty = usigned long long" << std::endl;
    std::cout << "--------------------" << std::endl;
    typedef unsigned long long Ty;

    doit<Ty>(thrust::_CUDA::_PAR, N, seed);
  }


  return 0;
}

