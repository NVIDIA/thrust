#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/version.h>

#include <string>
#include <iostream>
#include <cassert>

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
  benchmark(Test& test, size_t iterations = 100)
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
  benchmark(Test& test, size_t iterations = 100)
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

  typedef thrust::device_vector<unsigned int>     Vector;
  typedef testing::random_integers<unsigned int>  RandomIntegers;
  typedef testing::random_integers<bool>          RandomBooleans;
  
  RandomIntegers A(N, 123);
  RandomIntegers B(N, 234);
  RandomIntegers C(N, 345);
  RandomBooleans D(N, 456);
  Vector         T(N, 1);
  Vector         F(N, 0);
  Vector         S(N); thrust::sequence(S.begin(), S.end());
  Vector         U1(2*N, 0);
  Vector         U2(2*N, 0);

  thrust::identity<unsigned int> I;

  { AdjacentDifference<Vector>          temp(A,B);           benchmark(temp); } // adjacent_difference
  { LowerBound<Vector>                  temp(A,B,C);         benchmark(temp); } // binary_search
  { UpperBound<Vector>                  temp(A,B,C);         benchmark(temp); }
  { BinarySearch<Vector>                temp(A,B,C);         benchmark(temp); }
  { Copy<Vector>                        temp(A,B);           benchmark(temp); } // copy
  { CopyN<Vector>                       temp(A,B);           benchmark(temp); }
  { CopyIf<Vector>                      temp(A,D,B);         benchmark(temp); }
  { Count<Vector>                       temp(D);             benchmark(temp); } // count
  { CountIf<Vector>                     temp(D);             benchmark(temp); }
  { Equal<Vector>                       temp(A,A);           benchmark(temp); } // equal
  { MinElement<Vector>                  temp(A);             benchmark(temp); } // extrema
  { MaxElement<Vector>                  temp(A);             benchmark(temp); }
  { MinMaxElement<Vector>               temp(A);             benchmark(temp); }
  { Fill<Vector>                        temp(A);             benchmark(temp); } // fill
  { FillN<Vector>                       temp(A);             benchmark(temp); }
  { Find<Vector>                        temp(F,1);           benchmark(temp); } // find
  { FindIf<Vector>                      temp(F);             benchmark(temp); }
  { FindIfNot<Vector>                   temp(T);             benchmark(temp); }
  { ForEach<Vector>                     temp(A);             benchmark(temp); } // for_each
  { Gather<Vector>                      temp(S,A,B);         benchmark(temp); } // gather
  { GatherIf<Vector>                    temp(S,D,A,B);       benchmark(temp); }
  { Generate<Vector>                    temp(A);             benchmark(temp); } // generate
  { GenerateN<Vector>                   temp(A);             benchmark(temp); }
  { InnerProduct<Vector>                temp(A,B);           benchmark(temp); } // inner_product
  { AllOf<Vector>                       temp(T);             benchmark(temp); } // logical
  { AnyOf<Vector>                       temp(F);             benchmark(temp); }
  { NoneOf<Vector>                      temp(F);             benchmark(temp); }
  { Merge<Vector>                       temp(A,B,U1);        benchmark(temp); } // merge
  { Mismatch<Vector>                    temp(A,A);           benchmark(temp); } // mismatch
  { Partition<Vector>                   temp(A);             benchmark(temp); } // partition
  { PartitionCopy<Vector>               temp(D,A,B);         benchmark(temp); }
  { StablePartition<Vector>             temp(A);             benchmark(temp); }
  { StablePartitionCopy<Vector>         temp(D,A,B);         benchmark(temp); }
  { IsPartitioned<Vector>               temp(T);             benchmark(temp); }
  { PartitionPoint<Vector>              temp(T);             benchmark(temp); }
  { Reduce<Vector>                      temp(A);             benchmark(temp); } // reduce
  { ReduceByKey<Vector>                 temp(D,A,B,C);       benchmark(temp); }
  { Remove<Vector>                      temp(D,0);           benchmark(temp); } // remove
  { RemoveCopy<Vector>                  temp(D,A,0);         benchmark(temp); }
  { RemoveIf<Vector>                    temp(A,D);           benchmark(temp); }
  { RemoveCopyIf<Vector>                temp(A,D,B);         benchmark(temp); }
  { Replace<Vector>                     temp(D,0,2);         benchmark(temp); } // replace
  { ReplaceCopy<Vector>                 temp(D,A,0,2);       benchmark(temp); }
  { ReplaceIf<Vector>                   temp(A,D,I,0);       benchmark(temp); }
  { ReplaceCopyIf<Vector>               temp(A,D,B,I,0);     benchmark(temp); }
  { Reverse<Vector>                     temp(A);             benchmark(temp); }
  { ReverseCopy<Vector>                 temp(A,B);           benchmark(temp); }
  { InclusiveScan<Vector>               temp(A,B);           benchmark(temp); }
  { ExclusiveScan<Vector>               temp(A,B);           benchmark(temp); }
  { InclusiveScanByKey<Vector>          temp(D,A,B);         benchmark(temp); }
  { ExclusiveScanByKey<Vector>          temp(D,A,B);         benchmark(temp); }
  { Scatter<Vector>                     temp(A,S,B);         benchmark(temp); } // scatter
  { ScatterIf<Vector>                   temp(A,S,D,B);       benchmark(temp); }
  { Sequence<Vector>                    temp(A);             benchmark(temp); } // sequence
  { SetDifference<Vector>               temp(A,B,U1);        benchmark(temp); } // set_operations
  { SetIntersection<Vector>             temp(A,B,U1);        benchmark(temp); }
  { SetSymmetricDifference<Vector>      temp(A,B,U1);        benchmark(temp); }
  { SetUnion<Vector>                    temp(A,B,U1);        benchmark(temp); }
  { Sort<Vector>                        temp(A);             benchmark(temp); } // sort
  { SortByKey<Vector>                   temp(A,B);           benchmark(temp); }
  { StableSort<Vector>                  temp(A);             benchmark(temp); }
  { StableSortByKey<Vector>             temp(A,B);           benchmark(temp); }
  { ComparisonSort<Vector>              temp(A);             benchmark(temp); }
  { ComparisonSortByKey<Vector>         temp(A,B);           benchmark(temp); }
  { IsSorted<Vector>                    temp(S);             benchmark(temp); }
  { IsSortedUntil<Vector>               temp(S);             benchmark(temp); }
  { SwapRanges<Vector>                  temp(A,B);           benchmark(temp); } // swap
  { UnaryTransform<Vector>              temp(A,B);           benchmark(temp); } // transform
  { BinaryTransform<Vector>             temp(A,B,C);         benchmark(temp); }
  { UnaryTransformIf<Vector>            temp(A,D,B);         benchmark(temp); }
  { BinaryTransformIf<Vector>           temp(A,B,D,C);       benchmark(temp); }
  { TransformReduce<Vector>             temp(A);             benchmark(temp); } // transform_reduce
  { TransformInclusiveScan<Vector>      temp(A,B);           benchmark(temp); } // transform_scan
  { TransformExclusiveScan<Vector>      temp(A,B);           benchmark(temp); }
  { UninitializedCopy<Vector>           temp(A,B);           benchmark(temp); } // uninitialized_copy
  { UninitializedFill<Vector>           temp(A);             benchmark(temp); } // fill
  { UninitializedFillN<Vector>          temp(A);             benchmark(temp); }
  { Unique<Vector>                      temp(D);             benchmark(temp); } // unique
  { UniqueCopy<Vector>                  temp(D,A);           benchmark(temp); }
  { UniqueByKey<Vector>                 temp(D,A);           benchmark(temp); }
  { UniqueByKeyCopy<Vector>             temp(D,A,B,C);       benchmark(temp); }

#if THRUST_VERSION > 100700
  { MergeByKey<Vector>                  temp(A,B,C,D,U1,U2); benchmark(temp); } // merge_by_key
  { SetDifferenceByKey<Vector>          temp(A,B,C,D,U1,U2); benchmark(temp); } // set_operations by_key
  { SetIntersectionByKey<Vector>        temp(A,B,C,U1,U2);   benchmark(temp); }
  { SetSymmetricDifferenceByKey<Vector> temp(A,B,C,D,U1,U2); benchmark(temp); }
  { SetUnionByKey<Vector>               temp(A,B,C,D,U1,U2); benchmark(temp); }
  { Tabulate<Vector>                    temp(A);             benchmark(temp); } // tabulate
#endif

  // host<->device copy

  return 0;
}

