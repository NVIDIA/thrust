---
title: Shuffling
parent: Reordering
grand_parent: Algorithms
nav_exclude: false
has_children: true
has_toc: false
---

# Shuffling

<code class="doxybook">
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename RandomIterator,</span>
<span>&nbsp;&nbsp;typename URBG&gt;</span>
<span>__host__ __device__ void </span><span><b><a href="{{ site.baseurl }}/api/groups/group__shuffling.html#function-shuffle">thrust::shuffle</a></b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;RandomIterator first,</span>
<span>&nbsp;&nbsp;RandomIterator last,</span>
<span>&nbsp;&nbsp;URBG && g);</span>
<br>
<span>template &lt;typename RandomIterator,</span>
<span>&nbsp;&nbsp;typename URBG&gt;</span>
<span>__host__ __device__ void </span><span><b><a href="{{ site.baseurl }}/api/groups/group__shuffling.html#function-shuffle">thrust::shuffle</a></b>(RandomIterator first,</span>
<span>&nbsp;&nbsp;RandomIterator last,</span>
<span>&nbsp;&nbsp;URBG && g);</span>
<br>
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename RandomIterator,</span>
<span>&nbsp;&nbsp;typename OutputIterator,</span>
<span>&nbsp;&nbsp;typename URBG&gt;</span>
<span>__host__ __device__ void </span><span><b><a href="{{ site.baseurl }}/api/groups/group__shuffling.html#function-shuffle-copy">thrust::shuffle&#95;copy</a></b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;RandomIterator first,</span>
<span>&nbsp;&nbsp;RandomIterator last,</span>
<span>&nbsp;&nbsp;OutputIterator result,</span>
<span>&nbsp;&nbsp;URBG && g);</span>
<br>
<span>template &lt;typename RandomIterator,</span>
<span>&nbsp;&nbsp;typename OutputIterator,</span>
<span>&nbsp;&nbsp;typename URBG&gt;</span>
<span>__host__ __device__ void </span><span><b><a href="{{ site.baseurl }}/api/groups/group__shuffling.html#function-shuffle-copy">thrust::shuffle&#95;copy</a></b>(RandomIterator first,</span>
<span>&nbsp;&nbsp;RandomIterator last,</span>
<span>&nbsp;&nbsp;OutputIterator result,</span>
<span>&nbsp;&nbsp;URBG && g);</span>
</code>

## Functions

<h3 id="function-shuffle">
Function <code>thrust::shuffle</code>
</h3>

<code class="doxybook">
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename RandomIterator,</span>
<span>&nbsp;&nbsp;typename URBG&gt;</span>
<span>__host__ __device__ void </span><span><b>shuffle</b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;RandomIterator first,</span>
<span>&nbsp;&nbsp;RandomIterator last,</span>
<span>&nbsp;&nbsp;URBG && g);</span></code>
<code>shuffle</code> reorders the elements <code>[first, last)</code> by a uniform pseudorandom permutation, defined by random engine <code>g</code>.

The algorithm's execution is parallelized as determined by <code>exec</code>.


The following code snippet demonstrates how to use <code>shuffle</code> to create a random permutation using the <code>thrust::host</code> execution policy for parallelization:



```cpp
#include <thrust/shuffle.h>
#include <thrust/random.h>
#include <thrust/execution_policy.h>
int A[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
const int N = sizeof(A)/sizeof(int);
thrust::default_random_engine g;
thrust::shuffle(thrust::host, A, A + N, g);
// A is now {6, 5, 8, 7, 2, 1, 4, 3, 10, 9}
```

**Template Parameters**:
* **`DerivedPolicy`** The name of the derived execution policy. 
* **`RandomIterator`** is a random access iterator 
* **`URBG`** is a uniform random bit generator

**Function Parameters**:
* **`exec`** The execution policy to use for parallelization. 
* **`first`** The beginning of the sequence to shuffle. 
* **`last`** The end of the sequence to shuffle. 
* **`g`** A UniformRandomBitGenerator

**See**:
<code>shuffle&#95;copy</code>

<h3 id="function-shuffle">
Function <code>thrust::shuffle</code>
</h3>

<code class="doxybook">
<span>template &lt;typename RandomIterator,</span>
<span>&nbsp;&nbsp;typename URBG&gt;</span>
<span>__host__ __device__ void </span><span><b>shuffle</b>(RandomIterator first,</span>
<span>&nbsp;&nbsp;RandomIterator last,</span>
<span>&nbsp;&nbsp;URBG && g);</span></code>
<code>shuffle</code> reorders the elements <code>[first, last)</code> by a uniform pseudorandom permutation, defined by random engine <code>g</code>.


The following code snippet demonstrates how to use <code>shuffle</code> to create a random permutation.



```cpp
#include <thrust/shuffle.h>
#include <thrust/random.h>
int A[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
const int N = sizeof(A)/sizeof(int);
thrust::default_random_engine g;
thrust::shuffle(A, A + N, g);
// A is now {6, 5, 8, 7, 2, 1, 4, 3, 10, 9}
```

**Template Parameters**:
* **`RandomIterator`** is a random access iterator 
* **`URBG`** is a uniform random bit generator

**Function Parameters**:
* **`first`** The beginning of the sequence to shuffle. 
* **`last`** The end of the sequence to shuffle. 
* **`g`** A UniformRandomBitGenerator

**See**:
<code>shuffle&#95;copy</code>

<h3 id="function-shuffle-copy">
Function <code>thrust::shuffle&#95;copy</code>
</h3>

<code class="doxybook">
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename RandomIterator,</span>
<span>&nbsp;&nbsp;typename OutputIterator,</span>
<span>&nbsp;&nbsp;typename URBG&gt;</span>
<span>__host__ __device__ void </span><span><b>shuffle_copy</b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;RandomIterator first,</span>
<span>&nbsp;&nbsp;RandomIterator last,</span>
<span>&nbsp;&nbsp;OutputIterator result,</span>
<span>&nbsp;&nbsp;URBG && g);</span></code>
shuffle_copy differs from shuffle only in that the reordered sequence is written to different output sequences, rather than in place. <code>shuffle&#95;copy</code> reorders the elements <code>[first, last)</code> by a uniform pseudorandom permutation, defined by random engine <code>g</code>.

The algorithm's execution is parallelized as determined by <code>exec</code>.


The following code snippet demonstrates how to use <code>shuffle&#95;copy</code> to create a random permutation.



```cpp
#include <thrust/shuffle.h>
#include <thrust/random.h>
#include <thrust/execution_policy.h>
int A[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
int result[10];
const int N = sizeof(A)/sizeof(int);
thrust::default_random_engine g;
thrust::shuffle_copy(thrust::host, A, A + N, result, g);
// result is now {6, 5, 8, 7, 2, 1, 4, 3, 10, 9}
```

**Template Parameters**:
* **`DerivedPolicy`** The name of the derived execution policy. 
* **`RandomIterator`** is a random access iterator 
* **`OutputIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>. 
* **`URBG`** is a uniform random bit generator

**Function Parameters**:
* **`exec`** The execution policy to use for parallelization. 
* **`first`** The beginning of the sequence to shuffle. 
* **`last`** The end of the sequence to shuffle. 
* **`result`** Destination of shuffled sequence 
* **`g`** A UniformRandomBitGenerator

**See**:
<code>shuffle</code>

<h3 id="function-shuffle-copy">
Function <code>thrust::shuffle&#95;copy</code>
</h3>

<code class="doxybook">
<span>template &lt;typename RandomIterator,</span>
<span>&nbsp;&nbsp;typename OutputIterator,</span>
<span>&nbsp;&nbsp;typename URBG&gt;</span>
<span>__host__ __device__ void </span><span><b>shuffle_copy</b>(RandomIterator first,</span>
<span>&nbsp;&nbsp;RandomIterator last,</span>
<span>&nbsp;&nbsp;OutputIterator result,</span>
<span>&nbsp;&nbsp;URBG && g);</span></code>
shuffle_copy differs from shuffle only in that the reordered sequence is written to different output sequences, rather than in place. <code>shuffle&#95;copy</code> reorders the elements <code>[first, last)</code> by a uniform pseudorandom permutation, defined by random engine <code>g</code>.


The following code snippet demonstrates how to use <code>shuffle&#95;copy</code> to create a random permutation.



```cpp
#include <thrust/shuffle.h>
#include <thrust/random.h>
int A[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
int result[10];
const int N = sizeof(A)/sizeof(int);
thrust::default_random_engine g;
thrust::shuffle_copy(A, A + N, result, g);
// result is now {6, 5, 8, 7, 2, 1, 4, 3, 10, 9}
```

**Template Parameters**:
* **`RandomIterator`** is a random access iterator 
* **`OutputIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>. 
* **`URBG`** is a uniform random bit generator

**Function Parameters**:
* **`first`** The beginning of the sequence to shuffle. 
* **`last`** The end of the sequence to shuffle. 
* **`result`** Destination of shuffled sequence 
* **`g`** A UniformRandomBitGenerator

**See**:
<code>shuffle</code>


