---
title: Filling
parent: Transformations
grand_parent: Algorithms
nav_exclude: false
has_children: true
has_toc: false
---

# Filling

<code class="doxybook">
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename ForwardIterator,</span>
<span>&nbsp;&nbsp;typename T&gt;</span>
<span>__host__ __device__ void </span><span><b><a href="{{ site.baseurl }}/api/groups/group__filling.html#function-fill">thrust::fill</a></b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;ForwardIterator first,</span>
<span>&nbsp;&nbsp;ForwardIterator last,</span>
<span>&nbsp;&nbsp;const T & value);</span>
<br>
<span>template &lt;typename ForwardIterator,</span>
<span>&nbsp;&nbsp;typename T&gt;</span>
<span>__host__ __device__ void </span><span><b><a href="{{ site.baseurl }}/api/groups/group__filling.html#function-fill">thrust::fill</a></b>(ForwardIterator first,</span>
<span>&nbsp;&nbsp;ForwardIterator last,</span>
<span>&nbsp;&nbsp;const T & value);</span>
<br>
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename OutputIterator,</span>
<span>&nbsp;&nbsp;typename Size,</span>
<span>&nbsp;&nbsp;typename T&gt;</span>
<span>__host__ __device__ OutputIterator </span><span><b><a href="{{ site.baseurl }}/api/groups/group__filling.html#function-fill-n">thrust::fill&#95;n</a></b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;OutputIterator first,</span>
<span>&nbsp;&nbsp;Size n,</span>
<span>&nbsp;&nbsp;const T & value);</span>
<br>
<span>template &lt;typename OutputIterator,</span>
<span>&nbsp;&nbsp;typename Size,</span>
<span>&nbsp;&nbsp;typename T&gt;</span>
<span>__host__ __device__ OutputIterator </span><span><b><a href="{{ site.baseurl }}/api/groups/group__filling.html#function-fill-n">thrust::fill&#95;n</a></b>(OutputIterator first,</span>
<span>&nbsp;&nbsp;Size n,</span>
<span>&nbsp;&nbsp;const T & value);</span>
<br>
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename ForwardIterator,</span>
<span>&nbsp;&nbsp;typename T&gt;</span>
<span>__host__ __device__ void </span><span><b><a href="{{ site.baseurl }}/api/groups/group__filling.html#function-uninitialized-fill">thrust::uninitialized&#95;fill</a></b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;ForwardIterator first,</span>
<span>&nbsp;&nbsp;ForwardIterator last,</span>
<span>&nbsp;&nbsp;const T & x);</span>
<br>
<span>template &lt;typename ForwardIterator,</span>
<span>&nbsp;&nbsp;typename T&gt;</span>
<span>void </span><span><b><a href="{{ site.baseurl }}/api/groups/group__filling.html#function-uninitialized-fill">thrust::uninitialized&#95;fill</a></b>(ForwardIterator first,</span>
<span>&nbsp;&nbsp;ForwardIterator last,</span>
<span>&nbsp;&nbsp;const T & x);</span>
<br>
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename ForwardIterator,</span>
<span>&nbsp;&nbsp;typename Size,</span>
<span>&nbsp;&nbsp;typename T&gt;</span>
<span>__host__ __device__ ForwardIterator </span><span><b><a href="{{ site.baseurl }}/api/groups/group__filling.html#function-uninitialized-fill-n">thrust::uninitialized&#95;fill&#95;n</a></b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;ForwardIterator first,</span>
<span>&nbsp;&nbsp;Size n,</span>
<span>&nbsp;&nbsp;const T & x);</span>
<br>
<span>template &lt;typename ForwardIterator,</span>
<span>&nbsp;&nbsp;typename Size,</span>
<span>&nbsp;&nbsp;typename T&gt;</span>
<span>ForwardIterator </span><span><b><a href="{{ site.baseurl }}/api/groups/group__filling.html#function-uninitialized-fill-n">thrust::uninitialized&#95;fill&#95;n</a></b>(ForwardIterator first,</span>
<span>&nbsp;&nbsp;Size n,</span>
<span>&nbsp;&nbsp;const T & x);</span>
</code>

## Functions

<h3 id="function-fill">
Function <code>thrust::fill</code>
</h3>

<code class="doxybook">
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename ForwardIterator,</span>
<span>&nbsp;&nbsp;typename T&gt;</span>
<span>__host__ __device__ void </span><span><b>fill</b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;ForwardIterator first,</span>
<span>&nbsp;&nbsp;ForwardIterator last,</span>
<span>&nbsp;&nbsp;const T & value);</span></code>
<code>fill</code> assigns the value <code>value</code> to every element in the range <code>[first, last)</code>. That is, for every iterator <code>i</code> in <code>[first, last)</code>, it performs the assignment <code>&#42;i = value</code>.

The algorithm's execution is parallelized as determined by <code>exec</code>.


The following code snippet demonstrates how to use <code>fill</code> to set a <a href="{{ site.baseurl }}/api/classes/classthrust_1_1device__vector.html">thrust::device_vector</a>'s elements to a given value using the <code>thrust::device</code> execution policy for parallelization:



```cpp
#include <thrust/fill.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
...
thrust::device_vector<int> v(4);
thrust::fill(thrust::device, v.begin(), v.end(), 137);

// v[0] == 137, v[1] == 137, v[2] == 137, v[3] == 137
```

**Template Parameters**:
* **`DerivedPolicy`** The name of the derived execution policy. 
* **`ForwardIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a>, and <code>ForwardIterator</code> is mutable. 
* **`T`** is a model of <a href="https://en.cppreference.com/w/cpp/named_req/CopyAssignable">Assignable</a>, and <code>T's</code><code>value&#95;type</code> is convertible to <code>ForwardIterator's</code><code>value&#95;type</code>.

**Function Parameters**:
* **`exec`** The execution policy to use for parallelization. 
* **`first`** The beginning of the sequence. 
* **`last`** The end of the sequence. 
* **`value`** The value to be copied.

**See**:
* <a href="https://en.cppreference.com/w/cpp/algorithm/fill">https://en.cppreference.com/w/cpp/algorithm/fill</a>
* <code>fill&#95;n</code>
* <code>uninitialized&#95;fill</code>

<h3 id="function-fill">
Function <code>thrust::fill</code>
</h3>

<code class="doxybook">
<span>template &lt;typename ForwardIterator,</span>
<span>&nbsp;&nbsp;typename T&gt;</span>
<span>__host__ __device__ void </span><span><b>fill</b>(ForwardIterator first,</span>
<span>&nbsp;&nbsp;ForwardIterator last,</span>
<span>&nbsp;&nbsp;const T & value);</span></code>
<code>fill</code> assigns the value <code>value</code> to every element in the range <code>[first, last)</code>. That is, for every iterator <code>i</code> in <code>[first, last)</code>, it performs the assignment <code>&#42;i = value</code>.


The following code snippet demonstrates how to use <code>fill</code> to set a <a href="{{ site.baseurl }}/api/classes/classthrust_1_1device__vector.html">thrust::device_vector</a>'s elements to a given value.



```cpp
#include <thrust/fill.h>
#include <thrust/device_vector.h>
...
thrust::device_vector<int> v(4);
thrust::fill(v.begin(), v.end(), 137);

// v[0] == 137, v[1] == 137, v[2] == 137, v[3] == 137
```

**Template Parameters**:
* **`ForwardIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a>, and <code>ForwardIterator</code> is mutable. 
* **`T`** is a model of <a href="https://en.cppreference.com/w/cpp/named_req/CopyAssignable">Assignable</a>, and <code>T's</code><code>value&#95;type</code> is convertible to <code>ForwardIterator's</code><code>value&#95;type</code>.

**Function Parameters**:
* **`first`** The beginning of the sequence. 
* **`last`** The end of the sequence. 
* **`value`** The value to be copied.

**See**:
* <a href="https://en.cppreference.com/w/cpp/algorithm/fill">https://en.cppreference.com/w/cpp/algorithm/fill</a>
* <code>fill&#95;n</code>
* <code>uninitialized&#95;fill</code>

<h3 id="function-fill-n">
Function <code>thrust::fill&#95;n</code>
</h3>

<code class="doxybook">
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename OutputIterator,</span>
<span>&nbsp;&nbsp;typename Size,</span>
<span>&nbsp;&nbsp;typename T&gt;</span>
<span>__host__ __device__ OutputIterator </span><span><b>fill_n</b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;OutputIterator first,</span>
<span>&nbsp;&nbsp;Size n,</span>
<span>&nbsp;&nbsp;const T & value);</span></code>
<code>fill&#95;n</code> assigns the value <code>value</code> to every element in the range <code>[first, first+n)</code>. That is, for every iterator <code>i</code> in <code>[first, first+n)</code>, it performs the assignment <code>&#42;i = value</code>.

The algorithm's execution is parallelized as determined by <code>exec</code>.


The following code snippet demonstrates how to use <code>fill</code> to set a <a href="{{ site.baseurl }}/api/classes/classthrust_1_1device__vector.html">thrust::device_vector</a>'s elements to a given value using the <code>thrust::device</code> execution policy for parallelization:



```cpp
#include <thrust/fill.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
...
thrust::device_vector<int> v(4);
thrust::fill_n(thrust::device, v.begin(), v.size(), 137);

// v[0] == 137, v[1] == 137, v[2] == 137, v[3] == 137
```

**Template Parameters**:
* **`DerivedPolicy`** The name of the derived execution policy. 
* **`OutputIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>. 
* **`T`** is a model of <a href="https://en.cppreference.com/w/cpp/named_req/CopyAssignable">Assignable</a>, and <code>T's</code><code>value&#95;type</code> is convertible to a type in <code>OutputIterator's</code> set of <code>value&#95;type</code>.

**Function Parameters**:
* **`exec`** The execution policy to use for parallelization. 
* **`first`** The beginning of the sequence. 
* **`n`** The size of the sequence. 
* **`value`** The value to be copied. 

**Returns**:
<code>first + n</code>

**See**:
* <a href="https://en.cppreference.com/w/cpp/algorithm/fill_n">https://en.cppreference.com/w/cpp/algorithm/fill_n</a>
* <code>fill</code>
* <code>uninitialized&#95;fill&#95;n</code>

<h3 id="function-fill-n">
Function <code>thrust::fill&#95;n</code>
</h3>

<code class="doxybook">
<span>template &lt;typename OutputIterator,</span>
<span>&nbsp;&nbsp;typename Size,</span>
<span>&nbsp;&nbsp;typename T&gt;</span>
<span>__host__ __device__ OutputIterator </span><span><b>fill_n</b>(OutputIterator first,</span>
<span>&nbsp;&nbsp;Size n,</span>
<span>&nbsp;&nbsp;const T & value);</span></code>
<code>fill&#95;n</code> assigns the value <code>value</code> to every element in the range <code>[first, first+n)</code>. That is, for every iterator <code>i</code> in <code>[first, first+n)</code>, it performs the assignment <code>&#42;i = value</code>.


The following code snippet demonstrates how to use <code>fill</code> to set a <a href="{{ site.baseurl }}/api/classes/classthrust_1_1device__vector.html">thrust::device_vector</a>'s elements to a given value.



```cpp
#include <thrust/fill.h>
#include <thrust/device_vector.h>
...
thrust::device_vector<int> v(4);
thrust::fill_n(v.begin(), v.size(), 137);

// v[0] == 137, v[1] == 137, v[2] == 137, v[3] == 137
```

**Template Parameters**:
* **`OutputIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>. 
* **`T`** is a model of <a href="https://en.cppreference.com/w/cpp/named_req/CopyAssignable">Assignable</a>, and <code>T's</code><code>value&#95;type</code> is convertible to a type in <code>OutputIterator's</code> set of <code>value&#95;type</code>.

**Function Parameters**:
* **`first`** The beginning of the sequence. 
* **`n`** The size of the sequence. 
* **`value`** The value to be copied. 

**Returns**:
<code>first + n</code>

**See**:
* <a href="https://en.cppreference.com/w/cpp/algorithm/fill_n">https://en.cppreference.com/w/cpp/algorithm/fill_n</a>
* <code>fill</code>
* <code>uninitialized&#95;fill&#95;n</code>

<h3 id="function-uninitialized-fill">
Function <code>thrust::uninitialized&#95;fill</code>
</h3>

<code class="doxybook">
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename ForwardIterator,</span>
<span>&nbsp;&nbsp;typename T&gt;</span>
<span>__host__ __device__ void </span><span><b>uninitialized_fill</b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;ForwardIterator first,</span>
<span>&nbsp;&nbsp;ForwardIterator last,</span>
<span>&nbsp;&nbsp;const T & x);</span></code>
In <code>thrust</code>, the function <code>thrust::device&#95;new</code> allocates memory for an object and then creates an object at that location by calling a constructor. Occasionally, however, it is useful to separate those two operations. If each iterator in the range <code>[first, last)</code> points to uninitialized memory, then <code>uninitialized&#95;fill</code> creates copies of <code>x</code> in that range. That is, for each iterator <code>i</code> in the range <code>[first, last)</code>, <code>uninitialized&#95;fill</code> creates a copy of <code>x</code> in the location pointed to <code>i</code> by calling <code>ForwardIterator's</code><code>value&#95;type's</code> copy constructor.

The algorithm's execution is parallelized as determined by <code>exec</code>.


The following code snippet demonstrates how to use <code>uninitialized&#95;fill</code> to initialize a range of uninitialized memory using the <code>thrust::device</code> execution policy for parallelization:



```cpp
#include <thrust/uninitialized_fill.h>
#include <thrust/device_malloc.h>
#include <thrust/execution_policy.h>

struct Int
{
  __host__ __device__
  Int(int x) : val(x) {}
  int val;
};  
...
const int N = 137;

Int val(46);
thrust::device_ptr<Int> array = thrust::device_malloc<Int>(N);
thrust::uninitialized_fill(thrust::device, array, array + N, val);

// Int x = array[i];
// x.val == 46 for all 0 <= i < N
```

**Template Parameters**:
* **`DerivedPolicy`** The name of the derived execution policy. 
* **`ForwardIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a>, <code>ForwardIterator</code> is mutable, and <code>ForwardIterator's</code><code>value&#95;type</code> has a constructor that takes a single argument of type <code>T</code>.

**Function Parameters**:
* **`exec`** The execution policy to use for parallelization. 
* **`first`** The first element of the range of interest. 
* **`last`** The last element of the range of interest. 
* **`x`** The value to use as the exemplar of the copy constructor.

**See**:
* <a href="https://en.cppreference.com/w/cpp/memory/uninitialized_fill">https://en.cppreference.com/w/cpp/memory/uninitialized_fill</a>
* <code>uninitialized&#95;fill&#95;n</code>
* <code>fill</code>
* <code>uninitialized&#95;copy</code>
* <code>device&#95;new</code>
* <code>device&#95;malloc</code>

<h3 id="function-uninitialized-fill">
Function <code>thrust::uninitialized&#95;fill</code>
</h3>

<code class="doxybook">
<span>template &lt;typename ForwardIterator,</span>
<span>&nbsp;&nbsp;typename T&gt;</span>
<span>void </span><span><b>uninitialized_fill</b>(ForwardIterator first,</span>
<span>&nbsp;&nbsp;ForwardIterator last,</span>
<span>&nbsp;&nbsp;const T & x);</span></code>
In <code>thrust</code>, the function <code>thrust::device&#95;new</code> allocates memory for an object and then creates an object at that location by calling a constructor. Occasionally, however, it is useful to separate those two operations. If each iterator in the range <code>[first, last)</code> points to uninitialized memory, then <code>uninitialized&#95;fill</code> creates copies of <code>x</code> in that range. That is, for each iterator <code>i</code> in the range <code>[first, last)</code>, <code>uninitialized&#95;fill</code> creates a copy of <code>x</code> in the location pointed to <code>i</code> by calling <code>ForwardIterator's</code><code>value&#95;type's</code> copy constructor.


The following code snippet demonstrates how to use <code>uninitialized&#95;fill</code> to initialize a range of uninitialized memory.



```cpp
#include <thrust/uninitialized_fill.h>
#include <thrust/device_malloc.h>

struct Int
{
  __host__ __device__
  Int(int x) : val(x) {}
  int val;
};  
...
const int N = 137;

Int val(46);
thrust::device_ptr<Int> array = thrust::device_malloc<Int>(N);
thrust::uninitialized_fill(array, array + N, val);

// Int x = array[i];
// x.val == 46 for all 0 <= i < N
```

**Template Parameters**:
**`ForwardIterator`**: is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a>, <code>ForwardIterator</code> is mutable, and <code>ForwardIterator's</code><code>value&#95;type</code> has a constructor that takes a single argument of type <code>T</code>.

**Function Parameters**:
* **`first`** The first element of the range of interest. 
* **`last`** The last element of the range of interest. 
* **`x`** The value to use as the exemplar of the copy constructor.

**See**:
* <a href="https://en.cppreference.com/w/cpp/memory/uninitialized_fill">https://en.cppreference.com/w/cpp/memory/uninitialized_fill</a>
* <code>uninitialized&#95;fill&#95;n</code>
* <code>fill</code>
* <code>uninitialized&#95;copy</code>
* <code>device&#95;new</code>
* <code>device&#95;malloc</code>

<h3 id="function-uninitialized-fill-n">
Function <code>thrust::uninitialized&#95;fill&#95;n</code>
</h3>

<code class="doxybook">
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename ForwardIterator,</span>
<span>&nbsp;&nbsp;typename Size,</span>
<span>&nbsp;&nbsp;typename T&gt;</span>
<span>__host__ __device__ ForwardIterator </span><span><b>uninitialized_fill_n</b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;ForwardIterator first,</span>
<span>&nbsp;&nbsp;Size n,</span>
<span>&nbsp;&nbsp;const T & x);</span></code>
In <code>thrust</code>, the function <code>thrust::device&#95;new</code> allocates memory for an object and then creates an object at that location by calling a constructor. Occasionally, however, it is useful to separate those two operations. If each iterator in the range <code>[first, first+n)</code> points to uninitialized memory, then <code>uninitialized&#95;fill</code> creates copies of <code>x</code> in that range. That is, for each iterator <code>i</code> in the range <code>[first, first+n)</code>, <code>uninitialized&#95;fill</code> creates a copy of <code>x</code> in the location pointed to <code>i</code> by calling <code>ForwardIterator's</code><code>value&#95;type's</code> copy constructor.

The algorithm's execution is parallelized as determined by <code>exec</code>.


The following code snippet demonstrates how to use <code>uninitialized&#95;fill</code> to initialize a range of uninitialized memory using the <code>thrust::device</code> execution policy for parallelization:



```cpp
#include <thrust/uninitialized_fill.h>
#include <thrust/device_malloc.h>
#include <thrust/execution_policy.h>

struct Int
{
  __host__ __device__
  Int(int x) : val(x) {}
  int val;
};  
...
const int N = 137;

Int val(46);
thrust::device_ptr<Int> array = thrust::device_malloc<Int>(N);
thrust::uninitialized_fill_n(thrust::device, array, N, val);

// Int x = array[i];
// x.val == 46 for all 0 <= i < N
```

**Template Parameters**:
* **`DerivedPolicy`** The name of the derived execution policy. 
* **`ForwardIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a>, <code>ForwardIterator</code> is mutable, and <code>ForwardIterator's</code><code>value&#95;type</code> has a constructor that takes a single argument of type <code>T</code>.

**Function Parameters**:
* **`exec`** The execution policy to use for parallelization. 
* **`first`** The first element of the range of interest. 
* **`n`** The size of the range of interest. 
* **`x`** The value to use as the exemplar of the copy constructor. 

**Returns**:
<code>first+n</code>

**See**:
* <a href="https://en.cppreference.com/w/cpp/memory/uninitialized_fill">https://en.cppreference.com/w/cpp/memory/uninitialized_fill</a>
* <code>uninitialized&#95;fill</code>
* <code>fill</code>
* <code>uninitialized&#95;copy&#95;n</code>
* <code>device&#95;new</code>
* <code>device&#95;malloc</code>

<h3 id="function-uninitialized-fill-n">
Function <code>thrust::uninitialized&#95;fill&#95;n</code>
</h3>

<code class="doxybook">
<span>template &lt;typename ForwardIterator,</span>
<span>&nbsp;&nbsp;typename Size,</span>
<span>&nbsp;&nbsp;typename T&gt;</span>
<span>ForwardIterator </span><span><b>uninitialized_fill_n</b>(ForwardIterator first,</span>
<span>&nbsp;&nbsp;Size n,</span>
<span>&nbsp;&nbsp;const T & x);</span></code>
In <code>thrust</code>, the function <code>thrust::device&#95;new</code> allocates memory for an object and then creates an object at that location by calling a constructor. Occasionally, however, it is useful to separate those two operations. If each iterator in the range <code>[first, first+n)</code> points to uninitialized memory, then <code>uninitialized&#95;fill</code> creates copies of <code>x</code> in that range. That is, for each iterator <code>i</code> in the range <code>[first, first+n)</code>, <code>uninitialized&#95;fill</code> creates a copy of <code>x</code> in the location pointed to <code>i</code> by calling <code>ForwardIterator's</code><code>value&#95;type's</code> copy constructor.


The following code snippet demonstrates how to use <code>uninitialized&#95;fill</code> to initialize a range of uninitialized memory.



```cpp
#include <thrust/uninitialized_fill.h>
#include <thrust/device_malloc.h>

struct Int
{
  __host__ __device__
  Int(int x) : val(x) {}
  int val;
};  
...
const int N = 137;

Int val(46);
thrust::device_ptr<Int> array = thrust::device_malloc<Int>(N);
thrust::uninitialized_fill_n(array, N, val);

// Int x = array[i];
// x.val == 46 for all 0 <= i < N
```

**Template Parameters**:
**`ForwardIterator`**: is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a>, <code>ForwardIterator</code> is mutable, and <code>ForwardIterator's</code><code>value&#95;type</code> has a constructor that takes a single argument of type <code>T</code>.

**Function Parameters**:
* **`first`** The first element of the range of interest. 
* **`n`** The size of the range of interest. 
* **`x`** The value to use as the exemplar of the copy constructor. 

**Returns**:
<code>first+n</code>

**See**:
* <a href="https://en.cppreference.com/w/cpp/memory/uninitialized_fill">https://en.cppreference.com/w/cpp/memory/uninitialized_fill</a>
* <code>uninitialized&#95;fill</code>
* <code>fill</code>
* <code>uninitialized&#95;copy&#95;n</code>
* <code>device&#95;new</code>
* <code>device&#95;malloc</code>


