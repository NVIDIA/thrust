---
title: Copying
parent: Algorithms
grand_parent: API
nav_exclude: false
has_children: true
has_toc: false
---

# Copying

## Groups

* **[Gathering]({{ site.baseurl }}/api/groups/group__gathering.html)**
* **[Scattering]({{ site.baseurl }}/api/groups/group__scattering.html)**

<code class="doxybook">
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename InputIterator,</span>
<span>&nbsp;&nbsp;typename OutputIterator&gt;</span>
<span>__host__ __device__ OutputIterator </span><span><b><a href="{{ site.baseurl }}/api/groups/group__copying.html#function-copy">thrust::copy</a></b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;InputIterator first,</span>
<span>&nbsp;&nbsp;InputIterator last,</span>
<span>&nbsp;&nbsp;OutputIterator result);</span>
<br>
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename InputIterator,</span>
<span>&nbsp;&nbsp;typename Size,</span>
<span>&nbsp;&nbsp;typename OutputIterator&gt;</span>
<span>__host__ __device__ OutputIterator </span><span><b><a href="{{ site.baseurl }}/api/groups/group__copying.html#function-copy-n">thrust::copy&#95;n</a></b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;InputIterator first,</span>
<span>&nbsp;&nbsp;Size n,</span>
<span>&nbsp;&nbsp;OutputIterator result);</span>
<br>
<span>template &lt;typename InputIterator,</span>
<span>&nbsp;&nbsp;typename OutputIterator&gt;</span>
<span>OutputIterator </span><span><b><a href="{{ site.baseurl }}/api/groups/group__copying.html#function-copy">thrust::copy</a></b>(InputIterator first,</span>
<span>&nbsp;&nbsp;InputIterator last,</span>
<span>&nbsp;&nbsp;OutputIterator result);</span>
<br>
<span>template &lt;typename InputIterator,</span>
<span>&nbsp;&nbsp;typename Size,</span>
<span>&nbsp;&nbsp;typename OutputIterator&gt;</span>
<span>OutputIterator </span><span><b><a href="{{ site.baseurl }}/api/groups/group__copying.html#function-copy-n">thrust::copy&#95;n</a></b>(InputIterator first,</span>
<span>&nbsp;&nbsp;Size n,</span>
<span>&nbsp;&nbsp;OutputIterator result);</span>
<br>
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename ForwardIterator1,</span>
<span>&nbsp;&nbsp;typename ForwardIterator2&gt;</span>
<span>__host__ __device__ ForwardIterator2 </span><span><b><a href="{{ site.baseurl }}/api/groups/group__copying.html#function-swap-ranges">thrust::swap&#95;ranges</a></b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;ForwardIterator1 first1,</span>
<span>&nbsp;&nbsp;ForwardIterator1 last1,</span>
<span>&nbsp;&nbsp;ForwardIterator2 first2);</span>
<br>
<span>template &lt;typename ForwardIterator1,</span>
<span>&nbsp;&nbsp;typename ForwardIterator2&gt;</span>
<span>ForwardIterator2 </span><span><b><a href="{{ site.baseurl }}/api/groups/group__copying.html#function-swap-ranges">thrust::swap&#95;ranges</a></b>(ForwardIterator1 first1,</span>
<span>&nbsp;&nbsp;ForwardIterator1 last1,</span>
<span>&nbsp;&nbsp;ForwardIterator2 first2);</span>
<br>
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename InputIterator,</span>
<span>&nbsp;&nbsp;typename ForwardIterator&gt;</span>
<span>__host__ __device__ ForwardIterator </span><span><b><a href="{{ site.baseurl }}/api/groups/group__copying.html#function-uninitialized-copy">thrust::uninitialized&#95;copy</a></b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;InputIterator first,</span>
<span>&nbsp;&nbsp;InputIterator last,</span>
<span>&nbsp;&nbsp;ForwardIterator result);</span>
<br>
<span>template &lt;typename InputIterator,</span>
<span>&nbsp;&nbsp;typename ForwardIterator&gt;</span>
<span>ForwardIterator </span><span><b><a href="{{ site.baseurl }}/api/groups/group__copying.html#function-uninitialized-copy">thrust::uninitialized&#95;copy</a></b>(InputIterator first,</span>
<span>&nbsp;&nbsp;InputIterator last,</span>
<span>&nbsp;&nbsp;ForwardIterator result);</span>
<br>
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename InputIterator,</span>
<span>&nbsp;&nbsp;typename Size,</span>
<span>&nbsp;&nbsp;typename ForwardIterator&gt;</span>
<span>__host__ __device__ ForwardIterator </span><span><b><a href="{{ site.baseurl }}/api/groups/group__copying.html#function-uninitialized-copy-n">thrust::uninitialized&#95;copy&#95;n</a></b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;InputIterator first,</span>
<span>&nbsp;&nbsp;Size n,</span>
<span>&nbsp;&nbsp;ForwardIterator result);</span>
<br>
<span>template &lt;typename InputIterator,</span>
<span>&nbsp;&nbsp;typename Size,</span>
<span>&nbsp;&nbsp;typename ForwardIterator&gt;</span>
<span>ForwardIterator </span><span><b><a href="{{ site.baseurl }}/api/groups/group__copying.html#function-uninitialized-copy-n">thrust::uninitialized&#95;copy&#95;n</a></b>(InputIterator first,</span>
<span>&nbsp;&nbsp;Size n,</span>
<span>&nbsp;&nbsp;ForwardIterator result);</span>
</code>

## Functions

<h3 id="function-copy">
Function <code>thrust::copy</code>
</h3>

<code class="doxybook">
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename InputIterator,</span>
<span>&nbsp;&nbsp;typename OutputIterator&gt;</span>
<span>__host__ __device__ OutputIterator </span><span><b>copy</b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;InputIterator first,</span>
<span>&nbsp;&nbsp;InputIterator last,</span>
<span>&nbsp;&nbsp;OutputIterator result);</span></code>
<code>copy</code> copies elements from the range [<code>first</code>, <code>last</code>) to the range [<code>result</code>, <code>result</code> + (<code>last</code> - <code>first</code>)). That is, it performs the assignments *<code>result</code> = *<code>first</code>, *(<code>result</code> + <code>1</code>) = *(<code>first</code> + <code>1</code>), and so on. Generally, for every integer <code>n</code> from <code>0</code> to <code>last</code> - <code>first</code>, <code>copy</code> performs the assignment *(<code>result</code> + <code>n</code>) = *(<code>first</code> + <code>n</code>). Unlike <code>std::copy</code>, <code>copy</code> offers no guarantee on order of operation. As a result, calling <code>copy</code> with overlapping source and destination ranges has undefined behavior.

The return value is <code>result</code> + (<code>last</code> - <code>first</code>).

The algorithm's execution is parallelized as determined by <code>exec</code>.


The following code snippet demonstrates how to use <code>copy</code> to copy from one range to another using the <code>thrust::device</code> parallelization policy:



```cpp
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
...

thrust::device_vector<int> vec0(100);
thrust::device_vector<int> vec1(100);
...

thrust::copy(thrust::device, vec0.begin(), vec0.end(), vec1.begin());

// vec1 is now a copy of vec0
```

**Template Parameters**:
* **`DerivedPolicy`** The name of the derived execution policy. 
* **`InputIterator`** must be a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a> and <code>InputIterator's</code><code>value&#95;type</code> must be convertible to <code>OutputIterator's</code><code>value&#95;type</code>. 
* **`OutputIterator`** must be a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>.

**Function Parameters**:
* **`exec`** The execution policy to use for parallelization. 
* **`first`** The beginning of the sequence to copy. 
* **`last`** The end of the sequence to copy. 
* **`result`** The destination sequence. 

**Preconditions**:
<code>result</code> may be equal to <code>first</code>, but <code>result</code> shall not be in the range <code>[first, last)</code> otherwise.

**Returns**:
The end of the destination sequence. 

**See**:
<a href="https://en.cppreference.com/w/cpp/algorithm/copy">https://en.cppreference.com/w/cpp/algorithm/copy</a>

<h3 id="function-copy-n">
Function <code>thrust::copy&#95;n</code>
</h3>

<code class="doxybook">
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename InputIterator,</span>
<span>&nbsp;&nbsp;typename Size,</span>
<span>&nbsp;&nbsp;typename OutputIterator&gt;</span>
<span>__host__ __device__ OutputIterator </span><span><b>copy_n</b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;InputIterator first,</span>
<span>&nbsp;&nbsp;Size n,</span>
<span>&nbsp;&nbsp;OutputIterator result);</span></code>
<code>copy&#95;n</code> copies elements from the range <code>[first, first + n)</code> to the range <code>[result, result + n)</code>. That is, it performs the assignments <code>&#42;result = &#42;first, &#42;(result + 1) = &#42;(first + 1)</code>, and so on. Generally, for every integer <code>i</code> from <code>0</code> to <code>n</code>, <code>copy</code> performs the assignment *(<code>result</code> + <code>i</code>) = *(<code>first</code> + <code>i</code>). Unlike <code>std::copy&#95;n</code>, <code>copy&#95;n</code> offers no guarantee on order of operation. As a result, calling <code>copy&#95;n</code> with overlapping source and destination ranges has undefined behavior.

The return value is <code>result</code> + <code>n</code>.

The algorithm's execution is parallelized as determined by <code>exec</code>.


The following code snippet demonstrates how to use <code>copy</code> to copy from one range to another using the <code>thrust::device</code> parallelization policy:



```cpp
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
...
size_t n = 100;
thrust::device_vector<int> vec0(n);
thrust::device_vector<int> vec1(n);
...
thrust::copy_n(thrust::device, vec0.begin(), n, vec1.begin());

// vec1 is now a copy of vec0
```

**Template Parameters**:
* **`DerivedPolicy`** The name of the derived execution policy. 
* **`InputIterator`** must be a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a> and <code>InputIterator's</code><code>value&#95;type</code> must be convertible to <code>OutputIterator's</code><code>value&#95;type</code>. 
* **`Size`** is an integral type. 
* **`OutputIterator`** must be a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>.

**Function Parameters**:
* **`exec`** The execution policy to use for parallelization. 
* **`first`** The beginning of the range to copy. 
* **`n`** The number of elements to copy. 
* **`result`** The beginning destination range. 

**Preconditions**:
<code>result</code> may be equal to <code>first</code>, but <code>result</code> shall not be in the range <code>[first, first + n)</code> otherwise.

**Returns**:
The end of the destination range.

**See**:
* <a href="https://en.cppreference.com/w/cpp/algorithm/copy_n">https://en.cppreference.com/w/cpp/algorithm/copy_n</a>
* thrust::copy 

<h3 id="function-copy">
Function <code>thrust::copy</code>
</h3>

<code class="doxybook">
<span>template &lt;typename InputIterator,</span>
<span>&nbsp;&nbsp;typename OutputIterator&gt;</span>
<span>OutputIterator </span><span><b>copy</b>(InputIterator first,</span>
<span>&nbsp;&nbsp;InputIterator last,</span>
<span>&nbsp;&nbsp;OutputIterator result);</span></code>
<code>copy</code> copies elements from the range [<code>first</code>, <code>last</code>) to the range [<code>result</code>, <code>result</code> + (<code>last</code> - <code>first</code>)). That is, it performs the assignments *<code>result</code> = *<code>first</code>, *(<code>result</code> + <code>1</code>) = *(<code>first</code> + <code>1</code>), and so on. Generally, for every integer <code>n</code> from <code>0</code> to <code>last</code> - <code>first</code>, <code>copy</code> performs the assignment *(<code>result</code> + <code>n</code>) = *(<code>first</code> + <code>n</code>). Unlike <code>std::copy</code>, <code>copy</code> offers no guarantee on order of operation. As a result, calling <code>copy</code> with overlapping source and destination ranges has undefined behavior.

The return value is <code>result</code> + (<code>last</code> - <code>first</code>).


The following code snippet demonstrates how to use <code>copy</code> to copy from one range to another.



```cpp
#include <thrust/copy.h>
#include <thrust/device_vector.h>
...

thrust::device_vector<int> vec0(100);
thrust::device_vector<int> vec1(100);
...

thrust::copy(vec0.begin(), vec0.end(),
             vec1.begin());

// vec1 is now a copy of vec0
```

**Template Parameters**:
* **`InputIterator`** must be a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a> and <code>InputIterator's</code><code>value&#95;type</code> must be convertible to <code>OutputIterator's</code><code>value&#95;type</code>. 
* **`OutputIterator`** must be a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>.

**Function Parameters**:
* **`first`** The beginning of the sequence to copy. 
* **`last`** The end of the sequence to copy. 
* **`result`** The destination sequence. 

**Preconditions**:
<code>result</code> may be equal to <code>first</code>, but <code>result</code> shall not be in the range <code>[first, last)</code> otherwise.

**Returns**:
The end of the destination sequence. 

**See**:
<a href="https://en.cppreference.com/w/cpp/algorithm/copy">https://en.cppreference.com/w/cpp/algorithm/copy</a>

<h3 id="function-copy-n">
Function <code>thrust::copy&#95;n</code>
</h3>

<code class="doxybook">
<span>template &lt;typename InputIterator,</span>
<span>&nbsp;&nbsp;typename Size,</span>
<span>&nbsp;&nbsp;typename OutputIterator&gt;</span>
<span>OutputIterator </span><span><b>copy_n</b>(InputIterator first,</span>
<span>&nbsp;&nbsp;Size n,</span>
<span>&nbsp;&nbsp;OutputIterator result);</span></code>
<code>copy&#95;n</code> copies elements from the range <code>[first, first + n)</code> to the range <code>[result, result + n)</code>. That is, it performs the assignments <code>&#42;result = &#42;first, &#42;(result + 1) = &#42;(first + 1)</code>, and so on. Generally, for every integer <code>i</code> from <code>0</code> to <code>n</code>, <code>copy</code> performs the assignment *(<code>result</code> + <code>i</code>) = *(<code>first</code> + <code>i</code>). Unlike <code>std::copy&#95;n</code>, <code>copy&#95;n</code> offers no guarantee on order of operation. As a result, calling <code>copy&#95;n</code> with overlapping source and destination ranges has undefined behavior.

The return value is <code>result</code> + <code>n</code>.


The following code snippet demonstrates how to use <code>copy</code> to copy from one range to another.



```cpp
#include <thrust/copy.h>
#include <thrust/device_vector.h>
...
size_t n = 100;
thrust::device_vector<int> vec0(n);
thrust::device_vector<int> vec1(n);
...
thrust::copy_n(vec0.begin(), n, vec1.begin());

// vec1 is now a copy of vec0
```

**Template Parameters**:
* **`InputIterator`** must be a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a> and <code>InputIterator's</code><code>value&#95;type</code> must be convertible to <code>OutputIterator's</code><code>value&#95;type</code>. 
* **`Size`** is an integral type. 
* **`OutputIterator`** must be a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>.

**Function Parameters**:
* **`first`** The beginning of the range to copy. 
* **`n`** The number of elements to copy. 
* **`result`** The beginning destination range. 

**Preconditions**:
<code>result</code> may be equal to <code>first</code>, but <code>result</code> shall not be in the range <code>[first, first + n)</code> otherwise.

**Returns**:
The end of the destination range.

**See**:
* <a href="https://en.cppreference.com/w/cpp/algorithm/copy_n">https://en.cppreference.com/w/cpp/algorithm/copy_n</a>
* thrust::copy 

<h3 id="function-swap-ranges">
Function <code>thrust::swap&#95;ranges</code>
</h3>

<code class="doxybook">
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename ForwardIterator1,</span>
<span>&nbsp;&nbsp;typename ForwardIterator2&gt;</span>
<span>__host__ __device__ ForwardIterator2 </span><span><b>swap_ranges</b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;ForwardIterator1 first1,</span>
<span>&nbsp;&nbsp;ForwardIterator1 last1,</span>
<span>&nbsp;&nbsp;ForwardIterator2 first2);</span></code>
<code>swap&#95;ranges</code> swaps each of the elements in the range <code>[first1, last1)</code> with the corresponding element in the range <code>[first2, first2 + (last1 - first1))</code>. That is, for each integer <code>n</code> such that <code>0 &lt;= n &lt; (last1 - first1)</code>, it swaps <code>&#42;(first1 + n)</code> and <code>&#42;(first2 + n)</code>. The return value is <code>first2 + (last1 - first1)</code>.

The algorithm's execution is parallelized as determined by <code>exec</code>.


The following code snippet demonstrates how to use <code>swap&#95;ranges</code> to swap the contents of two <code>thrust::device&#95;vectors</code> using the <code>thrust::device</code> execution policy for parallelization:



```cpp
#include <thrust/swap.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
...
thrust::device_vector<int> v1(2), v2(2);
v1[0] = 1;
v1[1] = 2;
v2[0] = 3;
v2[1] = 4;

thrust::swap_ranges(thrust::device, v1.begin(), v1.end(), v2.begin());

// v1[0] == 3, v1[1] == 4, v2[0] == 1, v2[1] == 2
```

**Template Parameters**:
* **`DerivedPolicy`** The name of the derived execution policy. 
* **`ForwardIterator1`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a>, and <code>ForwardIterator1's</code><code>value&#95;type</code> must be convertible to <code>ForwardIterator2's</code><code>value&#95;type</code>. 
* **`ForwardIterator2`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a>, and <code>ForwardIterator2's</code><code>value&#95;type</code> must be convertible to <code>ForwardIterator1's</code><code>value&#95;type</code>.

**Function Parameters**:
* **`exec`** The execution policy to use for parallelization. 
* **`first1`** The beginning of the first sequence to swap. 
* **`last1`** One position past the last element of the first sequence to swap. 
* **`first2`** The beginning of the second sequence to swap. 

**Preconditions**:
<code>first1</code> may equal <code>first2</code>, but the range <code>[first1, last1)</code> shall not overlap the range <code>[first2, first2 + (last1 - first1))</code> otherwise.

**Returns**:
An iterator pointing to one position past the last element of the second sequence to swap.

**See**:
* <a href="https://en.cppreference.com/w/cpp/algorithm/swap_ranges">https://en.cppreference.com/w/cpp/algorithm/swap_ranges</a>
* <code><a href="{{ site.baseurl }}/api/groups/group__swap.html">Swap</a></code>

<h3 id="function-swap-ranges">
Function <code>thrust::swap&#95;ranges</code>
</h3>

<code class="doxybook">
<span>template &lt;typename ForwardIterator1,</span>
<span>&nbsp;&nbsp;typename ForwardIterator2&gt;</span>
<span>ForwardIterator2 </span><span><b>swap_ranges</b>(ForwardIterator1 first1,</span>
<span>&nbsp;&nbsp;ForwardIterator1 last1,</span>
<span>&nbsp;&nbsp;ForwardIterator2 first2);</span></code>
<code>swap&#95;ranges</code> swaps each of the elements in the range <code>[first1, last1)</code> with the corresponding element in the range <code>[first2, first2 + (last1 - first1))</code>. That is, for each integer <code>n</code> such that <code>0 &lt;= n &lt; (last1 - first1)</code>, it swaps <code>&#42;(first1 + n)</code> and <code>&#42;(first2 + n)</code>. The return value is <code>first2 + (last1 - first1)</code>.


The following code snippet demonstrates how to use <code>swap&#95;ranges</code> to swap the contents of two <code>thrust::device&#95;vectors</code>.



```cpp
#include <thrust/swap.h>
#include <thrust/device_vector.h>
...
thrust::device_vector<int> v1(2), v2(2);
v1[0] = 1;
v1[1] = 2;
v2[0] = 3;
v2[1] = 4;

thrust::swap_ranges(v1.begin(), v1.end(), v2.begin());

// v1[0] == 3, v1[1] == 4, v2[0] == 1, v2[1] == 2
```

**Template Parameters**:
* **`ForwardIterator1`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a>, and <code>ForwardIterator1's</code><code>value&#95;type</code> must be convertible to <code>ForwardIterator2's</code><code>value&#95;type</code>. 
* **`ForwardIterator2`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a>, and <code>ForwardIterator2's</code><code>value&#95;type</code> must be convertible to <code>ForwardIterator1's</code><code>value&#95;type</code>.

**Function Parameters**:
* **`first1`** The beginning of the first sequence to swap. 
* **`last1`** One position past the last element of the first sequence to swap. 
* **`first2`** The beginning of the second sequence to swap. 

**Preconditions**:
<code>first1</code> may equal <code>first2</code>, but the range <code>[first1, last1)</code> shall not overlap the range <code>[first2, first2 + (last1 - first1))</code> otherwise.

**Returns**:
An iterator pointing to one position past the last element of the second sequence to swap.

**See**:
* <a href="https://en.cppreference.com/w/cpp/algorithm/swap_ranges">https://en.cppreference.com/w/cpp/algorithm/swap_ranges</a>
* <code><a href="{{ site.baseurl }}/api/groups/group__swap.html">Swap</a></code>

<h3 id="function-uninitialized-copy">
Function <code>thrust::uninitialized&#95;copy</code>
</h3>

<code class="doxybook">
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename InputIterator,</span>
<span>&nbsp;&nbsp;typename ForwardIterator&gt;</span>
<span>__host__ __device__ ForwardIterator </span><span><b>uninitialized_copy</b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;InputIterator first,</span>
<span>&nbsp;&nbsp;InputIterator last,</span>
<span>&nbsp;&nbsp;ForwardIterator result);</span></code>
In <code>thrust</code>, the function <code>thrust::device&#95;new</code> allocates memory for an object and then creates an object at that location by calling a constructor. Occasionally, however, it is useful to separate those two operations. If each iterator in the range <code>[result, result + (last - first))</code> points to uninitialized memory, then <code>uninitialized&#95;copy</code> creates a copy of <code>[first, last)</code> in that range. That is, for each iterator <code>i</code> in the input, <code>uninitialized&#95;copy</code> creates a copy of <code>&#42;i</code> in the location pointed to by the corresponding iterator in the output range by <code>ForwardIterator's</code><code>value&#95;type's</code> copy constructor with *i as its argument.

The algorithm's execution is parallelized as determined by <code>exec</code>.


The following code snippet demonstrates how to use <code>uninitialized&#95;copy</code> to initialize a range of uninitialized memory using the <code>thrust::device</code> execution policy for parallelization:



```cpp
#include <thrust/uninitialized_copy.h>
#include <thrust/device_malloc.h>
#include <thrust/device_vector.h>
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
thrust::device_vector<Int> input(N, val);
thrust::device_ptr<Int> array = thrust::device_malloc<Int>(N);
thrust::uninitialized_copy(thrust::device, input.begin(), input.end(), array);

// Int x = array[i];
// x.val == 46 for all 0 <= i < N
```

**Template Parameters**:
* **`DerivedPolicy`** The name of the derived execution policy. 
* **`InputIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>. 
* **`ForwardIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a>, <code>ForwardIterator</code> is mutable, and <code>ForwardIterator's</code><code>value&#95;type</code> has a constructor that takes a single argument whose type is <code>InputIterator's</code><code>value&#95;type</code>.

**Function Parameters**:
* **`exec`** The execution policy to use for parallelization. 
* **`first`** The first element of the input range to copy from. 
* **`last`** The last element of the input range to copy from. 
* **`result`** The first element of the output range to copy to. 

**Preconditions**:
<code>first</code> may equal <code>result</code>, but the range <code>[first, last)</code> and the range <code>[result, result + (last - first))</code> shall not overlap otherwise.

**Returns**:
An iterator pointing to the last element of the output range.

**See**:
* <a href="https://en.cppreference.com/w/cpp/memory/uninitialized_copy">https://en.cppreference.com/w/cpp/memory/uninitialized_copy</a>
* <code>copy</code>
* <code>uninitialized&#95;fill</code>
* <code>device&#95;new</code>
* <code>device&#95;malloc</code>

<h3 id="function-uninitialized-copy">
Function <code>thrust::uninitialized&#95;copy</code>
</h3>

<code class="doxybook">
<span>template &lt;typename InputIterator,</span>
<span>&nbsp;&nbsp;typename ForwardIterator&gt;</span>
<span>ForwardIterator </span><span><b>uninitialized_copy</b>(InputIterator first,</span>
<span>&nbsp;&nbsp;InputIterator last,</span>
<span>&nbsp;&nbsp;ForwardIterator result);</span></code>
In <code>thrust</code>, the function <code>thrust::device&#95;new</code> allocates memory for an object and then creates an object at that location by calling a constructor. Occasionally, however, it is useful to separate those two operations. If each iterator in the range <code>[result, result + (last - first))</code> points to uninitialized memory, then <code>uninitialized&#95;copy</code> creates a copy of <code>[first, last)</code> in that range. That is, for each iterator <code>i</code> in the input, <code>uninitialized&#95;copy</code> creates a copy of <code>&#42;i</code> in the location pointed to by the corresponding iterator in the output range by <code>ForwardIterator's</code><code>value&#95;type's</code> copy constructor with *i as its argument.


The following code snippet demonstrates how to use <code>uninitialized&#95;copy</code> to initialize a range of uninitialized memory.



```cpp
#include <thrust/uninitialized_copy.h>
#include <thrust/device_malloc.h>
#include <thrust/device_vector.h>

struct Int
{
  __host__ __device__
  Int(int x) : val(x) {}
  int val;
};  
...
const int N = 137;

Int val(46);
thrust::device_vector<Int> input(N, val);
thrust::device_ptr<Int> array = thrust::device_malloc<Int>(N);
thrust::uninitialized_copy(input.begin(), input.end(), array);

// Int x = array[i];
// x.val == 46 for all 0 <= i < N
```

**Template Parameters**:
* **`InputIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>. 
* **`ForwardIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a>, <code>ForwardIterator</code> is mutable, and <code>ForwardIterator's</code><code>value&#95;type</code> has a constructor that takes a single argument whose type is <code>InputIterator's</code><code>value&#95;type</code>.

**Function Parameters**:
* **`first`** The first element of the input range to copy from. 
* **`last`** The last element of the input range to copy from. 
* **`result`** The first element of the output range to copy to. 

**Preconditions**:
<code>first</code> may equal <code>result</code>, but the range <code>[first, last)</code> and the range <code>[result, result + (last - first))</code> shall not overlap otherwise.

**Returns**:
An iterator pointing to the last element of the output range.

**See**:
* <a href="https://en.cppreference.com/w/cpp/memory/uninitialized_copy">https://en.cppreference.com/w/cpp/memory/uninitialized_copy</a>
* <code>copy</code>
* <code>uninitialized&#95;fill</code>
* <code>device&#95;new</code>
* <code>device&#95;malloc</code>

<h3 id="function-uninitialized-copy-n">
Function <code>thrust::uninitialized&#95;copy&#95;n</code>
</h3>

<code class="doxybook">
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename InputIterator,</span>
<span>&nbsp;&nbsp;typename Size,</span>
<span>&nbsp;&nbsp;typename ForwardIterator&gt;</span>
<span>__host__ __device__ ForwardIterator </span><span><b>uninitialized_copy_n</b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;InputIterator first,</span>
<span>&nbsp;&nbsp;Size n,</span>
<span>&nbsp;&nbsp;ForwardIterator result);</span></code>
In <code>thrust</code>, the function <code>thrust::device&#95;new</code> allocates memory for an object and then creates an object at that location by calling a constructor. Occasionally, however, it is useful to separate those two operations. If each iterator in the range <code>[result, result + n)</code> points to uninitialized memory, then <code>uninitialized&#95;copy&#95;n</code> creates a copy of <code>[first, first + n)</code> in that range. That is, for each iterator <code>i</code> in the input, <code>uninitialized&#95;copy&#95;n</code> creates a copy of <code>&#42;i</code> in the location pointed to by the corresponding iterator in the output range by <code>InputIterator's</code><code>value&#95;type's</code> copy constructor with *i as its argument.

The algorithm's execution is parallelized as determined by <code>exec</code>.


The following code snippet demonstrates how to use <code>uninitialized&#95;copy</code> to initialize a range of uninitialized memory using the <code>thrust::device</code> execution policy for parallelization:



```cpp
#include <thrust/uninitialized_copy.h>
#include <thrust/device_malloc.h>
#include <thrust/device_vector.h>
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
thrust::device_vector<Int> input(N, val);
thrust::device_ptr<Int> array = thrust::device_malloc<Int>(N);
thrust::uninitialized_copy_n(thrust::device, input.begin(), N, array);

// Int x = array[i];
// x.val == 46 for all 0 <= i < N
```

**Template Parameters**:
* **`DerivedPolicy`** The name of the derived execution policy. 
* **`InputIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>. 
* **`Size`** is an integral type. 
* **`ForwardIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a>, <code>ForwardIterator</code> is mutable, and <code>ForwardIterator's</code><code>value&#95;type</code> has a constructor that takes a single argument whose type is <code>InputIterator's</code><code>value&#95;type</code>.

**Function Parameters**:
* **`exec`** The execution policy to use for parallelization. 
* **`first`** The first element of the input range to copy from. 
* **`n`** The number of elements to copy. 
* **`result`** The first element of the output range to copy to. 

**Preconditions**:
<code>first</code> may equal <code>result</code>, but the range <code>[first, first + n)</code> and the range <code>[result, result + n)</code> shall not overlap otherwise.

**Returns**:
An iterator pointing to the last element of the output range.

**See**:
* <a href="https://en.cppreference.com/w/cpp/memory/uninitialized_copy">https://en.cppreference.com/w/cpp/memory/uninitialized_copy</a>
* <code>uninitialized&#95;copy</code>
* <code>copy</code>
* <code>uninitialized&#95;fill</code>
* <code>device&#95;new</code>
* <code>device&#95;malloc</code>

<h3 id="function-uninitialized-copy-n">
Function <code>thrust::uninitialized&#95;copy&#95;n</code>
</h3>

<code class="doxybook">
<span>template &lt;typename InputIterator,</span>
<span>&nbsp;&nbsp;typename Size,</span>
<span>&nbsp;&nbsp;typename ForwardIterator&gt;</span>
<span>ForwardIterator </span><span><b>uninitialized_copy_n</b>(InputIterator first,</span>
<span>&nbsp;&nbsp;Size n,</span>
<span>&nbsp;&nbsp;ForwardIterator result);</span></code>
In <code>thrust</code>, the function <code>thrust::device&#95;new</code> allocates memory for an object and then creates an object at that location by calling a constructor. Occasionally, however, it is useful to separate those two operations. If each iterator in the range <code>[result, result + n)</code> points to uninitialized memory, then <code>uninitialized&#95;copy&#95;n</code> creates a copy of <code>[first, first + n)</code> in that range. That is, for each iterator <code>i</code> in the input, <code>uninitialized&#95;copy&#95;n</code> creates a copy of <code>&#42;i</code> in the location pointed to by the corresponding iterator in the output range by <code>InputIterator's</code><code>value&#95;type's</code> copy constructor with *i as its argument.


The following code snippet demonstrates how to use <code>uninitialized&#95;copy</code> to initialize a range of uninitialized memory.



```cpp
#include <thrust/uninitialized_copy.h>
#include <thrust/device_malloc.h>
#include <thrust/device_vector.h>

struct Int
{
  __host__ __device__
  Int(int x) : val(x) {}
  int val;
};  
...
const int N = 137;

Int val(46);
thrust::device_vector<Int> input(N, val);
thrust::device_ptr<Int> array = thrust::device_malloc<Int>(N);
thrust::uninitialized_copy_n(input.begin(), N, array);

// Int x = array[i];
// x.val == 46 for all 0 <= i < N
```

**Template Parameters**:
* **`InputIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>. 
* **`Size`** is an integral type. 
* **`ForwardIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a>, <code>ForwardIterator</code> is mutable, and <code>ForwardIterator's</code><code>value&#95;type</code> has a constructor that takes a single argument whose type is <code>InputIterator's</code><code>value&#95;type</code>.

**Function Parameters**:
* **`first`** The first element of the input range to copy from. 
* **`n`** The number of elements to copy. 
* **`result`** The first element of the output range to copy to. 

**Preconditions**:
<code>first</code> may equal <code>result</code>, but the range <code>[first, first + n)</code> and the range <code>[result, result + n)</code> shall not overlap otherwise.

**Returns**:
An iterator pointing to the last element of the output range.

**See**:
* <a href="https://en.cppreference.com/w/cpp/memory/uninitialized_copy">https://en.cppreference.com/w/cpp/memory/uninitialized_copy</a>
* <code>uninitialized&#95;copy</code>
* <code>copy</code>
* <code>uninitialized&#95;fill</code>
* <code>device&#95;new</code>
* <code>device&#95;malloc</code>


