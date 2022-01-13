---
title: Vectorized Searches
parent: Binary Search
grand_parent: Searching
nav_exclude: false
has_children: true
has_toc: false
---

# Vectorized Searches

<code class="doxybook">
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename ForwardIterator,</span>
<span>&nbsp;&nbsp;typename InputIterator,</span>
<span>&nbsp;&nbsp;typename OutputIterator&gt;</span>
<span>__host__ __device__ OutputIterator </span><span><b><a href="{{ site.baseurl }}/api/groups/group__vectorized__binary__search.html#function-lower-bound">thrust::lower&#95;bound</a></b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;ForwardIterator first,</span>
<span>&nbsp;&nbsp;ForwardIterator last,</span>
<span>&nbsp;&nbsp;InputIterator values_first,</span>
<span>&nbsp;&nbsp;InputIterator values_last,</span>
<span>&nbsp;&nbsp;OutputIterator result);</span>
<br>
<span>template &lt;class ForwardIterator,</span>
<span>&nbsp;&nbsp;class InputIterator,</span>
<span>&nbsp;&nbsp;class OutputIterator&gt;</span>
<span>OutputIterator </span><span><b><a href="{{ site.baseurl }}/api/groups/group__vectorized__binary__search.html#function-lower-bound">thrust::lower&#95;bound</a></b>(ForwardIterator first,</span>
<span>&nbsp;&nbsp;ForwardIterator last,</span>
<span>&nbsp;&nbsp;InputIterator values_first,</span>
<span>&nbsp;&nbsp;InputIterator values_last,</span>
<span>&nbsp;&nbsp;OutputIterator result);</span>
<br>
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename ForwardIterator,</span>
<span>&nbsp;&nbsp;typename InputIterator,</span>
<span>&nbsp;&nbsp;typename OutputIterator,</span>
<span>&nbsp;&nbsp;typename StrictWeakOrdering&gt;</span>
<span>__host__ __device__ OutputIterator </span><span><b><a href="{{ site.baseurl }}/api/groups/group__vectorized__binary__search.html#function-lower-bound">thrust::lower&#95;bound</a></b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;ForwardIterator first,</span>
<span>&nbsp;&nbsp;ForwardIterator last,</span>
<span>&nbsp;&nbsp;InputIterator values_first,</span>
<span>&nbsp;&nbsp;InputIterator values_last,</span>
<span>&nbsp;&nbsp;OutputIterator result,</span>
<span>&nbsp;&nbsp;StrictWeakOrdering comp);</span>
<br>
<span>template &lt;class ForwardIterator,</span>
<span>&nbsp;&nbsp;class InputIterator,</span>
<span>&nbsp;&nbsp;class OutputIterator,</span>
<span>&nbsp;&nbsp;class StrictWeakOrdering&gt;</span>
<span>OutputIterator </span><span><b><a href="{{ site.baseurl }}/api/groups/group__vectorized__binary__search.html#function-lower-bound">thrust::lower&#95;bound</a></b>(ForwardIterator first,</span>
<span>&nbsp;&nbsp;ForwardIterator last,</span>
<span>&nbsp;&nbsp;InputIterator values_first,</span>
<span>&nbsp;&nbsp;InputIterator values_last,</span>
<span>&nbsp;&nbsp;OutputIterator result,</span>
<span>&nbsp;&nbsp;StrictWeakOrdering comp);</span>
<br>
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename ForwardIterator,</span>
<span>&nbsp;&nbsp;typename InputIterator,</span>
<span>&nbsp;&nbsp;typename OutputIterator&gt;</span>
<span>__host__ __device__ OutputIterator </span><span><b><a href="{{ site.baseurl }}/api/groups/group__vectorized__binary__search.html#function-upper-bound">thrust::upper&#95;bound</a></b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;ForwardIterator first,</span>
<span>&nbsp;&nbsp;ForwardIterator last,</span>
<span>&nbsp;&nbsp;InputIterator values_first,</span>
<span>&nbsp;&nbsp;InputIterator values_last,</span>
<span>&nbsp;&nbsp;OutputIterator result);</span>
<br>
<span>template &lt;class ForwardIterator,</span>
<span>&nbsp;&nbsp;class InputIterator,</span>
<span>&nbsp;&nbsp;class OutputIterator&gt;</span>
<span>OutputIterator </span><span><b><a href="{{ site.baseurl }}/api/groups/group__vectorized__binary__search.html#function-upper-bound">thrust::upper&#95;bound</a></b>(ForwardIterator first,</span>
<span>&nbsp;&nbsp;ForwardIterator last,</span>
<span>&nbsp;&nbsp;InputIterator values_first,</span>
<span>&nbsp;&nbsp;InputIterator values_last,</span>
<span>&nbsp;&nbsp;OutputIterator result);</span>
<br>
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename ForwardIterator,</span>
<span>&nbsp;&nbsp;typename InputIterator,</span>
<span>&nbsp;&nbsp;typename OutputIterator,</span>
<span>&nbsp;&nbsp;typename StrictWeakOrdering&gt;</span>
<span>__host__ __device__ OutputIterator </span><span><b><a href="{{ site.baseurl }}/api/groups/group__vectorized__binary__search.html#function-upper-bound">thrust::upper&#95;bound</a></b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;ForwardIterator first,</span>
<span>&nbsp;&nbsp;ForwardIterator last,</span>
<span>&nbsp;&nbsp;InputIterator values_first,</span>
<span>&nbsp;&nbsp;InputIterator values_last,</span>
<span>&nbsp;&nbsp;OutputIterator result,</span>
<span>&nbsp;&nbsp;StrictWeakOrdering comp);</span>
<br>
<span>template &lt;class ForwardIterator,</span>
<span>&nbsp;&nbsp;class InputIterator,</span>
<span>&nbsp;&nbsp;class OutputIterator,</span>
<span>&nbsp;&nbsp;class StrictWeakOrdering&gt;</span>
<span>OutputIterator </span><span><b><a href="{{ site.baseurl }}/api/groups/group__vectorized__binary__search.html#function-upper-bound">thrust::upper&#95;bound</a></b>(ForwardIterator first,</span>
<span>&nbsp;&nbsp;ForwardIterator last,</span>
<span>&nbsp;&nbsp;InputIterator values_first,</span>
<span>&nbsp;&nbsp;InputIterator values_last,</span>
<span>&nbsp;&nbsp;OutputIterator result,</span>
<span>&nbsp;&nbsp;StrictWeakOrdering comp);</span>
<br>
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename ForwardIterator,</span>
<span>&nbsp;&nbsp;typename InputIterator,</span>
<span>&nbsp;&nbsp;typename OutputIterator&gt;</span>
<span>__host__ __device__ OutputIterator </span><span><b><a href="{{ site.baseurl }}/api/groups/group__vectorized__binary__search.html#function-binary-search">thrust::binary&#95;search</a></b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;ForwardIterator first,</span>
<span>&nbsp;&nbsp;ForwardIterator last,</span>
<span>&nbsp;&nbsp;InputIterator values_first,</span>
<span>&nbsp;&nbsp;InputIterator values_last,</span>
<span>&nbsp;&nbsp;OutputIterator result);</span>
<br>
<span>template &lt;class ForwardIterator,</span>
<span>&nbsp;&nbsp;class InputIterator,</span>
<span>&nbsp;&nbsp;class OutputIterator&gt;</span>
<span>OutputIterator </span><span><b><a href="{{ site.baseurl }}/api/groups/group__vectorized__binary__search.html#function-binary-search">thrust::binary&#95;search</a></b>(ForwardIterator first,</span>
<span>&nbsp;&nbsp;ForwardIterator last,</span>
<span>&nbsp;&nbsp;InputIterator values_first,</span>
<span>&nbsp;&nbsp;InputIterator values_last,</span>
<span>&nbsp;&nbsp;OutputIterator result);</span>
<br>
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename ForwardIterator,</span>
<span>&nbsp;&nbsp;typename InputIterator,</span>
<span>&nbsp;&nbsp;typename OutputIterator,</span>
<span>&nbsp;&nbsp;typename StrictWeakOrdering&gt;</span>
<span>__host__ __device__ OutputIterator </span><span><b><a href="{{ site.baseurl }}/api/groups/group__vectorized__binary__search.html#function-binary-search">thrust::binary&#95;search</a></b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;ForwardIterator first,</span>
<span>&nbsp;&nbsp;ForwardIterator last,</span>
<span>&nbsp;&nbsp;InputIterator values_first,</span>
<span>&nbsp;&nbsp;InputIterator values_last,</span>
<span>&nbsp;&nbsp;OutputIterator result,</span>
<span>&nbsp;&nbsp;StrictWeakOrdering comp);</span>
<br>
<span>template &lt;class ForwardIterator,</span>
<span>&nbsp;&nbsp;class InputIterator,</span>
<span>&nbsp;&nbsp;class OutputIterator,</span>
<span>&nbsp;&nbsp;class StrictWeakOrdering&gt;</span>
<span>OutputIterator </span><span><b><a href="{{ site.baseurl }}/api/groups/group__vectorized__binary__search.html#function-binary-search">thrust::binary&#95;search</a></b>(ForwardIterator first,</span>
<span>&nbsp;&nbsp;ForwardIterator last,</span>
<span>&nbsp;&nbsp;InputIterator values_first,</span>
<span>&nbsp;&nbsp;InputIterator values_last,</span>
<span>&nbsp;&nbsp;OutputIterator result,</span>
<span>&nbsp;&nbsp;StrictWeakOrdering comp);</span>
</code>

## Functions

<h3 id="function-lower-bound">
Function <code>thrust::lower&#95;bound</code>
</h3>

<code class="doxybook">
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename ForwardIterator,</span>
<span>&nbsp;&nbsp;typename InputIterator,</span>
<span>&nbsp;&nbsp;typename OutputIterator&gt;</span>
<span>__host__ __device__ OutputIterator </span><span><b>lower_bound</b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;ForwardIterator first,</span>
<span>&nbsp;&nbsp;ForwardIterator last,</span>
<span>&nbsp;&nbsp;InputIterator values_first,</span>
<span>&nbsp;&nbsp;InputIterator values_last,</span>
<span>&nbsp;&nbsp;OutputIterator result);</span></code>
<code>lower&#95;bound</code> is a vectorized version of binary search: for each iterator <code>v</code> in <code>[values&#95;first, values&#95;last)</code> it attempts to find the value <code>&#42;v</code> in an ordered range <code>[first, last)</code>. Specifically, it returns the index of first position where value could be inserted without violating the ordering.

The algorithm's execution is parallelized as determined by <code>exec</code>.


The following code snippet demonstrates how to use <code>lower&#95;bound</code> to search for multiple values in a ordered range using the <code>thrust::device</code> execution policy for parallelization:



```cpp
#include <thrust/binary_search.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
...
thrust::device_vector<int> input(5);

input[0] = 0;
input[1] = 2;
input[2] = 5;
input[3] = 7;
input[4] = 8;

thrust::device_vector<int> values(6);
values[0] = 0; 
values[1] = 1;
values[2] = 2;
values[3] = 3;
values[4] = 8;
values[5] = 9;

thrust::device_vector<unsigned int> output(6);

thrust::lower_bound(thrust::device,
                    input.begin(), input.end(),
                    values.begin(), values.end(),
                    output.begin());

// output is now [0, 1, 1, 2, 4, 5]
```

**Template Parameters**:
* **`DerivedPolicy`** The name of the derived execution policy. 
* **`ForwardIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a>. 
* **`InputIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>. and <code>InputIterator's</code><code>value&#95;type</code> is <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThanComparable</a>. 
* **`OutputIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>. and <code>ForwardIterator's</code> difference_type is convertible to <code>OutputIterator's</code><code>value&#95;type</code>.

**Function Parameters**:
* **`exec`** The execution policy to use for parallelization. 
* **`first`** The beginning of the ordered sequence. 
* **`last`** The end of the ordered sequence. 
* **`values_first`** The beginning of the search values sequence. 
* **`values_last`** The end of the search values sequence. 
* **`result`** The beginning of the output sequence.

**Preconditions**:
The ranges <code>[first,last)</code> and <code>[result, result + (last - first))</code> shall not overlap.

**See**:
* <a href="https://en.cppreference.com/w/cpp/algorithm/lower_bound">https://en.cppreference.com/w/cpp/algorithm/lower_bound</a>
* <code>upper&#95;bound</code>
* <code>equal&#95;range</code>
* <code><a href="{{ site.baseurl }}/api/groups/group__binary__search.html">Binary Search</a></code>

<h3 id="function-lower-bound">
Function <code>thrust::lower&#95;bound</code>
</h3>

<code class="doxybook">
<span>template &lt;class ForwardIterator,</span>
<span>&nbsp;&nbsp;class InputIterator,</span>
<span>&nbsp;&nbsp;class OutputIterator&gt;</span>
<span>OutputIterator </span><span><b>lower_bound</b>(ForwardIterator first,</span>
<span>&nbsp;&nbsp;ForwardIterator last,</span>
<span>&nbsp;&nbsp;InputIterator values_first,</span>
<span>&nbsp;&nbsp;InputIterator values_last,</span>
<span>&nbsp;&nbsp;OutputIterator result);</span></code>
<code>lower&#95;bound</code> is a vectorized version of binary search: for each iterator <code>v</code> in <code>[values&#95;first, values&#95;last)</code> it attempts to find the value <code>&#42;v</code> in an ordered range <code>[first, last)</code>. Specifically, it returns the index of first position where value could be inserted without violating the ordering.


The following code snippet demonstrates how to use <code>lower&#95;bound</code> to search for multiple values in a ordered range.



```cpp
#include <thrust/binary_search.h>
#include <thrust/device_vector.h>
...
thrust::device_vector<int> input(5);

input[0] = 0;
input[1] = 2;
input[2] = 5;
input[3] = 7;
input[4] = 8;

thrust::device_vector<int> values(6);
values[0] = 0; 
values[1] = 1;
values[2] = 2;
values[3] = 3;
values[4] = 8;
values[5] = 9;

thrust::device_vector<unsigned int> output(6);

thrust::lower_bound(input.begin(), input.end(),
                    values.begin(), values.end(),
                    output.begin());

// output is now [0, 1, 1, 2, 4, 5]
```

**Template Parameters**:
* **`ForwardIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a>. 
* **`InputIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>. and <code>InputIterator's</code><code>value&#95;type</code> is <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThanComparable</a>. 
* **`OutputIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>. and <code>ForwardIterator's</code> difference_type is convertible to <code>OutputIterator's</code><code>value&#95;type</code>.

**Function Parameters**:
* **`first`** The beginning of the ordered sequence. 
* **`last`** The end of the ordered sequence. 
* **`values_first`** The beginning of the search values sequence. 
* **`values_last`** The end of the search values sequence. 
* **`result`** The beginning of the output sequence.

**Preconditions**:
The ranges <code>[first,last)</code> and <code>[result, result + (last - first))</code> shall not overlap.

**See**:
* <a href="https://en.cppreference.com/w/cpp/algorithm/lower_bound">https://en.cppreference.com/w/cpp/algorithm/lower_bound</a>
* <code>upper&#95;bound</code>
* <code>equal&#95;range</code>
* <code><a href="{{ site.baseurl }}/api/groups/group__binary__search.html">Binary Search</a></code>

<h3 id="function-lower-bound">
Function <code>thrust::lower&#95;bound</code>
</h3>

<code class="doxybook">
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename ForwardIterator,</span>
<span>&nbsp;&nbsp;typename InputIterator,</span>
<span>&nbsp;&nbsp;typename OutputIterator,</span>
<span>&nbsp;&nbsp;typename StrictWeakOrdering&gt;</span>
<span>__host__ __device__ OutputIterator </span><span><b>lower_bound</b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;ForwardIterator first,</span>
<span>&nbsp;&nbsp;ForwardIterator last,</span>
<span>&nbsp;&nbsp;InputIterator values_first,</span>
<span>&nbsp;&nbsp;InputIterator values_last,</span>
<span>&nbsp;&nbsp;OutputIterator result,</span>
<span>&nbsp;&nbsp;StrictWeakOrdering comp);</span></code>
<code>lower&#95;bound</code> is a vectorized version of binary search: for each iterator <code>v</code> in <code>[values&#95;first, values&#95;last)</code> it attempts to find the value <code>&#42;v</code> in an ordered range <code>[first, last)</code>. Specifically, it returns the index of first position where value could be inserted without violating the ordering. This version of <code>lower&#95;bound</code> uses function object <code>comp</code> for comparison.

The algorithm's execution is parallelized as determined by <code>exec</code>.


The following code snippet demonstrates how to use <code>lower&#95;bound</code> to search for multiple values in a ordered range.



```cpp
#include <thrust/binary_search.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>
...
thrust::device_vector<int> input(5);

input[0] = 0;
input[1] = 2;
input[2] = 5;
input[3] = 7;
input[4] = 8;

thrust::device_vector<int> values(6);
values[0] = 0; 
values[1] = 1;
values[2] = 2;
values[3] = 3;
values[4] = 8;
values[5] = 9;

thrust::device_vector<unsigned int> output(6);

thrust::lower_bound(input.begin(), input.end(),
                    values.begin(), values.end(), 
                    output.begin(),
                    thrust::less<int>());

// output is now [0, 1, 1, 2, 4, 5]
```

**Template Parameters**:
* **`DerivedPolicy`** The name of the derived execution policy. 
* **`ForwardIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a>. 
* **`InputIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>. and <code>InputIterator's</code><code>value&#95;type</code> is comparable to <code>ForwardIterator's</code><code>value&#95;type</code>. 
* **`OutputIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>. and <code>ForwardIterator's</code> difference_type is convertible to <code>OutputIterator's</code><code>value&#95;type</code>. 
* **`StrictWeakOrdering`** is a model of <a href="https://en.cppreference.com/w/cpp/concepts/strict_weak_order">Strict Weak Ordering</a>.

**Function Parameters**:
* **`exec`** The execution policy to use for parallelization. 
* **`first`** The beginning of the ordered sequence. 
* **`last`** The end of the ordered sequence. 
* **`values_first`** The beginning of the search values sequence. 
* **`values_last`** The end of the search values sequence. 
* **`result`** The beginning of the output sequence. 
* **`comp`** The comparison operator.

**Preconditions**:
The ranges <code>[first,last)</code> and <code>[result, result + (last - first))</code> shall not overlap.

**See**:
* <a href="https://en.cppreference.com/w/cpp/algorithm/lower_bound">https://en.cppreference.com/w/cpp/algorithm/lower_bound</a>
* <code>upper&#95;bound</code>
* <code>equal&#95;range</code>
* <code><a href="{{ site.baseurl }}/api/groups/group__binary__search.html">Binary Search</a></code>

<h3 id="function-lower-bound">
Function <code>thrust::lower&#95;bound</code>
</h3>

<code class="doxybook">
<span>template &lt;class ForwardIterator,</span>
<span>&nbsp;&nbsp;class InputIterator,</span>
<span>&nbsp;&nbsp;class OutputIterator,</span>
<span>&nbsp;&nbsp;class StrictWeakOrdering&gt;</span>
<span>OutputIterator </span><span><b>lower_bound</b>(ForwardIterator first,</span>
<span>&nbsp;&nbsp;ForwardIterator last,</span>
<span>&nbsp;&nbsp;InputIterator values_first,</span>
<span>&nbsp;&nbsp;InputIterator values_last,</span>
<span>&nbsp;&nbsp;OutputIterator result,</span>
<span>&nbsp;&nbsp;StrictWeakOrdering comp);</span></code>
<code>lower&#95;bound</code> is a vectorized version of binary search: for each iterator <code>v</code> in <code>[values&#95;first, values&#95;last)</code> it attempts to find the value <code>&#42;v</code> in an ordered range <code>[first, last)</code>. Specifically, it returns the index of first position where value could be inserted without violating the ordering. This version of <code>lower&#95;bound</code> uses function object <code>comp</code> for comparison.


The following code snippet demonstrates how to use <code>lower&#95;bound</code> to search for multiple values in a ordered range.



```cpp
#include <thrust/binary_search.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
...
thrust::device_vector<int> input(5);

input[0] = 0;
input[1] = 2;
input[2] = 5;
input[3] = 7;
input[4] = 8;

thrust::device_vector<int> values(6);
values[0] = 0; 
values[1] = 1;
values[2] = 2;
values[3] = 3;
values[4] = 8;
values[5] = 9;

thrust::device_vector<unsigned int> output(6);

thrust::lower_bound(input.begin(), input.end(),
                    values.begin(), values.end(), 
                    output.begin(),
                    thrust::less<int>());

// output is now [0, 1, 1, 2, 4, 5]
```

**Template Parameters**:
* **`ForwardIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a>. 
* **`InputIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>. and <code>InputIterator's</code><code>value&#95;type</code> is comparable to <code>ForwardIterator's</code><code>value&#95;type</code>. 
* **`OutputIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>. and <code>ForwardIterator's</code> difference_type is convertible to <code>OutputIterator's</code><code>value&#95;type</code>. 
* **`StrictWeakOrdering`** is a model of <a href="https://en.cppreference.com/w/cpp/concepts/strict_weak_order">Strict Weak Ordering</a>.

**Function Parameters**:
* **`first`** The beginning of the ordered sequence. 
* **`last`** The end of the ordered sequence. 
* **`values_first`** The beginning of the search values sequence. 
* **`values_last`** The end of the search values sequence. 
* **`result`** The beginning of the output sequence. 
* **`comp`** The comparison operator.

**Preconditions**:
The ranges <code>[first,last)</code> and <code>[result, result + (last - first))</code> shall not overlap.

**See**:
* <a href="https://en.cppreference.com/w/cpp/algorithm/lower_bound">https://en.cppreference.com/w/cpp/algorithm/lower_bound</a>
* <code>upper&#95;bound</code>
* <code>equal&#95;range</code>
* <code><a href="{{ site.baseurl }}/api/groups/group__binary__search.html">Binary Search</a></code>

<h3 id="function-upper-bound">
Function <code>thrust::upper&#95;bound</code>
</h3>

<code class="doxybook">
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename ForwardIterator,</span>
<span>&nbsp;&nbsp;typename InputIterator,</span>
<span>&nbsp;&nbsp;typename OutputIterator&gt;</span>
<span>__host__ __device__ OutputIterator </span><span><b>upper_bound</b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;ForwardIterator first,</span>
<span>&nbsp;&nbsp;ForwardIterator last,</span>
<span>&nbsp;&nbsp;InputIterator values_first,</span>
<span>&nbsp;&nbsp;InputIterator values_last,</span>
<span>&nbsp;&nbsp;OutputIterator result);</span></code>
<code>upper&#95;bound</code> is a vectorized version of binary search: for each iterator <code>v</code> in <code>[values&#95;first, values&#95;last)</code> it attempts to find the value <code>&#42;v</code> in an ordered range <code>[first, last)</code>. Specifically, it returns the index of last position where value could be inserted without violating the ordering.

The algorithm's execution is parallelized as determined by <code>exec</code>.


The following code snippet demonstrates how to use <code>upper&#95;bound</code> to search for multiple values in a ordered range using the <code>thrust::device</code> execution policy for parallelization:



```cpp
#include <thrust/binary_search.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
...
thrust::device_vector<int> input(5);

input[0] = 0;
input[1] = 2;
input[2] = 5;
input[3] = 7;
input[4] = 8;

thrust::device_vector<int> values(6);
values[0] = 0; 
values[1] = 1;
values[2] = 2;
values[3] = 3;
values[4] = 8;
values[5] = 9;

thrust::device_vector<unsigned int> output(6);

thrust::upper_bound(thrust::device,
                    input.begin(), input.end(),
                    values.begin(), values.end(),
                    output.begin());

// output is now [1, 1, 2, 2, 5, 5]
```

**Template Parameters**:
* **`DerivedPolicy`** The name of the derived execution policy. 
* **`ForwardIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a>. 
* **`InputIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>. and <code>InputIterator's</code><code>value&#95;type</code> is <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThanComparable</a>. 
* **`OutputIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>. and <code>ForwardIterator's</code> difference_type is convertible to <code>OutputIterator's</code><code>value&#95;type</code>.

**Function Parameters**:
* **`exec`** The execution policy to use for parallelization. 
* **`first`** The beginning of the ordered sequence. 
* **`last`** The end of the ordered sequence. 
* **`values_first`** The beginning of the search values sequence. 
* **`values_last`** The end of the search values sequence. 
* **`result`** The beginning of the output sequence.

**Preconditions**:
The ranges <code>[first,last)</code> and <code>[result, result + (last - first))</code> shall not overlap.

**See**:
* <a href="https://en.cppreference.com/w/cpp/algorithm/upper_bound">https://en.cppreference.com/w/cpp/algorithm/upper_bound</a>
* <code>upper&#95;bound</code>
* <code>equal&#95;range</code>
* <code><a href="{{ site.baseurl }}/api/groups/group__binary__search.html">Binary Search</a></code>

<h3 id="function-upper-bound">
Function <code>thrust::upper&#95;bound</code>
</h3>

<code class="doxybook">
<span>template &lt;class ForwardIterator,</span>
<span>&nbsp;&nbsp;class InputIterator,</span>
<span>&nbsp;&nbsp;class OutputIterator&gt;</span>
<span>OutputIterator </span><span><b>upper_bound</b>(ForwardIterator first,</span>
<span>&nbsp;&nbsp;ForwardIterator last,</span>
<span>&nbsp;&nbsp;InputIterator values_first,</span>
<span>&nbsp;&nbsp;InputIterator values_last,</span>
<span>&nbsp;&nbsp;OutputIterator result);</span></code>
<code>upper&#95;bound</code> is a vectorized version of binary search: for each iterator <code>v</code> in <code>[values&#95;first, values&#95;last)</code> it attempts to find the value <code>&#42;v</code> in an ordered range <code>[first, last)</code>. Specifically, it returns the index of last position where value could be inserted without violating the ordering.


The following code snippet demonstrates how to use <code>upper&#95;bound</code> to search for multiple values in a ordered range.



```cpp
#include <thrust/binary_search.h>
#include <thrust/device_vector.h>
...
thrust::device_vector<int> input(5);

input[0] = 0;
input[1] = 2;
input[2] = 5;
input[3] = 7;
input[4] = 8;

thrust::device_vector<int> values(6);
values[0] = 0; 
values[1] = 1;
values[2] = 2;
values[3] = 3;
values[4] = 8;
values[5] = 9;

thrust::device_vector<unsigned int> output(6);

thrust::upper_bound(input.begin(), input.end(),
                    values.begin(), values.end(),
                    output.begin());

// output is now [1, 1, 2, 2, 5, 5]
```

**Template Parameters**:
* **`ForwardIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a>. 
* **`InputIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>. and <code>InputIterator's</code><code>value&#95;type</code> is <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThanComparable</a>. 
* **`OutputIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>. and <code>ForwardIterator's</code> difference_type is convertible to <code>OutputIterator's</code><code>value&#95;type</code>.

**Function Parameters**:
* **`first`** The beginning of the ordered sequence. 
* **`last`** The end of the ordered sequence. 
* **`values_first`** The beginning of the search values sequence. 
* **`values_last`** The end of the search values sequence. 
* **`result`** The beginning of the output sequence.

**Preconditions**:
The ranges <code>[first,last)</code> and <code>[result, result + (last - first))</code> shall not overlap.

**See**:
* <a href="https://en.cppreference.com/w/cpp/algorithm/upper_bound">https://en.cppreference.com/w/cpp/algorithm/upper_bound</a>
* <code>upper&#95;bound</code>
* <code>equal&#95;range</code>
* <code><a href="{{ site.baseurl }}/api/groups/group__binary__search.html">Binary Search</a></code>

<h3 id="function-upper-bound">
Function <code>thrust::upper&#95;bound</code>
</h3>

<code class="doxybook">
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename ForwardIterator,</span>
<span>&nbsp;&nbsp;typename InputIterator,</span>
<span>&nbsp;&nbsp;typename OutputIterator,</span>
<span>&nbsp;&nbsp;typename StrictWeakOrdering&gt;</span>
<span>__host__ __device__ OutputIterator </span><span><b>upper_bound</b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;ForwardIterator first,</span>
<span>&nbsp;&nbsp;ForwardIterator last,</span>
<span>&nbsp;&nbsp;InputIterator values_first,</span>
<span>&nbsp;&nbsp;InputIterator values_last,</span>
<span>&nbsp;&nbsp;OutputIterator result,</span>
<span>&nbsp;&nbsp;StrictWeakOrdering comp);</span></code>
<code>upper&#95;bound</code> is a vectorized version of binary search: for each iterator <code>v</code> in <code>[values&#95;first, values&#95;last)</code> it attempts to find the value <code>&#42;v</code> in an ordered range <code>[first, last)</code>. Specifically, it returns the index of first position where value could be inserted without violating the ordering. This version of <code>upper&#95;bound</code> uses function object <code>comp</code> for comparison.

The algorithm's execution is parallelized as determined by <code>exec</code>.


The following code snippet demonstrates how to use <code>upper&#95;bound</code> to search for multiple values in a ordered range using the <code>thrust::device</code> execution policy for parallelization:



```cpp
#include <thrust/binary_search.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>
...
thrust::device_vector<int> input(5);

input[0] = 0;
input[1] = 2;
input[2] = 5;
input[3] = 7;
input[4] = 8;

thrust::device_vector<int> values(6);
values[0] = 0; 
values[1] = 1;
values[2] = 2;
values[3] = 3;
values[4] = 8;
values[5] = 9;

thrust::device_vector<unsigned int> output(6);

thrust::upper_bound(thrust::device,
                    input.begin(), input.end(),
                    values.begin(), values.end(), 
                    output.begin(),
                    thrust::less<int>());

// output is now [1, 1, 2, 2, 5, 5]
```

**Template Parameters**:
* **`DerivedPolicy`** The name of the derived execution policy. 
* **`ForwardIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a>. 
* **`InputIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>. and <code>InputIterator's</code><code>value&#95;type</code> is comparable to <code>ForwardIterator's</code><code>value&#95;type</code>. 
* **`OutputIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>. and <code>ForwardIterator's</code> difference_type is convertible to <code>OutputIterator's</code><code>value&#95;type</code>. 
* **`StrictWeakOrdering`** is a model of <a href="https://en.cppreference.com/w/cpp/concepts/strict_weak_order">Strict Weak Ordering</a>.

**Function Parameters**:
* **`exec`** The execution policy to use for parallelization. 
* **`first`** The beginning of the ordered sequence. 
* **`last`** The end of the ordered sequence. 
* **`values_first`** The beginning of the search values sequence. 
* **`values_last`** The end of the search values sequence. 
* **`result`** The beginning of the output sequence. 
* **`comp`** The comparison operator.

**Preconditions**:
The ranges <code>[first,last)</code> and <code>[result, result + (last - first))</code> shall not overlap.

**See**:
* <a href="https://en.cppreference.com/w/cpp/algorithm/upper_bound">https://en.cppreference.com/w/cpp/algorithm/upper_bound</a>
* <code>lower&#95;bound</code>
* <code>equal&#95;range</code>
* <code><a href="{{ site.baseurl }}/api/groups/group__binary__search.html">Binary Search</a></code>

<h3 id="function-upper-bound">
Function <code>thrust::upper&#95;bound</code>
</h3>

<code class="doxybook">
<span>template &lt;class ForwardIterator,</span>
<span>&nbsp;&nbsp;class InputIterator,</span>
<span>&nbsp;&nbsp;class OutputIterator,</span>
<span>&nbsp;&nbsp;class StrictWeakOrdering&gt;</span>
<span>OutputIterator </span><span><b>upper_bound</b>(ForwardIterator first,</span>
<span>&nbsp;&nbsp;ForwardIterator last,</span>
<span>&nbsp;&nbsp;InputIterator values_first,</span>
<span>&nbsp;&nbsp;InputIterator values_last,</span>
<span>&nbsp;&nbsp;OutputIterator result,</span>
<span>&nbsp;&nbsp;StrictWeakOrdering comp);</span></code>
<code>upper&#95;bound</code> is a vectorized version of binary search: for each iterator <code>v</code> in <code>[values&#95;first, values&#95;last)</code> it attempts to find the value <code>&#42;v</code> in an ordered range <code>[first, last)</code>. Specifically, it returns the index of first position where value could be inserted without violating the ordering. This version of <code>upper&#95;bound</code> uses function object <code>comp</code> for comparison.


The following code snippet demonstrates how to use <code>upper&#95;bound</code> to search for multiple values in a ordered range.



```cpp
#include <thrust/binary_search.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
...
thrust::device_vector<int> input(5);

input[0] = 0;
input[1] = 2;
input[2] = 5;
input[3] = 7;
input[4] = 8;

thrust::device_vector<int> values(6);
values[0] = 0; 
values[1] = 1;
values[2] = 2;
values[3] = 3;
values[4] = 8;
values[5] = 9;

thrust::device_vector<unsigned int> output(6);

thrust::upper_bound(input.begin(), input.end(),
                    values.begin(), values.end(), 
                    output.begin(),
                    thrust::less<int>());

// output is now [1, 1, 2, 2, 5, 5]
```

**Template Parameters**:
* **`ForwardIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a>. 
* **`InputIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>. and <code>InputIterator's</code><code>value&#95;type</code> is comparable to <code>ForwardIterator's</code><code>value&#95;type</code>. 
* **`OutputIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>. and <code>ForwardIterator's</code> difference_type is convertible to <code>OutputIterator's</code><code>value&#95;type</code>. 
* **`StrictWeakOrdering`** is a model of <a href="https://en.cppreference.com/w/cpp/concepts/strict_weak_order">Strict Weak Ordering</a>.

**Function Parameters**:
* **`first`** The beginning of the ordered sequence. 
* **`last`** The end of the ordered sequence. 
* **`values_first`** The beginning of the search values sequence. 
* **`values_last`** The end of the search values sequence. 
* **`result`** The beginning of the output sequence. 
* **`comp`** The comparison operator.

**Preconditions**:
The ranges <code>[first,last)</code> and <code>[result, result + (last - first))</code> shall not overlap.

**See**:
* <a href="https://en.cppreference.com/w/cpp/algorithm/upper_bound">https://en.cppreference.com/w/cpp/algorithm/upper_bound</a>
* <code>lower&#95;bound</code>
* <code>equal&#95;range</code>
* <code><a href="{{ site.baseurl }}/api/groups/group__binary__search.html">Binary Search</a></code>

<h3 id="function-binary-search">
Function <code>thrust::binary&#95;search</code>
</h3>

<code class="doxybook">
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename ForwardIterator,</span>
<span>&nbsp;&nbsp;typename InputIterator,</span>
<span>&nbsp;&nbsp;typename OutputIterator&gt;</span>
<span>__host__ __device__ OutputIterator </span><span><b>binary_search</b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;ForwardIterator first,</span>
<span>&nbsp;&nbsp;ForwardIterator last,</span>
<span>&nbsp;&nbsp;InputIterator values_first,</span>
<span>&nbsp;&nbsp;InputIterator values_last,</span>
<span>&nbsp;&nbsp;OutputIterator result);</span></code>
<code>binary&#95;search</code> is a vectorized version of binary search: for each iterator <code>v</code> in <code>[values&#95;first, values&#95;last)</code> it attempts to find the value <code>&#42;v</code> in an ordered range <code>[first, last)</code>. It returns <code>true</code> if an element that is equivalent to <code>value</code> is present in <code>[first, last)</code> and <code>false</code> if no such element exists.

The algorithm's execution is parallelized as determined by <code>exec</code>.


The following code snippet demonstrates how to use <code>binary&#95;search</code> to search for multiple values in a ordered range using the <code>thrust::device</code> execution policy for parallelization:



```cpp
#include <thrust/binary_search.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
...
thrust::device_vector<int> input(5);

input[0] = 0;
input[1] = 2;
input[2] = 5;
input[3] = 7;
input[4] = 8;

thrust::device_vector<int> values(6);
values[0] = 0; 
values[1] = 1;
values[2] = 2;
values[3] = 3;
values[4] = 8;
values[5] = 9;

thrust::device_vector<bool> output(6);

thrust::binary_search(thrust::device,
                      input.begin(), input.end(),
                      values.begin(), values.end(),
                      output.begin());

// output is now [true, false, true, false, true, false]
```

**Template Parameters**:
* **`DerivedPolicy`** The name of the derived execution policy. 
* **`ForwardIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a>. 
* **`InputIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>. and <code>InputIterator's</code><code>value&#95;type</code> is <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThanComparable</a>. 
* **`OutputIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>. and bool is convertible to <code>OutputIterator's</code><code>value&#95;type</code>.

**Function Parameters**:
* **`exec`** The execution policy to use for parallelization. 
* **`first`** The beginning of the ordered sequence. 
* **`last`** The end of the ordered sequence. 
* **`values_first`** The beginning of the search values sequence. 
* **`values_last`** The end of the search values sequence. 
* **`result`** The beginning of the output sequence.

**Preconditions**:
The ranges <code>[first,last)</code> and <code>[result, result + (last - first))</code> shall not overlap.

**See**:
* <a href="https://en.cppreference.com/w/cpp/algorithm/binary_search">https://en.cppreference.com/w/cpp/algorithm/binary_search</a>
* <code>lower&#95;bound</code>
* <code>upper&#95;bound</code>
* <code>equal&#95;range</code>

<h3 id="function-binary-search">
Function <code>thrust::binary&#95;search</code>
</h3>

<code class="doxybook">
<span>template &lt;class ForwardIterator,</span>
<span>&nbsp;&nbsp;class InputIterator,</span>
<span>&nbsp;&nbsp;class OutputIterator&gt;</span>
<span>OutputIterator </span><span><b>binary_search</b>(ForwardIterator first,</span>
<span>&nbsp;&nbsp;ForwardIterator last,</span>
<span>&nbsp;&nbsp;InputIterator values_first,</span>
<span>&nbsp;&nbsp;InputIterator values_last,</span>
<span>&nbsp;&nbsp;OutputIterator result);</span></code>
<code>binary&#95;search</code> is a vectorized version of binary search: for each iterator <code>v</code> in <code>[values&#95;first, values&#95;last)</code> it attempts to find the value <code>&#42;v</code> in an ordered range <code>[first, last)</code>. It returns <code>true</code> if an element that is equivalent to <code>value</code> is present in <code>[first, last)</code> and <code>false</code> if no such element exists.


The following code snippet demonstrates how to use <code>binary&#95;search</code> to search for multiple values in a ordered range.



```cpp
#include <thrust/binary_search.h>
#include <thrust/device_vector.h>
...
thrust::device_vector<int> input(5);

input[0] = 0;
input[1] = 2;
input[2] = 5;
input[3] = 7;
input[4] = 8;

thrust::device_vector<int> values(6);
values[0] = 0; 
values[1] = 1;
values[2] = 2;
values[3] = 3;
values[4] = 8;
values[5] = 9;

thrust::device_vector<bool> output(6);

thrust::binary_search(input.begin(), input.end(),
                      values.begin(), values.end(),
                      output.begin());

// output is now [true, false, true, false, true, false]
```

**Template Parameters**:
* **`ForwardIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a>. 
* **`InputIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>. and <code>InputIterator's</code><code>value&#95;type</code> is <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThanComparable</a>. 
* **`OutputIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>. and bool is convertible to <code>OutputIterator's</code><code>value&#95;type</code>.

**Function Parameters**:
* **`first`** The beginning of the ordered sequence. 
* **`last`** The end of the ordered sequence. 
* **`values_first`** The beginning of the search values sequence. 
* **`values_last`** The end of the search values sequence. 
* **`result`** The beginning of the output sequence.

**Preconditions**:
The ranges <code>[first,last)</code> and <code>[result, result + (last - first))</code> shall not overlap.

**See**:
* <a href="https://en.cppreference.com/w/cpp/algorithm/binary_search">https://en.cppreference.com/w/cpp/algorithm/binary_search</a>
* <code>lower&#95;bound</code>
* <code>upper&#95;bound</code>
* <code>equal&#95;range</code>

<h3 id="function-binary-search">
Function <code>thrust::binary&#95;search</code>
</h3>

<code class="doxybook">
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename ForwardIterator,</span>
<span>&nbsp;&nbsp;typename InputIterator,</span>
<span>&nbsp;&nbsp;typename OutputIterator,</span>
<span>&nbsp;&nbsp;typename StrictWeakOrdering&gt;</span>
<span>__host__ __device__ OutputIterator </span><span><b>binary_search</b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;ForwardIterator first,</span>
<span>&nbsp;&nbsp;ForwardIterator last,</span>
<span>&nbsp;&nbsp;InputIterator values_first,</span>
<span>&nbsp;&nbsp;InputIterator values_last,</span>
<span>&nbsp;&nbsp;OutputIterator result,</span>
<span>&nbsp;&nbsp;StrictWeakOrdering comp);</span></code>
<code>binary&#95;search</code> is a vectorized version of binary search: for each iterator <code>v</code> in <code>[values&#95;first, values&#95;last)</code> it attempts to find the value <code>&#42;v</code> in an ordered range <code>[first, last)</code>. It returns <code>true</code> if an element that is equivalent to <code>value</code> is present in <code>[first, last)</code> and <code>false</code> if no such element exists. This version of <code>binary&#95;search</code> uses function object <code>comp</code> for comparison.

The algorithm's execution is parallelized as determined by <code>exec</code>.


The following code snippet demonstrates how to use <code>binary&#95;search</code> to search for multiple values in a ordered range using the <code>thrust::device</code> execution policy for parallelization:



```cpp
#include <thrust/binary_search.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>
...
thrust::device_vector<int> input(5);

input[0] = 0;
input[1] = 2;
input[2] = 5;
input[3] = 7;
input[4] = 8;

thrust::device_vector<int> values(6);
values[0] = 0; 
values[1] = 1;
values[2] = 2;
values[3] = 3;
values[4] = 8;
values[5] = 9;

thrust::device_vector<bool> output(6);

thrust::binary_search(thrust::device,
                      input.begin(), input.end(),
                      values.begin(), values.end(),
                      output.begin(),
                      thrust::less<T>());

// output is now [true, false, true, false, true, false]
```

**Template Parameters**:
* **`DerivedPolicy`** The name of the derived execution policy. 
* **`ForwardIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a>. 
* **`InputIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>. and <code>InputIterator's</code><code>value&#95;type</code> is <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThanComparable</a>. 
* **`OutputIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>. and bool is convertible to <code>OutputIterator's</code><code>value&#95;type</code>. 
* **`StrictWeakOrdering`** is a model of <a href="https://en.cppreference.com/w/cpp/concepts/strict_weak_order">Strict Weak Ordering</a>.

**Function Parameters**:
* **`exec`** The execution policy to use for parallelization. 
* **`first`** The beginning of the ordered sequence. 
* **`last`** The end of the ordered sequence. 
* **`values_first`** The beginning of the search values sequence. 
* **`values_last`** The end of the search values sequence. 
* **`result`** The beginning of the output sequence. 
* **`comp`** The comparison operator.

**Preconditions**:
The ranges <code>[first,last)</code> and <code>[result, result + (last - first))</code> shall not overlap.

**See**:
* <a href="https://en.cppreference.com/w/cpp/algorithm/binary_search">https://en.cppreference.com/w/cpp/algorithm/binary_search</a>
* <code>lower&#95;bound</code>
* <code>upper&#95;bound</code>
* <code>equal&#95;range</code>

<h3 id="function-binary-search">
Function <code>thrust::binary&#95;search</code>
</h3>

<code class="doxybook">
<span>template &lt;class ForwardIterator,</span>
<span>&nbsp;&nbsp;class InputIterator,</span>
<span>&nbsp;&nbsp;class OutputIterator,</span>
<span>&nbsp;&nbsp;class StrictWeakOrdering&gt;</span>
<span>OutputIterator </span><span><b>binary_search</b>(ForwardIterator first,</span>
<span>&nbsp;&nbsp;ForwardIterator last,</span>
<span>&nbsp;&nbsp;InputIterator values_first,</span>
<span>&nbsp;&nbsp;InputIterator values_last,</span>
<span>&nbsp;&nbsp;OutputIterator result,</span>
<span>&nbsp;&nbsp;StrictWeakOrdering comp);</span></code>
<code>binary&#95;search</code> is a vectorized version of binary search: for each iterator <code>v</code> in <code>[values&#95;first, values&#95;last)</code> it attempts to find the value <code>&#42;v</code> in an ordered range <code>[first, last)</code>. It returns <code>true</code> if an element that is equivalent to <code>value</code> is present in <code>[first, last)</code> and <code>false</code> if no such element exists. This version of <code>binary&#95;search</code> uses function object <code>comp</code> for comparison.


The following code snippet demonstrates how to use <code>binary&#95;search</code> to search for multiple values in a ordered range.



```cpp
#include <thrust/binary_search.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
...
thrust::device_vector<int> input(5);

input[0] = 0;
input[1] = 2;
input[2] = 5;
input[3] = 7;
input[4] = 8;

thrust::device_vector<int> values(6);
values[0] = 0; 
values[1] = 1;
values[2] = 2;
values[3] = 3;
values[4] = 8;
values[5] = 9;

thrust::device_vector<bool> output(6);

thrust::binary_search(input.begin(), input.end(),
                      values.begin(), values.end(),
                      output.begin(),
                      thrust::less<T>());

// output is now [true, false, true, false, true, false]
```

**Template Parameters**:
* **`ForwardIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a>. 
* **`InputIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>. and <code>InputIterator's</code><code>value&#95;type</code> is <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThanComparable</a>. 
* **`OutputIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>. and bool is convertible to <code>OutputIterator's</code><code>value&#95;type</code>. 
* **`StrictWeakOrdering`** is a model of <a href="https://en.cppreference.com/w/cpp/concepts/strict_weak_order">Strict Weak Ordering</a>.

**Function Parameters**:
* **`first`** The beginning of the ordered sequence. 
* **`last`** The end of the ordered sequence. 
* **`values_first`** The beginning of the search values sequence. 
* **`values_last`** The end of the search values sequence. 
* **`result`** The beginning of the output sequence. 
* **`comp`** The comparison operator.

**Preconditions**:
The ranges <code>[first,last)</code> and <code>[result, result + (last - first))</code> shall not overlap.

**See**:
* <a href="https://en.cppreference.com/w/cpp/algorithm/binary_search">https://en.cppreference.com/w/cpp/algorithm/binary_search</a>
* <code>lower&#95;bound</code>
* <code>upper&#95;bound</code>
* <code>equal&#95;range</code>


