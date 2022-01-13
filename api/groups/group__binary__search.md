---
title: Binary Search
parent: Searching
grand_parent: Algorithms
nav_exclude: false
has_children: true
has_toc: false
---

# Binary Search

## Groups

* **[Vectorized Searches]({{ site.baseurl }}/api/groups/group__vectorized__binary__search.html)**

<code class="doxybook">
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename ForwardIterator,</span>
<span>&nbsp;&nbsp;typename LessThanComparable&gt;</span>
<span>__host__ __device__ ForwardIterator </span><span><b><a href="{{ site.baseurl }}/api/groups/group__binary__search.html#function-lower-bound">thrust::lower&#95;bound</a></b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;ForwardIterator first,</span>
<span>&nbsp;&nbsp;ForwardIterator last,</span>
<span>&nbsp;&nbsp;const LessThanComparable & value);</span>
<br>
<span>template &lt;class ForwardIterator,</span>
<span>&nbsp;&nbsp;class LessThanComparable&gt;</span>
<span>ForwardIterator </span><span><b><a href="{{ site.baseurl }}/api/groups/group__binary__search.html#function-lower-bound">thrust::lower&#95;bound</a></b>(ForwardIterator first,</span>
<span>&nbsp;&nbsp;ForwardIterator last,</span>
<span>&nbsp;&nbsp;const LessThanComparable & value);</span>
<br>
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename ForwardIterator,</span>
<span>&nbsp;&nbsp;typename T,</span>
<span>&nbsp;&nbsp;typename StrictWeakOrdering&gt;</span>
<span>__host__ __device__ ForwardIterator </span><span><b><a href="{{ site.baseurl }}/api/groups/group__binary__search.html#function-lower-bound">thrust::lower&#95;bound</a></b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;ForwardIterator first,</span>
<span>&nbsp;&nbsp;ForwardIterator last,</span>
<span>&nbsp;&nbsp;const T & value,</span>
<span>&nbsp;&nbsp;StrictWeakOrdering comp);</span>
<br>
<span>template &lt;class ForwardIterator,</span>
<span>&nbsp;&nbsp;class T,</span>
<span>&nbsp;&nbsp;class StrictWeakOrdering&gt;</span>
<span>ForwardIterator </span><span><b><a href="{{ site.baseurl }}/api/groups/group__binary__search.html#function-lower-bound">thrust::lower&#95;bound</a></b>(ForwardIterator first,</span>
<span>&nbsp;&nbsp;ForwardIterator last,</span>
<span>&nbsp;&nbsp;const T & value,</span>
<span>&nbsp;&nbsp;StrictWeakOrdering comp);</span>
<br>
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename ForwardIterator,</span>
<span>&nbsp;&nbsp;typename LessThanComparable&gt;</span>
<span>__host__ __device__ ForwardIterator </span><span><b><a href="{{ site.baseurl }}/api/groups/group__binary__search.html#function-upper-bound">thrust::upper&#95;bound</a></b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;ForwardIterator first,</span>
<span>&nbsp;&nbsp;ForwardIterator last,</span>
<span>&nbsp;&nbsp;const LessThanComparable & value);</span>
<br>
<span>template &lt;class ForwardIterator,</span>
<span>&nbsp;&nbsp;class LessThanComparable&gt;</span>
<span>ForwardIterator </span><span><b><a href="{{ site.baseurl }}/api/groups/group__binary__search.html#function-upper-bound">thrust::upper&#95;bound</a></b>(ForwardIterator first,</span>
<span>&nbsp;&nbsp;ForwardIterator last,</span>
<span>&nbsp;&nbsp;const LessThanComparable & value);</span>
<br>
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename ForwardIterator,</span>
<span>&nbsp;&nbsp;typename T,</span>
<span>&nbsp;&nbsp;typename StrictWeakOrdering&gt;</span>
<span>__host__ __device__ ForwardIterator </span><span><b><a href="{{ site.baseurl }}/api/groups/group__binary__search.html#function-upper-bound">thrust::upper&#95;bound</a></b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;ForwardIterator first,</span>
<span>&nbsp;&nbsp;ForwardIterator last,</span>
<span>&nbsp;&nbsp;const T & value,</span>
<span>&nbsp;&nbsp;StrictWeakOrdering comp);</span>
<br>
<span>template &lt;class ForwardIterator,</span>
<span>&nbsp;&nbsp;class T,</span>
<span>&nbsp;&nbsp;class StrictWeakOrdering&gt;</span>
<span>ForwardIterator </span><span><b><a href="{{ site.baseurl }}/api/groups/group__binary__search.html#function-upper-bound">thrust::upper&#95;bound</a></b>(ForwardIterator first,</span>
<span>&nbsp;&nbsp;ForwardIterator last,</span>
<span>&nbsp;&nbsp;const T & value,</span>
<span>&nbsp;&nbsp;StrictWeakOrdering comp);</span>
<br>
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename ForwardIterator,</span>
<span>&nbsp;&nbsp;typename LessThanComparable&gt;</span>
<span>__host__ __device__ bool </span><span><b><a href="{{ site.baseurl }}/api/groups/group__binary__search.html#function-binary-search">thrust::binary&#95;search</a></b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;ForwardIterator first,</span>
<span>&nbsp;&nbsp;ForwardIterator last,</span>
<span>&nbsp;&nbsp;const LessThanComparable & value);</span>
<br>
<span>template &lt;class ForwardIterator,</span>
<span>&nbsp;&nbsp;class LessThanComparable&gt;</span>
<span>bool </span><span><b><a href="{{ site.baseurl }}/api/groups/group__binary__search.html#function-binary-search">thrust::binary&#95;search</a></b>(ForwardIterator first,</span>
<span>&nbsp;&nbsp;ForwardIterator last,</span>
<span>&nbsp;&nbsp;const LessThanComparable & value);</span>
<br>
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename ForwardIterator,</span>
<span>&nbsp;&nbsp;typename T,</span>
<span>&nbsp;&nbsp;typename StrictWeakOrdering&gt;</span>
<span>__host__ __device__ bool </span><span><b><a href="{{ site.baseurl }}/api/groups/group__binary__search.html#function-binary-search">thrust::binary&#95;search</a></b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;ForwardIterator first,</span>
<span>&nbsp;&nbsp;ForwardIterator last,</span>
<span>&nbsp;&nbsp;const T & value,</span>
<span>&nbsp;&nbsp;StrictWeakOrdering comp);</span>
<br>
<span>template &lt;class ForwardIterator,</span>
<span>&nbsp;&nbsp;class T,</span>
<span>&nbsp;&nbsp;class StrictWeakOrdering&gt;</span>
<span>bool </span><span><b><a href="{{ site.baseurl }}/api/groups/group__binary__search.html#function-binary-search">thrust::binary&#95;search</a></b>(ForwardIterator first,</span>
<span>&nbsp;&nbsp;ForwardIterator last,</span>
<span>&nbsp;&nbsp;const T & value,</span>
<span>&nbsp;&nbsp;StrictWeakOrdering comp);</span>
<br>
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename ForwardIterator,</span>
<span>&nbsp;&nbsp;typename LessThanComparable&gt;</span>
<span>__host__ __device__ <a href="{{ site.baseurl }}/api/classes/structthrust_1_1pair.html">thrust::pair</a>< ForwardIterator, ForwardIterator > </span><span><b><a href="{{ site.baseurl }}/api/groups/group__binary__search.html#function-equal-range">thrust::equal&#95;range</a></b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;ForwardIterator first,</span>
<span>&nbsp;&nbsp;ForwardIterator last,</span>
<span>&nbsp;&nbsp;const LessThanComparable & value);</span>
<br>
<span>template &lt;class ForwardIterator,</span>
<span>&nbsp;&nbsp;class LessThanComparable&gt;</span>
<span><a href="{{ site.baseurl }}/api/classes/structthrust_1_1pair.html">thrust::pair</a>< ForwardIterator, ForwardIterator > </span><span><b><a href="{{ site.baseurl }}/api/groups/group__binary__search.html#function-equal-range">thrust::equal&#95;range</a></b>(ForwardIterator first,</span>
<span>&nbsp;&nbsp;ForwardIterator last,</span>
<span>&nbsp;&nbsp;const LessThanComparable & value);</span>
<br>
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename ForwardIterator,</span>
<span>&nbsp;&nbsp;typename T,</span>
<span>&nbsp;&nbsp;typename StrictWeakOrdering&gt;</span>
<span>__host__ __device__ <a href="{{ site.baseurl }}/api/classes/structthrust_1_1pair.html">thrust::pair</a>< ForwardIterator, ForwardIterator > </span><span><b><a href="{{ site.baseurl }}/api/groups/group__binary__search.html#function-equal-range">thrust::equal&#95;range</a></b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;ForwardIterator first,</span>
<span>&nbsp;&nbsp;ForwardIterator last,</span>
<span>&nbsp;&nbsp;const T & value,</span>
<span>&nbsp;&nbsp;StrictWeakOrdering comp);</span>
<br>
<span>template &lt;class ForwardIterator,</span>
<span>&nbsp;&nbsp;class T,</span>
<span>&nbsp;&nbsp;class StrictWeakOrdering&gt;</span>
<span><a href="{{ site.baseurl }}/api/classes/structthrust_1_1pair.html">thrust::pair</a>< ForwardIterator, ForwardIterator > </span><span><b><a href="{{ site.baseurl }}/api/groups/group__binary__search.html#function-equal-range">thrust::equal&#95;range</a></b>(ForwardIterator first,</span>
<span>&nbsp;&nbsp;ForwardIterator last,</span>
<span>&nbsp;&nbsp;const T & value,</span>
<span>&nbsp;&nbsp;StrictWeakOrdering comp);</span>
</code>

## Functions

<h3 id="function-lower-bound">
Function <code>thrust::lower&#95;bound</code>
</h3>

<code class="doxybook">
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename ForwardIterator,</span>
<span>&nbsp;&nbsp;typename LessThanComparable&gt;</span>
<span>__host__ __device__ ForwardIterator </span><span><b>lower_bound</b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;ForwardIterator first,</span>
<span>&nbsp;&nbsp;ForwardIterator last,</span>
<span>&nbsp;&nbsp;const LessThanComparable & value);</span></code>
<code>lower&#95;bound</code> is a version of binary search: it attempts to find the element value in an ordered range <code>[first, last)</code>. Specifically, it returns the first position where value could be inserted without violating the ordering. This version of <code>lower&#95;bound</code> uses <code>operator&lt;</code> for comparison and returns the furthermost iterator <code>i</code> in <code>[first, last)</code> such that, for every iterator <code>j</code> in <code>[first, i)</code>, <code>&#42;j &lt; value</code>.

The algorithm's execution is parallelized as determined by <code>exec</code>.


The following code snippet demonstrates how to use <code>lower&#95;bound</code> to search for values in a ordered range using the <code>thrust::device</code> execution policy for parallelization:



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

thrust::lower_bound(thrust::device, input.begin(), input.end(), 0); // returns input.begin()
thrust::lower_bound(thrust::device, input.begin(), input.end(), 1); // returns input.begin() + 1
thrust::lower_bound(thrust::device, input.begin(), input.end(), 2); // returns input.begin() + 1
thrust::lower_bound(thrust::device, input.begin(), input.end(), 3); // returns input.begin() + 2
thrust::lower_bound(thrust::device, input.begin(), input.end(), 8); // returns input.begin() + 4
thrust::lower_bound(thrust::device, input.begin(), input.end(), 9); // returns input.end()
```

**Template Parameters**:
* **`DerivedPolicy`** The name of the derived execution policy. 
* **`ForwardIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a>. 
* **`LessThanComparable`** is a model of <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThanComparable</a>.

**Function Parameters**:
* **`exec`** The execution policy to use for parallelization. 
* **`first`** The beginning of the ordered sequence. 
* **`last`** The end of the ordered sequence. 
* **`value`** The value to be searched. 

**Returns**:
The furthermost iterator <code>i</code>, such that <code>&#42;i &lt; value</code>.

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
<span>&nbsp;&nbsp;class LessThanComparable&gt;</span>
<span>ForwardIterator </span><span><b>lower_bound</b>(ForwardIterator first,</span>
<span>&nbsp;&nbsp;ForwardIterator last,</span>
<span>&nbsp;&nbsp;const LessThanComparable & value);</span></code>
<code>lower&#95;bound</code> is a version of binary search: it attempts to find the element value in an ordered range <code>[first, last)</code>. Specifically, it returns the first position where value could be inserted without violating the ordering. This version of <code>lower&#95;bound</code> uses <code>operator&lt;</code> for comparison and returns the furthermost iterator <code>i</code> in <code>[first, last)</code> such that, for every iterator <code>j</code> in <code>[first, i)</code>, <code>&#42;j &lt; value</code>.


The following code snippet demonstrates how to use <code>lower&#95;bound</code> to search for values in a ordered range.



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

thrust::lower_bound(input.begin(), input.end(), 0); // returns input.begin()
thrust::lower_bound(input.begin(), input.end(), 1); // returns input.begin() + 1
thrust::lower_bound(input.begin(), input.end(), 2); // returns input.begin() + 1
thrust::lower_bound(input.begin(), input.end(), 3); // returns input.begin() + 2
thrust::lower_bound(input.begin(), input.end(), 8); // returns input.begin() + 4
thrust::lower_bound(input.begin(), input.end(), 9); // returns input.end()
```

**Template Parameters**:
* **`ForwardIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a>. 
* **`LessThanComparable`** is a model of <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThanComparable</a>.

**Function Parameters**:
* **`first`** The beginning of the ordered sequence. 
* **`last`** The end of the ordered sequence. 
* **`value`** The value to be searched. 

**Returns**:
The furthermost iterator <code>i</code>, such that <code>&#42;i &lt; value</code>.

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
<span>&nbsp;&nbsp;typename T,</span>
<span>&nbsp;&nbsp;typename StrictWeakOrdering&gt;</span>
<span>__host__ __device__ ForwardIterator </span><span><b>lower_bound</b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;ForwardIterator first,</span>
<span>&nbsp;&nbsp;ForwardIterator last,</span>
<span>&nbsp;&nbsp;const T & value,</span>
<span>&nbsp;&nbsp;StrictWeakOrdering comp);</span></code>
<code>lower&#95;bound</code> is a version of binary search: it attempts to find the element value in an ordered range <code>[first, last)</code>. Specifically, it returns the first position where value could be inserted without violating the ordering. This version of <code>lower&#95;bound</code> uses function object <code>comp</code> for comparison and returns the furthermost iterator <code>i</code> in <code>[first, last)</code> such that, for every iterator <code>j</code> in <code>[first, i)</code>, <code>comp(&#42;j, value)</code> is <code>true</code>.

The algorithm's execution is parallelized as determined by <code>exec</code>.


The following code snippet demonstrates how to use <code>lower&#95;bound</code> to search for values in a ordered range using the <code>thrust::device</code> execution policy for parallelization:



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

thrust::lower_bound(input.begin(), input.end(), 0, thrust::less<int>()); // returns input.begin()
thrust::lower_bound(input.begin(), input.end(), 1, thrust::less<int>()); // returns input.begin() + 1
thrust::lower_bound(input.begin(), input.end(), 2, thrust::less<int>()); // returns input.begin() + 1
thrust::lower_bound(input.begin(), input.end(), 3, thrust::less<int>()); // returns input.begin() + 2
thrust::lower_bound(input.begin(), input.end(), 8, thrust::less<int>()); // returns input.begin() + 4
thrust::lower_bound(input.begin(), input.end(), 9, thrust::less<int>()); // returns input.end()
```

**Template Parameters**:
* **`DerivedPolicy`** The name of the derived execution policy. 
* **`ForwardIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a>. 
* **`T`** is comparable to <code>ForwardIterator's</code><code>value&#95;type</code>. 
* **`StrictWeakOrdering`** is a model of <a href="https://en.cppreference.com/w/cpp/concepts/strict_weak_order">Strict Weak Ordering</a>.

**Function Parameters**:
* **`exec`** The execution policy to use for parallelization. 
* **`first`** The beginning of the ordered sequence. 
* **`last`** The end of the ordered sequence. 
* **`value`** The value to be searched. 
* **`comp`** The comparison operator. 

**Returns**:
The furthermost iterator <code>i</code>, such that <code>comp(&#42;i, value)</code> is <code>true</code>.

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
<span>&nbsp;&nbsp;class T,</span>
<span>&nbsp;&nbsp;class StrictWeakOrdering&gt;</span>
<span>ForwardIterator </span><span><b>lower_bound</b>(ForwardIterator first,</span>
<span>&nbsp;&nbsp;ForwardIterator last,</span>
<span>&nbsp;&nbsp;const T & value,</span>
<span>&nbsp;&nbsp;StrictWeakOrdering comp);</span></code>
<code>lower&#95;bound</code> is a version of binary search: it attempts to find the element value in an ordered range <code>[first, last)</code>. Specifically, it returns the first position where value could be inserted without violating the ordering. This version of <code>lower&#95;bound</code> uses function object <code>comp</code> for comparison and returns the furthermost iterator <code>i</code> in <code>[first, last)</code> such that, for every iterator <code>j</code> in <code>[first, i)</code>, <code>comp(&#42;j, value)</code> is <code>true</code>.


The following code snippet demonstrates how to use <code>lower&#95;bound</code> to search for values in a ordered range.



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

thrust::lower_bound(input.begin(), input.end(), 0, thrust::less<int>()); // returns input.begin()
thrust::lower_bound(input.begin(), input.end(), 1, thrust::less<int>()); // returns input.begin() + 1
thrust::lower_bound(input.begin(), input.end(), 2, thrust::less<int>()); // returns input.begin() + 1
thrust::lower_bound(input.begin(), input.end(), 3, thrust::less<int>()); // returns input.begin() + 2
thrust::lower_bound(input.begin(), input.end(), 8, thrust::less<int>()); // returns input.begin() + 4
thrust::lower_bound(input.begin(), input.end(), 9, thrust::less<int>()); // returns input.end()
```

**Template Parameters**:
* **`ForwardIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a>. 
* **`T`** is comparable to <code>ForwardIterator's</code><code>value&#95;type</code>. 
* **`StrictWeakOrdering`** is a model of <a href="https://en.cppreference.com/w/cpp/concepts/strict_weak_order">Strict Weak Ordering</a>.

**Function Parameters**:
* **`first`** The beginning of the ordered sequence. 
* **`last`** The end of the ordered sequence. 
* **`value`** The value to be searched. 
* **`comp`** The comparison operator. 

**Returns**:
The furthermost iterator <code>i</code>, such that <code>comp(&#42;i, value)</code> is <code>true</code>.

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
<span>&nbsp;&nbsp;typename LessThanComparable&gt;</span>
<span>__host__ __device__ ForwardIterator </span><span><b>upper_bound</b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;ForwardIterator first,</span>
<span>&nbsp;&nbsp;ForwardIterator last,</span>
<span>&nbsp;&nbsp;const LessThanComparable & value);</span></code>
<code>upper&#95;bound</code> is a version of binary search: it attempts to find the element value in an ordered range <code>[first, last)</code>. Specifically, it returns the last position where value could be inserted without violating the ordering. This version of <code>upper&#95;bound</code> uses <code>operator&lt;</code> for comparison and returns the furthermost iterator <code>i</code> in <code>[first, last)</code> such that, for every iterator <code>j</code> in <code>[first, i)</code>, <code>value &lt; &#42;j</code> is <code>false</code>.

The algorithm's execution is parallelized as determined by <code>exec</code>.


The following code snippet demonstrates how to use <code>upper&#95;bound</code> to search for values in a ordered range using the <code>thrust::device</code> execution policy for parallelism:



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

thrust::upper_bound(thrust::device, input.begin(), input.end(), 0); // returns input.begin() + 1
thrust::upper_bound(thrust::device, input.begin(), input.end(), 1); // returns input.begin() + 1
thrust::upper_bound(thrust::device, input.begin(), input.end(), 2); // returns input.begin() + 2
thrust::upper_bound(thrust::device, input.begin(), input.end(), 3); // returns input.begin() + 2
thrust::upper_bound(thrust::device, input.begin(), input.end(), 8); // returns input.end()
thrust::upper_bound(thrust::device, input.begin(), input.end(), 9); // returns input.end()
```

**Template Parameters**:
* **`DerivedPolicy`** The name of the derived execution policy. 
* **`ForwardIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a>. 
* **`LessThanComparable`** is a model of <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThanComparable</a>.

**Function Parameters**:
* **`exec`** The execution policy to use for parallelization. 
* **`first`** The beginning of the ordered sequence. 
* **`last`** The end of the ordered sequence. 
* **`value`** The value to be searched. 

**Returns**:
The furthermost iterator <code>i</code>, such that <code>value &lt; &#42;i</code> is <code>false</code>.

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
<span>&nbsp;&nbsp;class LessThanComparable&gt;</span>
<span>ForwardIterator </span><span><b>upper_bound</b>(ForwardIterator first,</span>
<span>&nbsp;&nbsp;ForwardIterator last,</span>
<span>&nbsp;&nbsp;const LessThanComparable & value);</span></code>
<code>upper&#95;bound</code> is a version of binary search: it attempts to find the element value in an ordered range <code>[first, last)</code>. Specifically, it returns the last position where value could be inserted without violating the ordering. This version of <code>upper&#95;bound</code> uses <code>operator&lt;</code> for comparison and returns the furthermost iterator <code>i</code> in <code>[first, last)</code> such that, for every iterator <code>j</code> in <code>[first, i)</code>, <code>value &lt; &#42;j</code> is <code>false</code>.


The following code snippet demonstrates how to use <code>upper&#95;bound</code> to search for values in a ordered range.



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

thrust::upper_bound(input.begin(), input.end(), 0); // returns input.begin() + 1
thrust::upper_bound(input.begin(), input.end(), 1); // returns input.begin() + 1
thrust::upper_bound(input.begin(), input.end(), 2); // returns input.begin() + 2
thrust::upper_bound(input.begin(), input.end(), 3); // returns input.begin() + 2
thrust::upper_bound(input.begin(), input.end(), 8); // returns input.end()
thrust::upper_bound(input.begin(), input.end(), 9); // returns input.end()
```

**Template Parameters**:
* **`ForwardIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a>. 
* **`LessThanComparable`** is a model of <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThanComparable</a>.

**Function Parameters**:
* **`first`** The beginning of the ordered sequence. 
* **`last`** The end of the ordered sequence. 
* **`value`** The value to be searched. 

**Returns**:
The furthermost iterator <code>i</code>, such that <code>value &lt; &#42;i</code> is <code>false</code>.

**See**:
* <a href="https://en.cppreference.com/w/cpp/algorithm/upper_bound">https://en.cppreference.com/w/cpp/algorithm/upper_bound</a>
* <code>lower&#95;bound</code>
* <code>equal&#95;range</code>
* <code><a href="{{ site.baseurl }}/api/groups/group__binary__search.html">Binary Search</a></code>

<h3 id="function-upper-bound">
Function <code>thrust::upper&#95;bound</code>
</h3>

<code class="doxybook">
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename ForwardIterator,</span>
<span>&nbsp;&nbsp;typename T,</span>
<span>&nbsp;&nbsp;typename StrictWeakOrdering&gt;</span>
<span>__host__ __device__ ForwardIterator </span><span><b>upper_bound</b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;ForwardIterator first,</span>
<span>&nbsp;&nbsp;ForwardIterator last,</span>
<span>&nbsp;&nbsp;const T & value,</span>
<span>&nbsp;&nbsp;StrictWeakOrdering comp);</span></code>
<code>upper&#95;bound</code> is a version of binary search: it attempts to find the element value in an ordered range <code>[first, last)</code>. Specifically, it returns the last position where value could be inserted without violating the ordering. This version of <code>upper&#95;bound</code> uses function object <code>comp</code> for comparison and returns the furthermost iterator <code>i</code> in <code>[first, last)</code> such that, for every iterator <code>j</code> in <code>[first, i)</code>, <code>comp(value, &#42;j)</code> is <code>false</code>.

The algorithm's execution is parallelized as determined by <code>exec</code>.


The following code snippet demonstrates how to use <code>upper&#95;bound</code> to search for values in a ordered range using the <code>thrust::device</code> execution policy for parallelization:



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

thrust::upper_bound(thrust::device, input.begin(), input.end(), 0, thrust::less<int>()); // returns input.begin() + 1
thrust::upper_bound(thrust::device, input.begin(), input.end(), 1, thrust::less<int>()); // returns input.begin() + 1
thrust::upper_bound(thrust::device, input.begin(), input.end(), 2, thrust::less<int>()); // returns input.begin() + 2
thrust::upper_bound(thrust::device, input.begin(), input.end(), 3, thrust::less<int>()); // returns input.begin() + 2
thrust::upper_bound(thrust::device, input.begin(), input.end(), 8, thrust::less<int>()); // returns input.end()
thrust::upper_bound(thrust::device, input.begin(), input.end(), 9, thrust::less<int>()); // returns input.end()
```

**Template Parameters**:
* **`DerivedPolicy`** The name of the derived execution policy. 
* **`ForwardIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a>. 
* **`T`** is comparable to <code>ForwardIterator's</code><code>value&#95;type</code>. 
* **`StrictWeakOrdering`** is a model of <a href="https://en.cppreference.com/w/cpp/concepts/strict_weak_order">Strict Weak Ordering</a>.

**Function Parameters**:
* **`exec`** The execution policy to use for parallelization. 
* **`first`** The beginning of the ordered sequence. 
* **`last`** The end of the ordered sequence. 
* **`value`** The value to be searched. 
* **`comp`** The comparison operator. 

**Returns**:
The furthermost iterator <code>i</code>, such that <code>comp(value, &#42;i)</code> is <code>false</code>.

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
<span>&nbsp;&nbsp;class T,</span>
<span>&nbsp;&nbsp;class StrictWeakOrdering&gt;</span>
<span>ForwardIterator </span><span><b>upper_bound</b>(ForwardIterator first,</span>
<span>&nbsp;&nbsp;ForwardIterator last,</span>
<span>&nbsp;&nbsp;const T & value,</span>
<span>&nbsp;&nbsp;StrictWeakOrdering comp);</span></code>
<code>upper&#95;bound</code> is a version of binary search: it attempts to find the element value in an ordered range <code>[first, last)</code>. Specifically, it returns the last position where value could be inserted without violating the ordering. This version of <code>upper&#95;bound</code> uses function object <code>comp</code> for comparison and returns the furthermost iterator <code>i</code> in <code>[first, last)</code> such that, for every iterator <code>j</code> in <code>[first, i)</code>, <code>comp(value, &#42;j)</code> is <code>false</code>.


The following code snippet demonstrates how to use <code>upper&#95;bound</code> to search for values in a ordered range.



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

thrust::upper_bound(input.begin(), input.end(), 0, thrust::less<int>()); // returns input.begin() + 1
thrust::upper_bound(input.begin(), input.end(), 1, thrust::less<int>()); // returns input.begin() + 1
thrust::upper_bound(input.begin(), input.end(), 2, thrust::less<int>()); // returns input.begin() + 2
thrust::upper_bound(input.begin(), input.end(), 3, thrust::less<int>()); // returns input.begin() + 2
thrust::upper_bound(input.begin(), input.end(), 8, thrust::less<int>()); // returns input.end()
thrust::upper_bound(input.begin(), input.end(), 9, thrust::less<int>()); // returns input.end()
```

**Template Parameters**:
* **`ForwardIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a>. 
* **`T`** is comparable to <code>ForwardIterator's</code><code>value&#95;type</code>. 
* **`StrictWeakOrdering`** is a model of <a href="https://en.cppreference.com/w/cpp/concepts/strict_weak_order">Strict Weak Ordering</a>.

**Function Parameters**:
* **`first`** The beginning of the ordered sequence. 
* **`last`** The end of the ordered sequence. 
* **`value`** The value to be searched. 
* **`comp`** The comparison operator. 

**Returns**:
The furthermost iterator <code>i</code>, such that <code>comp(value, &#42;i)</code> is <code>false</code>.

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
<span>&nbsp;&nbsp;typename LessThanComparable&gt;</span>
<span>__host__ __device__ bool </span><span><b>binary_search</b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;ForwardIterator first,</span>
<span>&nbsp;&nbsp;ForwardIterator last,</span>
<span>&nbsp;&nbsp;const LessThanComparable & value);</span></code>
<code>binary&#95;search</code> is a version of binary search: it attempts to find the element value in an ordered range <code>[first, last)</code>. It returns <code>true</code> if an element that is equivalent to <code>value</code> is present in <code>[first, last)</code> and <code>false</code> if no such element exists. Specifically, this version returns <code>true</code> if and only if there exists an iterator <code>i</code> in <code>[first, last)</code> such that <code>&#42;i &lt; value</code> and <code>value &lt; &#42;i</code> are both <code>false</code>.

The algorithm's execution is parallelized as determined by <code>exec</code>.


The following code snippet demonstrates how to use <code>binary&#95;search</code> to search for values in a ordered range using the <code>thrust::device</code> execution policy for parallelization:



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

thrust::binary_search(thrust::device, input.begin(), input.end(), 0); // returns true
thrust::binary_search(thrust::device, input.begin(), input.end(), 1); // returns false
thrust::binary_search(thrust::device, input.begin(), input.end(), 2); // returns true
thrust::binary_search(thrust::device, input.begin(), input.end(), 3); // returns false
thrust::binary_search(thrust::device, input.begin(), input.end(), 8); // returns true
thrust::binary_search(thrust::device, input.begin(), input.end(), 9); // returns false
```

**Template Parameters**:
* **`DerivedPolicy`** The name of the derived execution policy. 
* **`ForwardIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a>. 
* **`LessThanComparable`** is a model of <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThanComparable</a>.

**Function Parameters**:
* **`exec`** The execution policy to use for parallelization. 
* **`first`** The beginning of the ordered sequence. 
* **`last`** The end of the ordered sequence. 
* **`value`** The value to be searched. 

**Returns**:
<code>true</code> if an equivalent element exists in <code>[first, last)</code>, otherwise <code>false</code>.

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
<span>&nbsp;&nbsp;class LessThanComparable&gt;</span>
<span>bool </span><span><b>binary_search</b>(ForwardIterator first,</span>
<span>&nbsp;&nbsp;ForwardIterator last,</span>
<span>&nbsp;&nbsp;const LessThanComparable & value);</span></code>
<code>binary&#95;search</code> is a version of binary search: it attempts to find the element value in an ordered range <code>[first, last)</code>. It returns <code>true</code> if an element that is equivalent to <code>value</code> is present in <code>[first, last)</code> and <code>false</code> if no such element exists. Specifically, this version returns <code>true</code> if and only if there exists an iterator <code>i</code> in <code>[first, last)</code> such that <code>&#42;i &lt; value</code> and <code>value &lt; &#42;i</code> are both <code>false</code>.


The following code snippet demonstrates how to use <code>binary&#95;search</code> to search for values in a ordered range.



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

thrust::binary_search(input.begin(), input.end(), 0); // returns true
thrust::binary_search(input.begin(), input.end(), 1); // returns false
thrust::binary_search(input.begin(), input.end(), 2); // returns true
thrust::binary_search(input.begin(), input.end(), 3); // returns false
thrust::binary_search(input.begin(), input.end(), 8); // returns true
thrust::binary_search(input.begin(), input.end(), 9); // returns false
```

**Template Parameters**:
* **`ForwardIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a>. 
* **`LessThanComparable`** is a model of <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThanComparable</a>.

**Function Parameters**:
* **`first`** The beginning of the ordered sequence. 
* **`last`** The end of the ordered sequence. 
* **`value`** The value to be searched. 

**Returns**:
<code>true</code> if an equivalent element exists in <code>[first, last)</code>, otherwise <code>false</code>.

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
<span>&nbsp;&nbsp;typename T,</span>
<span>&nbsp;&nbsp;typename StrictWeakOrdering&gt;</span>
<span>__host__ __device__ bool </span><span><b>binary_search</b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;ForwardIterator first,</span>
<span>&nbsp;&nbsp;ForwardIterator last,</span>
<span>&nbsp;&nbsp;const T & value,</span>
<span>&nbsp;&nbsp;StrictWeakOrdering comp);</span></code>
<code>binary&#95;search</code> is a version of binary search: it attempts to find the element value in an ordered range <code>[first, last)</code>. It returns <code>true</code> if an element that is equivalent to <code>value</code> is present in <code>[first, last)</code> and <code>false</code> if no such element exists. Specifically, this version returns <code>true</code> if and only if there exists an iterator <code>i</code> in <code>[first, last)</code> such that <code>comp(&#42;i, value)</code> and <code>comp(value, &#42;i)</code> are both <code>false</code>.

The algorithm's execution is parallelized as determined by <code>exec</code>.


The following code snippet demonstrates how to use <code>binary&#95;search</code> to search for values in a ordered range using the <code>thrust::device</code> execution policy for parallelization:



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

thrust::binary_search(thrust::device, input.begin(), input.end(), 0, thrust::less<int>()); // returns true
thrust::binary_search(thrust::device, input.begin(), input.end(), 1, thrust::less<int>()); // returns false
thrust::binary_search(thrust::device, input.begin(), input.end(), 2, thrust::less<int>()); // returns true
thrust::binary_search(thrust::device, input.begin(), input.end(), 3, thrust::less<int>()); // returns false
thrust::binary_search(thrust::device, input.begin(), input.end(), 8, thrust::less<int>()); // returns true
thrust::binary_search(thrust::device, input.begin(), input.end(), 9, thrust::less<int>()); // returns false
```

**Template Parameters**:
* **`DerivedPolicy`** The name of the derived execution policy. 
* **`ForwardIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a>. 
* **`T`** is comparable to <code>ForwardIterator's</code><code>value&#95;type</code>. 
* **`StrictWeakOrdering`** is a model of <a href="https://en.cppreference.com/w/cpp/concepts/strict_weak_order">Strict Weak Ordering</a>.

**Function Parameters**:
* **`exec`** The execution policy to use for parallelization. 
* **`first`** The beginning of the ordered sequence. 
* **`last`** The end of the ordered sequence. 
* **`value`** The value to be searched. 
* **`comp`** The comparison operator. 

**Returns**:
<code>true</code> if an equivalent element exists in <code>[first, last)</code>, otherwise <code>false</code>.

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
<span>&nbsp;&nbsp;class T,</span>
<span>&nbsp;&nbsp;class StrictWeakOrdering&gt;</span>
<span>bool </span><span><b>binary_search</b>(ForwardIterator first,</span>
<span>&nbsp;&nbsp;ForwardIterator last,</span>
<span>&nbsp;&nbsp;const T & value,</span>
<span>&nbsp;&nbsp;StrictWeakOrdering comp);</span></code>
<code>binary&#95;search</code> is a version of binary search: it attempts to find the element value in an ordered range <code>[first, last)</code>. It returns <code>true</code> if an element that is equivalent to <code>value</code> is present in <code>[first, last)</code> and <code>false</code> if no such element exists. Specifically, this version returns <code>true</code> if and only if there exists an iterator <code>i</code> in <code>[first, last)</code> such that <code>comp(&#42;i, value)</code> and <code>comp(value, &#42;i)</code> are both <code>false</code>.


The following code snippet demonstrates how to use <code>binary&#95;search</code> to search for values in a ordered range.



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

thrust::binary_search(input.begin(), input.end(), 0, thrust::less<int>()); // returns true
thrust::binary_search(input.begin(), input.end(), 1, thrust::less<int>()); // returns false
thrust::binary_search(input.begin(), input.end(), 2, thrust::less<int>()); // returns true
thrust::binary_search(input.begin(), input.end(), 3, thrust::less<int>()); // returns false
thrust::binary_search(input.begin(), input.end(), 8, thrust::less<int>()); // returns true
thrust::binary_search(input.begin(), input.end(), 9, thrust::less<int>()); // returns false
```

**Template Parameters**:
* **`ForwardIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a>. 
* **`T`** is comparable to <code>ForwardIterator's</code><code>value&#95;type</code>. 
* **`StrictWeakOrdering`** is a model of <a href="https://en.cppreference.com/w/cpp/concepts/strict_weak_order">Strict Weak Ordering</a>.

**Function Parameters**:
* **`first`** The beginning of the ordered sequence. 
* **`last`** The end of the ordered sequence. 
* **`value`** The value to be searched. 
* **`comp`** The comparison operator. 

**Returns**:
<code>true</code> if an equivalent element exists in <code>[first, last)</code>, otherwise <code>false</code>.

**See**:
* <a href="https://en.cppreference.com/w/cpp/algorithm/binary_search">https://en.cppreference.com/w/cpp/algorithm/binary_search</a>
* <code>lower&#95;bound</code>
* <code>upper&#95;bound</code>
* <code>equal&#95;range</code>

<h3 id="function-equal-range">
Function <code>thrust::equal&#95;range</code>
</h3>

<code class="doxybook">
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename ForwardIterator,</span>
<span>&nbsp;&nbsp;typename LessThanComparable&gt;</span>
<span>__host__ __device__ <a href="{{ site.baseurl }}/api/classes/structthrust_1_1pair.html">thrust::pair</a>< ForwardIterator, ForwardIterator > </span><span><b>equal_range</b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;ForwardIterator first,</span>
<span>&nbsp;&nbsp;ForwardIterator last,</span>
<span>&nbsp;&nbsp;const LessThanComparable & value);</span></code>
<code>equal&#95;range</code> is a version of binary search: it attempts to find the element value in an ordered range <code>[first, last)</code>. The value returned by <code>equal&#95;range</code> is essentially a combination of the values returned by <code>lower&#95;bound</code> and <code>upper&#95;bound:</code> it returns a <code>pair</code> of iterators <code>i</code> and <code>j</code> such that <code>i</code> is the first position where value could be inserted without violating the ordering and <code>j</code> is the last position where value could be inserted without violating the ordering. It follows that every element in the range <code>[i, j)</code> is equivalent to value, and that <code>[i, j)</code> is the largest subrange of <code>[first, last)</code> that has this property.

This version of <code>equal&#95;range</code> returns a <code>pair</code> of iterators <code>[i, j)</code>, where <code>i</code> is the furthermost iterator in <code>[first, last)</code> such that, for every iterator <code>k</code> in <code>[first, i)</code>, <code>&#42;k &lt; value</code>. <code>j</code> is the furthermost iterator in <code>[first, last)</code> such that, for every iterator <code>k</code> in <code>[first, j)</code>, <code>value &lt; &#42;k</code> is <code>false</code>. For every iterator <code>k</code> in <code>[i, j)</code>, neither <code>value &lt; &#42;k</code> nor <code>&#42;k &lt; value</code> is <code>true</code>.

The algorithm's execution is parallelized as determined by <code>exec</code>.


The following code snippet demonstrates how to use <code>equal&#95;range</code> to search for values in a ordered range using the <code>thrust::device</code> execution policy for parallelization:



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

thrust::equal_range(thrust::device, input.begin(), input.end(), 0); // returns [input.begin(), input.begin() + 1)
thrust::equal_range(thrust::device, input.begin(), input.end(), 1); // returns [input.begin() + 1, input.begin() + 1)
thrust::equal_range(thrust::device, input.begin(), input.end(), 2); // returns [input.begin() + 1, input.begin() + 2)
thrust::equal_range(thrust::device, input.begin(), input.end(), 3); // returns [input.begin() + 2, input.begin() + 2)
thrust::equal_range(thrust::device, input.begin(), input.end(), 8); // returns [input.begin() + 4, input.end)
thrust::equal_range(thrust::device, input.begin(), input.end(), 9); // returns [input.end(), input.end)
```

**Template Parameters**:
* **`DerivedPolicy`** The name of the derived execution policy. 
* **`ForwardIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a>. 
* **`LessThanComparable`** is a model of <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThanComparable</a>.

**Function Parameters**:
* **`exec`** The execution policy to use for parallelization. 
* **`first`** The beginning of the ordered sequence. 
* **`last`** The end of the ordered sequence. 
* **`value`** The value to be searched. 

**Returns**:
A <code>pair</code> of iterators <code>[i, j)</code> that define the range of equivalent elements.

**See**:
* <a href="https://en.cppreference.com/w/cpp/algorithm/equal_range">https://en.cppreference.com/w/cpp/algorithm/equal_range</a>
* <code>lower&#95;bound</code>
* <code>upper&#95;bound</code>
* <code><a href="{{ site.baseurl }}/api/groups/group__binary__search.html">Binary Search</a></code>

<h3 id="function-equal-range">
Function <code>thrust::equal&#95;range</code>
</h3>

<code class="doxybook">
<span>template &lt;class ForwardIterator,</span>
<span>&nbsp;&nbsp;class LessThanComparable&gt;</span>
<span><a href="{{ site.baseurl }}/api/classes/structthrust_1_1pair.html">thrust::pair</a>< ForwardIterator, ForwardIterator > </span><span><b>equal_range</b>(ForwardIterator first,</span>
<span>&nbsp;&nbsp;ForwardIterator last,</span>
<span>&nbsp;&nbsp;const LessThanComparable & value);</span></code>
<code>equal&#95;range</code> is a version of binary search: it attempts to find the element value in an ordered range <code>[first, last)</code>. The value returned by <code>equal&#95;range</code> is essentially a combination of the values returned by <code>lower&#95;bound</code> and <code>upper&#95;bound:</code> it returns a <code>pair</code> of iterators <code>i</code> and <code>j</code> such that <code>i</code> is the first position where value could be inserted without violating the ordering and <code>j</code> is the last position where value could be inserted without violating the ordering. It follows that every element in the range <code>[i, j)</code> is equivalent to value, and that <code>[i, j)</code> is the largest subrange of <code>[first, last)</code> that has this property.

This version of <code>equal&#95;range</code> returns a <code>pair</code> of iterators <code>[i, j)</code>, where <code>i</code> is the furthermost iterator in <code>[first, last)</code> such that, for every iterator <code>k</code> in <code>[first, i)</code>, <code>&#42;k &lt; value</code>. <code>j</code> is the furthermost iterator in <code>[first, last)</code> such that, for every iterator <code>k</code> in <code>[first, j)</code>, <code>value &lt; &#42;k</code> is <code>false</code>. For every iterator <code>k</code> in <code>[i, j)</code>, neither <code>value &lt; &#42;k</code> nor <code>&#42;k &lt; value</code> is <code>true</code>.


The following code snippet demonstrates how to use <code>equal&#95;range</code> to search for values in a ordered range.



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

thrust::equal_range(input.begin(), input.end(), 0); // returns [input.begin(), input.begin() + 1)
thrust::equal_range(input.begin(), input.end(), 1); // returns [input.begin() + 1, input.begin() + 1)
thrust::equal_range(input.begin(), input.end(), 2); // returns [input.begin() + 1, input.begin() + 2)
thrust::equal_range(input.begin(), input.end(), 3); // returns [input.begin() + 2, input.begin() + 2)
thrust::equal_range(input.begin(), input.end(), 8); // returns [input.begin() + 4, input.end)
thrust::equal_range(input.begin(), input.end(), 9); // returns [input.end(), input.end)
```

**Template Parameters**:
* **`ForwardIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a>. 
* **`LessThanComparable`** is a model of <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThanComparable</a>.

**Function Parameters**:
* **`first`** The beginning of the ordered sequence. 
* **`last`** The end of the ordered sequence. 
* **`value`** The value to be searched. 

**Returns**:
A <code>pair</code> of iterators <code>[i, j)</code> that define the range of equivalent elements.

**See**:
* <a href="https://en.cppreference.com/w/cpp/algorithm/equal_range">https://en.cppreference.com/w/cpp/algorithm/equal_range</a>
* <code>lower&#95;bound</code>
* <code>upper&#95;bound</code>
* <code><a href="{{ site.baseurl }}/api/groups/group__binary__search.html">Binary Search</a></code>

<h3 id="function-equal-range">
Function <code>thrust::equal&#95;range</code>
</h3>

<code class="doxybook">
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename ForwardIterator,</span>
<span>&nbsp;&nbsp;typename T,</span>
<span>&nbsp;&nbsp;typename StrictWeakOrdering&gt;</span>
<span>__host__ __device__ <a href="{{ site.baseurl }}/api/classes/structthrust_1_1pair.html">thrust::pair</a>< ForwardIterator, ForwardIterator > </span><span><b>equal_range</b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;ForwardIterator first,</span>
<span>&nbsp;&nbsp;ForwardIterator last,</span>
<span>&nbsp;&nbsp;const T & value,</span>
<span>&nbsp;&nbsp;StrictWeakOrdering comp);</span></code>
<code>equal&#95;range</code> is a version of binary search: it attempts to find the element value in an ordered range <code>[first, last)</code>. The value returned by <code>equal&#95;range</code> is essentially a combination of the values returned by <code>lower&#95;bound</code> and <code>upper&#95;bound:</code> it returns a <code>pair</code> of iterators <code>i</code> and <code>j</code> such that <code>i</code> is the first position where value could be inserted without violating the ordering and <code>j</code> is the last position where value could be inserted without violating the ordering. It follows that every element in the range <code>[i, j)</code> is equivalent to value, and that <code>[i, j)</code> is the largest subrange of <code>[first, last)</code> that has this property.

This version of <code>equal&#95;range</code> returns a <code>pair</code> of iterators <code>[i, j)</code>. <code>i</code> is the furthermost iterator in <code>[first, last)</code> such that, for every iterator <code>k</code> in <code>[first, i)</code>, <code>comp(&#42;k, value)</code> is <code>true</code>. <code>j</code> is the furthermost iterator in <code>[first, last)</code> such that, for every iterator <code>k</code> in <code>[first, last)</code>, <code>comp(value, &#42;k)</code> is <code>false</code>. For every iterator <code>k</code> in <code>[i, j)</code>, neither <code>comp(value, &#42;k)</code> nor <code>comp(&#42;k, value)</code> is <code>true</code>.

The algorithm's execution is parallelized as determined by <code>exec</code>.


The following code snippet demonstrates how to use <code>equal&#95;range</code> to search for values in a ordered range using the <code>thrust::device</code> execution policy for parallelization:



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

thrust::equal_range(thrust::device, input.begin(), input.end(), 0, thrust::less<int>()); // returns [input.begin(), input.begin() + 1)
thrust::equal_range(thrust::device, input.begin(), input.end(), 1, thrust::less<int>()); // returns [input.begin() + 1, input.begin() + 1)
thrust::equal_range(thrust::device, input.begin(), input.end(), 2, thrust::less<int>()); // returns [input.begin() + 1, input.begin() + 2)
thrust::equal_range(thrust::device, input.begin(), input.end(), 3, thrust::less<int>()); // returns [input.begin() + 2, input.begin() + 2)
thrust::equal_range(thrust::device, input.begin(), input.end(), 8, thrust::less<int>()); // returns [input.begin() + 4, input.end)
thrust::equal_range(thrust::device, input.begin(), input.end(), 9, thrust::less<int>()); // returns [input.end(), input.end)
```

**Template Parameters**:
* **`DerivedPolicy`** The name of the derived execution policy. 
* **`ForwardIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a>. 
* **`T`** is comparable to <code>ForwardIterator's</code><code>value&#95;type</code>. 
* **`StrictWeakOrdering`** is a model of <a href="https://en.cppreference.com/w/cpp/concepts/strict_weak_order">Strict Weak Ordering</a>.

**Function Parameters**:
* **`exec`** The execution policy to use for parallelization. 
* **`first`** The beginning of the ordered sequence. 
* **`last`** The end of the ordered sequence. 
* **`value`** The value to be searched. 
* **`comp`** The comparison operator. 

**Returns**:
A <code>pair</code> of iterators <code>[i, j)</code> that define the range of equivalent elements.

**See**:
* <a href="https://en.cppreference.com/w/cpp/algorithm/equal_range">https://en.cppreference.com/w/cpp/algorithm/equal_range</a>
* <code>lower&#95;bound</code>
* <code>upper&#95;bound</code>
* <code><a href="{{ site.baseurl }}/api/groups/group__binary__search.html">Binary Search</a></code>

<h3 id="function-equal-range">
Function <code>thrust::equal&#95;range</code>
</h3>

<code class="doxybook">
<span>template &lt;class ForwardIterator,</span>
<span>&nbsp;&nbsp;class T,</span>
<span>&nbsp;&nbsp;class StrictWeakOrdering&gt;</span>
<span><a href="{{ site.baseurl }}/api/classes/structthrust_1_1pair.html">thrust::pair</a>< ForwardIterator, ForwardIterator > </span><span><b>equal_range</b>(ForwardIterator first,</span>
<span>&nbsp;&nbsp;ForwardIterator last,</span>
<span>&nbsp;&nbsp;const T & value,</span>
<span>&nbsp;&nbsp;StrictWeakOrdering comp);</span></code>
<code>equal&#95;range</code> is a version of binary search: it attempts to find the element value in an ordered range <code>[first, last)</code>. The value returned by <code>equal&#95;range</code> is essentially a combination of the values returned by <code>lower&#95;bound</code> and <code>upper&#95;bound:</code> it returns a <code>pair</code> of iterators <code>i</code> and <code>j</code> such that <code>i</code> is the first position where value could be inserted without violating the ordering and <code>j</code> is the last position where value could be inserted without violating the ordering. It follows that every element in the range <code>[i, j)</code> is equivalent to value, and that <code>[i, j)</code> is the largest subrange of <code>[first, last)</code> that has this property.

This version of <code>equal&#95;range</code> returns a <code>pair</code> of iterators <code>[i, j)</code>. <code>i</code> is the furthermost iterator in <code>[first, last)</code> such that, for every iterator <code>k</code> in <code>[first, i)</code>, <code>comp(&#42;k, value)</code> is <code>true</code>. <code>j</code> is the furthermost iterator in <code>[first, last)</code> such that, for every iterator <code>k</code> in <code>[first, last)</code>, <code>comp(value, &#42;k)</code> is <code>false</code>. For every iterator <code>k</code> in <code>[i, j)</code>, neither <code>comp(value, &#42;k)</code> nor <code>comp(&#42;k, value)</code> is <code>true</code>.


The following code snippet demonstrates how to use <code>equal&#95;range</code> to search for values in a ordered range.



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

thrust::equal_range(input.begin(), input.end(), 0, thrust::less<int>()); // returns [input.begin(), input.begin() + 1)
thrust::equal_range(input.begin(), input.end(), 1, thrust::less<int>()); // returns [input.begin() + 1, input.begin() + 1)
thrust::equal_range(input.begin(), input.end(), 2, thrust::less<int>()); // returns [input.begin() + 1, input.begin() + 2)
thrust::equal_range(input.begin(), input.end(), 3, thrust::less<int>()); // returns [input.begin() + 2, input.begin() + 2)
thrust::equal_range(input.begin(), input.end(), 8, thrust::less<int>()); // returns [input.begin() + 4, input.end)
thrust::equal_range(input.begin(), input.end(), 9, thrust::less<int>()); // returns [input.end(), input.end)
```

**Template Parameters**:
* **`ForwardIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a>. 
* **`T`** is comparable to <code>ForwardIterator's</code><code>value&#95;type</code>. 
* **`StrictWeakOrdering`** is a model of <a href="https://en.cppreference.com/w/cpp/concepts/strict_weak_order">Strict Weak Ordering</a>.

**Function Parameters**:
* **`first`** The beginning of the ordered sequence. 
* **`last`** The end of the ordered sequence. 
* **`value`** The value to be searched. 
* **`comp`** The comparison operator. 

**Returns**:
A <code>pair</code> of iterators <code>[i, j)</code> that define the range of equivalent elements.

**See**:
* <a href="https://en.cppreference.com/w/cpp/algorithm/equal_range">https://en.cppreference.com/w/cpp/algorithm/equal_range</a>
* <code>lower&#95;bound</code>
* <code>upper&#95;bound</code>
* <code><a href="{{ site.baseurl }}/api/groups/group__binary__search.html">Binary Search</a></code>


