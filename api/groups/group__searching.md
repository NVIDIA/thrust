---
title: Searching
parent: Algorithms
grand_parent: API
nav_exclude: false
has_children: true
has_toc: false
---

# Searching

## Groups

* **[Binary Search]({{ site.baseurl }}/api/groups/group__binary__search.html)**

<code class="doxybook">
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename InputIterator,</span>
<span>&nbsp;&nbsp;typename T&gt;</span>
<span>__host__ __device__ InputIterator </span><span><b><a href="{{ site.baseurl }}/api/groups/group__searching.html#function-find">thrust::find</a></b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;InputIterator first,</span>
<span>&nbsp;&nbsp;InputIterator last,</span>
<span>&nbsp;&nbsp;const T & value);</span>
<br>
<span>template &lt;typename InputIterator,</span>
<span>&nbsp;&nbsp;typename T&gt;</span>
<span>InputIterator </span><span><b><a href="{{ site.baseurl }}/api/groups/group__searching.html#function-find">thrust::find</a></b>(InputIterator first,</span>
<span>&nbsp;&nbsp;InputIterator last,</span>
<span>&nbsp;&nbsp;const T & value);</span>
<br>
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename InputIterator,</span>
<span>&nbsp;&nbsp;typename Predicate&gt;</span>
<span>__host__ __device__ InputIterator </span><span><b><a href="{{ site.baseurl }}/api/groups/group__searching.html#function-find-if">thrust::find&#95;if</a></b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;InputIterator first,</span>
<span>&nbsp;&nbsp;InputIterator last,</span>
<span>&nbsp;&nbsp;Predicate pred);</span>
<br>
<span>template &lt;typename InputIterator,</span>
<span>&nbsp;&nbsp;typename Predicate&gt;</span>
<span>InputIterator </span><span><b><a href="{{ site.baseurl }}/api/groups/group__searching.html#function-find-if">thrust::find&#95;if</a></b>(InputIterator first,</span>
<span>&nbsp;&nbsp;InputIterator last,</span>
<span>&nbsp;&nbsp;Predicate pred);</span>
<br>
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename InputIterator,</span>
<span>&nbsp;&nbsp;typename Predicate&gt;</span>
<span>__host__ __device__ InputIterator </span><span><b><a href="{{ site.baseurl }}/api/groups/group__searching.html#function-find-if-not">thrust::find&#95;if&#95;not</a></b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;InputIterator first,</span>
<span>&nbsp;&nbsp;InputIterator last,</span>
<span>&nbsp;&nbsp;Predicate pred);</span>
<br>
<span>template &lt;typename InputIterator,</span>
<span>&nbsp;&nbsp;typename Predicate&gt;</span>
<span>InputIterator </span><span><b><a href="{{ site.baseurl }}/api/groups/group__searching.html#function-find-if-not">thrust::find&#95;if&#95;not</a></b>(InputIterator first,</span>
<span>&nbsp;&nbsp;InputIterator last,</span>
<span>&nbsp;&nbsp;Predicate pred);</span>
<br>
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename InputIterator1,</span>
<span>&nbsp;&nbsp;typename InputIterator2&gt;</span>
<span>__host__ __device__ <a href="{{ site.baseurl }}/api/classes/structthrust_1_1pair.html">thrust::pair</a>< InputIterator1, InputIterator2 > </span><span><b><a href="{{ site.baseurl }}/api/groups/group__searching.html#function-mismatch">thrust::mismatch</a></b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;InputIterator1 first1,</span>
<span>&nbsp;&nbsp;InputIterator1 last1,</span>
<span>&nbsp;&nbsp;InputIterator2 first2);</span>
<br>
<span>template &lt;typename InputIterator1,</span>
<span>&nbsp;&nbsp;typename InputIterator2&gt;</span>
<span><a href="{{ site.baseurl }}/api/classes/structthrust_1_1pair.html">thrust::pair</a>< InputIterator1, InputIterator2 > </span><span><b><a href="{{ site.baseurl }}/api/groups/group__searching.html#function-mismatch">thrust::mismatch</a></b>(InputIterator1 first1,</span>
<span>&nbsp;&nbsp;InputIterator1 last1,</span>
<span>&nbsp;&nbsp;InputIterator2 first2);</span>
<br>
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename InputIterator1,</span>
<span>&nbsp;&nbsp;typename InputIterator2,</span>
<span>&nbsp;&nbsp;typename BinaryPredicate&gt;</span>
<span>__host__ __device__ <a href="{{ site.baseurl }}/api/classes/structthrust_1_1pair.html">thrust::pair</a>< InputIterator1, InputIterator2 > </span><span><b><a href="{{ site.baseurl }}/api/groups/group__searching.html#function-mismatch">thrust::mismatch</a></b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;InputIterator1 first1,</span>
<span>&nbsp;&nbsp;InputIterator1 last1,</span>
<span>&nbsp;&nbsp;InputIterator2 first2,</span>
<span>&nbsp;&nbsp;BinaryPredicate pred);</span>
<br>
<span>template &lt;typename InputIterator1,</span>
<span>&nbsp;&nbsp;typename InputIterator2,</span>
<span>&nbsp;&nbsp;typename BinaryPredicate&gt;</span>
<span><a href="{{ site.baseurl }}/api/classes/structthrust_1_1pair.html">thrust::pair</a>< InputIterator1, InputIterator2 > </span><span><b><a href="{{ site.baseurl }}/api/groups/group__searching.html#function-mismatch">thrust::mismatch</a></b>(InputIterator1 first1,</span>
<span>&nbsp;&nbsp;InputIterator1 last1,</span>
<span>&nbsp;&nbsp;InputIterator2 first2,</span>
<span>&nbsp;&nbsp;BinaryPredicate pred);</span>
<br>
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename ForwardIterator,</span>
<span>&nbsp;&nbsp;typename Predicate&gt;</span>
<span>__host__ __device__ ForwardIterator </span><span><b><a href="{{ site.baseurl }}/api/groups/group__searching.html#function-partition-point">thrust::partition&#95;point</a></b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;ForwardIterator first,</span>
<span>&nbsp;&nbsp;ForwardIterator last,</span>
<span>&nbsp;&nbsp;Predicate pred);</span>
<br>
<span>template &lt;typename ForwardIterator,</span>
<span>&nbsp;&nbsp;typename Predicate&gt;</span>
<span>ForwardIterator </span><span><b><a href="{{ site.baseurl }}/api/groups/group__searching.html#function-partition-point">thrust::partition&#95;point</a></b>(ForwardIterator first,</span>
<span>&nbsp;&nbsp;ForwardIterator last,</span>
<span>&nbsp;&nbsp;Predicate pred);</span>
</code>

## Functions

<h3 id="function-find">
Function <code>thrust::find</code>
</h3>

<code class="doxybook">
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename InputIterator,</span>
<span>&nbsp;&nbsp;typename T&gt;</span>
<span>__host__ __device__ InputIterator </span><span><b>find</b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;InputIterator first,</span>
<span>&nbsp;&nbsp;InputIterator last,</span>
<span>&nbsp;&nbsp;const T & value);</span></code>
<code>find</code> returns the first iterator <code>i</code> in the range <code>[first, last)</code> such that <code>&#42;i == value</code> or <code>last</code> if no such iterator exists.

The algorithm's execution is parallelized as determined by <code>exec</code>.



```cpp
#include <thrust/find.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
...
thrust::device_vector<int> input(4);

input[0] = 0;
input[1] = 5;
input[2] = 3;
input[3] = 7;

thrust::device_vector<int>::iterator iter;

iter = thrust::find(thrust::device, input.begin(), input.end(), 3); // returns input.first() + 2
iter = thrust::find(thrust::device, input.begin(), input.end(), 5); // returns input.first() + 1
iter = thrust::find(thrust::device, input.begin(), input.end(), 9); // returns input.end()
```

**Template Parameters**:
* **`DerivedPolicy`** The name of the derived execution policy. 
* **`InputIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a> and <code>InputIterator's</code><code>value&#95;type</code> is equality comparable to type <code>T</code>. 
* **`T`** is a model of <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">EqualityComparable</a>.

**Function Parameters**:
* **`exec`** The execution policy to use for parallelization. 
* **`first`** Beginning of the sequence to search. 
* **`last`** End of the sequence to search. 
* **`value`** The value to find. 

**Returns**:
The first iterator <code>i</code> such that <code>&#42;i == value</code> or <code>last</code>.

**See**:
* find_if 
* mismatch 

<h3 id="function-find">
Function <code>thrust::find</code>
</h3>

<code class="doxybook">
<span>template &lt;typename InputIterator,</span>
<span>&nbsp;&nbsp;typename T&gt;</span>
<span>InputIterator </span><span><b>find</b>(InputIterator first,</span>
<span>&nbsp;&nbsp;InputIterator last,</span>
<span>&nbsp;&nbsp;const T & value);</span></code>
<code>find</code> returns the first iterator <code>i</code> in the range <code>[first, last)</code> such that <code>&#42;i == value</code> or <code>last</code> if no such iterator exists.



```cpp
#include <thrust/find.h>
#include <thrust/device_vector.h>
...
thrust::device_vector<int> input(4);

input[0] = 0;
input[1] = 5;
input[2] = 3;
input[3] = 7;

thrust::device_vector<int>::iterator iter;

iter = thrust::find(input.begin(), input.end(), 3); // returns input.first() + 2
iter = thrust::find(input.begin(), input.end(), 5); // returns input.first() + 1
iter = thrust::find(input.begin(), input.end(), 9); // returns input.end()
```

**Template Parameters**:
* **`InputIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a> and <code>InputIterator's</code><code>value&#95;type</code> is equality comparable to type <code>T</code>. 
* **`T`** is a model of <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">EqualityComparable</a>.

**Function Parameters**:
* **`first`** Beginning of the sequence to search. 
* **`last`** End of the sequence to search. 
* **`value`** The value to find. 

**Returns**:
The first iterator <code>i</code> such that <code>&#42;i == value</code> or <code>last</code>.

**See**:
* find_if 
* mismatch 

<h3 id="function-find-if">
Function <code>thrust::find&#95;if</code>
</h3>

<code class="doxybook">
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename InputIterator,</span>
<span>&nbsp;&nbsp;typename Predicate&gt;</span>
<span>__host__ __device__ InputIterator </span><span><b>find_if</b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;InputIterator first,</span>
<span>&nbsp;&nbsp;InputIterator last,</span>
<span>&nbsp;&nbsp;Predicate pred);</span></code>
<code>find&#95;if</code> returns the first iterator <code>i</code> in the range <code>[first, last)</code> such that <code>pred(&#42;i)</code> is <code>true</code> or <code>last</code> if no such iterator exists.

The algorithm's execution is parallelized as determined by <code>exec</code>.



```cpp
#include <thrust/find.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
...

struct greater_than_four
{
  __host__ __device__
  bool operator()(int x)
  {
    return x > 4;
  }
};

struct greater_than_ten
{
  __host__ __device__
  bool operator()(int x)
  {
    return x > 10;
  }
};

...
thrust::device_vector<int> input(4);

input[0] = 0;
input[1] = 5;
input[2] = 3;
input[3] = 7;

thrust::device_vector<int>::iterator iter;

iter = thrust::find_if(thrust::device, input.begin(), input.end(), greater_than_four()); // returns input.first() + 1

iter = thrust::find_if(thrust::device, input.begin(), input.end(), greater_than_ten());  // returns input.end()
```

**Template Parameters**:
* **`DerivedPolicy`** The name of the derived execution policy. 
* **`InputIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>. 
* **`Predicate`** is a model of <a href="https://en.cppreference.com/w/cpp/concepts/predicate">Predicate</a>.

**Function Parameters**:
* **`exec`** The execution policy to use for parallelization. 
* **`first`** Beginning of the sequence to search. 
* **`last`** End of the sequence to search. 
* **`pred`** A predicate used to test range elements. 

**Returns**:
The first iterator <code>i</code> such that <code>pred(&#42;i)</code> is <code>true</code>, or <code>last</code>.

**See**:
* find 
* find_if_not 
* mismatch 

<h3 id="function-find-if">
Function <code>thrust::find&#95;if</code>
</h3>

<code class="doxybook">
<span>template &lt;typename InputIterator,</span>
<span>&nbsp;&nbsp;typename Predicate&gt;</span>
<span>InputIterator </span><span><b>find_if</b>(InputIterator first,</span>
<span>&nbsp;&nbsp;InputIterator last,</span>
<span>&nbsp;&nbsp;Predicate pred);</span></code>
<code>find&#95;if</code> returns the first iterator <code>i</code> in the range <code>[first, last)</code> such that <code>pred(&#42;i)</code> is <code>true</code> or <code>last</code> if no such iterator exists.



```cpp
#include <thrust/find.h>
#include <thrust/device_vector.h>

struct greater_than_four
{
  __host__ __device__
  bool operator()(int x)
  {
    return x > 4;
  }
};

struct greater_than_ten
{
  __host__ __device__
  bool operator()(int x)
  {
    return x > 10;
  }
};

...
thrust::device_vector<int> input(4);

input[0] = 0;
input[1] = 5;
input[2] = 3;
input[3] = 7;

thrust::device_vector<int>::iterator iter;

iter = thrust::find_if(input.begin(), input.end(), greater_than_four()); // returns input.first() + 1

iter = thrust::find_if(input.begin(), input.end(), greater_than_ten());  // returns input.end()
```

**Template Parameters**:
* **`InputIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>. 
* **`Predicate`** is a model of <a href="https://en.cppreference.com/w/cpp/concepts/predicate">Predicate</a>.

**Function Parameters**:
* **`first`** Beginning of the sequence to search. 
* **`last`** End of the sequence to search. 
* **`pred`** A predicate used to test range elements. 

**Returns**:
The first iterator <code>i</code> such that <code>pred(&#42;i)</code> is <code>true</code>, or <code>last</code>.

**See**:
* find 
* find_if_not 
* mismatch 

<h3 id="function-find-if-not">
Function <code>thrust::find&#95;if&#95;not</code>
</h3>

<code class="doxybook">
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename InputIterator,</span>
<span>&nbsp;&nbsp;typename Predicate&gt;</span>
<span>__host__ __device__ InputIterator </span><span><b>find_if_not</b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;InputIterator first,</span>
<span>&nbsp;&nbsp;InputIterator last,</span>
<span>&nbsp;&nbsp;Predicate pred);</span></code>
<code>find&#95;if&#95;not</code> returns the first iterator <code>i</code> in the range <code>[first, last)</code> such that <code>pred(&#42;i)</code> is <code>false</code> or <code>last</code> if no such iterator exists.

The algorithm's execution is parallelized as determined by <code>exec</code>.



```cpp
#include <thrust/find.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
...

struct greater_than_four
{
  __host__ __device__
  bool operator()(int x)
  {
    return x > 4;
  }
};

struct greater_than_ten
{
  __host__ __device__
  bool operator()(int x)
  {
    return x > 10;
  }
};

...
thrust::device_vector<int> input(4);

input[0] = 0;
input[1] = 5;
input[2] = 3;
input[3] = 7;

thrust::device_vector<int>::iterator iter;

iter = thrust::find_if_not(thrust::device, input.begin(), input.end(), greater_than_four()); // returns input.first()

iter = thrust::find_if_not(thrust::device, input.begin(), input.end(), greater_than_ten());  // returns input.first()
```

**Template Parameters**:
* **`DerivedPolicy`** The name of the derived execution policy. 
* **`InputIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>. 
* **`Predicate`** is a model of <a href="https://en.cppreference.com/w/cpp/concepts/predicate">Predicate</a>.

**Function Parameters**:
* **`exec`** The execution policy to use for parallelization. 
* **`first`** Beginning of the sequence to search. 
* **`last`** End of the sequence to search. 
* **`pred`** A predicate used to test range elements. 

**Returns**:
The first iterator <code>i</code> such that <code>pred(&#42;i)</code> is <code>false</code>, or <code>last</code>.

**See**:
* find 
* find_if 
* mismatch 

<h3 id="function-find-if-not">
Function <code>thrust::find&#95;if&#95;not</code>
</h3>

<code class="doxybook">
<span>template &lt;typename InputIterator,</span>
<span>&nbsp;&nbsp;typename Predicate&gt;</span>
<span>InputIterator </span><span><b>find_if_not</b>(InputIterator first,</span>
<span>&nbsp;&nbsp;InputIterator last,</span>
<span>&nbsp;&nbsp;Predicate pred);</span></code>
<code>find&#95;if&#95;not</code> returns the first iterator <code>i</code> in the range <code>[first, last)</code> such that <code>pred(&#42;i)</code> is <code>false</code> or <code>last</code> if no such iterator exists.



```cpp
#include <thrust/find.h>
#include <thrust/device_vector.h>

struct greater_than_four
{
  __host__ __device__
  bool operator()(int x)
  {
    return x > 4;
  }
};

struct greater_than_ten
{
  __host__ __device__
  bool operator()(int x)
  {
    return x > 10;
  }
};

...
thrust::device_vector<int> input(4);

input[0] = 0;
input[1] = 5;
input[2] = 3;
input[3] = 7;

thrust::device_vector<int>::iterator iter;

iter = thrust::find_if_not(input.begin(), input.end(), greater_than_four()); // returns input.first()

iter = thrust::find_if_not(input.begin(), input.end(), greater_than_ten());  // returns input.first()
```

**Template Parameters**:
* **`InputIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>. 
* **`Predicate`** is a model of <a href="https://en.cppreference.com/w/cpp/concepts/predicate">Predicate</a>.

**Function Parameters**:
* **`first`** Beginning of the sequence to search. 
* **`last`** End of the sequence to search. 
* **`pred`** A predicate used to test range elements. 

**Returns**:
The first iterator <code>i</code> such that <code>pred(&#42;i)</code> is <code>false</code>, or <code>last</code>.

**See**:
* find 
* find_if 
* mismatch 

<h3 id="function-mismatch">
Function <code>thrust::mismatch</code>
</h3>

<code class="doxybook">
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename InputIterator1,</span>
<span>&nbsp;&nbsp;typename InputIterator2&gt;</span>
<span>__host__ __device__ <a href="{{ site.baseurl }}/api/classes/structthrust_1_1pair.html">thrust::pair</a>< InputIterator1, InputIterator2 > </span><span><b>mismatch</b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;InputIterator1 first1,</span>
<span>&nbsp;&nbsp;InputIterator1 last1,</span>
<span>&nbsp;&nbsp;InputIterator2 first2);</span></code>
<code>mismatch</code> finds the first position where the two ranges <code>[first1, last1)</code> and <code>[first2, first2 + (last1 - first1))</code> differ. The two versions of <code>mismatch</code> use different tests for whether elements differ.

This version of <code>mismatch</code> finds the first iterator <code>i</code> in <code>[first1, last1)</code> such that <code>&#42;i == &#42;(first2 + (i - first1))</code> is <code>false</code>. The return value is a <code>pair</code> whose first element is <code>i</code> and whose second element is <code>&#42;(first2 + (i - first1))</code>. If no such iterator <code>i</code> exists, the return value is a <code>pair</code> whose first element is <code>last1</code> and whose second element is <code>&#42;(first2 + (last1 - first1))</code>.

The algorithm's execution is parallelized as determined by <code>exec</code>.



```cpp
#include <thrust/mismatch.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
...
thrust::device_vector<int> vec1(4);
thrust::device_vector<int> vec2(4);

vec1[0] = 0;  vec2[0] = 0; 
vec1[1] = 5;  vec2[1] = 5;
vec1[2] = 3;  vec2[2] = 8;
vec1[3] = 7;  vec2[3] = 7;

typedef thrust::device_vector<int>::iterator Iterator;
thrust::pair<Iterator,Iterator> result;

result = thrust::mismatch(thrust::device, vec1.begin(), vec1.end(), vec2.begin());

// result.first  is vec1.begin() + 2
// result.second is vec2.begin() + 2
```

**Template Parameters**:
* **`DerivedPolicy`** The name of the derived execution policy. 
* **`InputIterator1`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a> and <code>InputIterator1's</code><code>value&#95;type</code> is equality comparable to <code>InputIterator2's</code><code>value&#95;type</code>. 
* **`InputIterator2`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>.

**Function Parameters**:
* **`exec`** The execution policy to use for parallelization. 
* **`first1`** The beginning of the first sequence. 
* **`last1`** The end of the first sequence. 
* **`first2`** The beginning of the second sequence. 

**Returns**:
The first position where the sequences differ.

**See**:
* find 
* find_if 

<h3 id="function-mismatch">
Function <code>thrust::mismatch</code>
</h3>

<code class="doxybook">
<span>template &lt;typename InputIterator1,</span>
<span>&nbsp;&nbsp;typename InputIterator2&gt;</span>
<span><a href="{{ site.baseurl }}/api/classes/structthrust_1_1pair.html">thrust::pair</a>< InputIterator1, InputIterator2 > </span><span><b>mismatch</b>(InputIterator1 first1,</span>
<span>&nbsp;&nbsp;InputIterator1 last1,</span>
<span>&nbsp;&nbsp;InputIterator2 first2);</span></code>
<code>mismatch</code> finds the first position where the two ranges <code>[first1, last1)</code> and <code>[first2, first2 + (last1 - first1))</code> differ. The two versions of <code>mismatch</code> use different tests for whether elements differ.

This version of <code>mismatch</code> finds the first iterator <code>i</code> in <code>[first1, last1)</code> such that <code>&#42;i == &#42;(first2 + (i - first1))</code> is <code>false</code>. The return value is a <code>pair</code> whose first element is <code>i</code> and whose second element is <code>&#42;(first2 + (i - first1))</code>. If no such iterator <code>i</code> exists, the return value is a <code>pair</code> whose first element is <code>last1</code> and whose second element is <code>&#42;(first2 + (last1 - first1))</code>.



```cpp
#include <thrust/mismatch.h>
#include <thrust/device_vector.h>
...
thrust::device_vector<int> vec1(4);
thrust::device_vector<int> vec2(4);

vec1[0] = 0;  vec2[0] = 0; 
vec1[1] = 5;  vec2[1] = 5;
vec1[2] = 3;  vec2[2] = 8;
vec1[3] = 7;  vec2[3] = 7;

typedef thrust::device_vector<int>::iterator Iterator;
thrust::pair<Iterator,Iterator> result;

result = thrust::mismatch(vec1.begin(), vec1.end(), vec2.begin());

// result.first  is vec1.begin() + 2
// result.second is vec2.begin() + 2
```

**Template Parameters**:
* **`InputIterator1`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a> and <code>InputIterator1's</code><code>value&#95;type</code> is equality comparable to <code>InputIterator2's</code><code>value&#95;type</code>. 
* **`InputIterator2`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>.

**Function Parameters**:
* **`first1`** The beginning of the first sequence. 
* **`last1`** The end of the first sequence. 
* **`first2`** The beginning of the second sequence. 

**Returns**:
The first position where the sequences differ.

**See**:
* find 
* find_if 

<h3 id="function-mismatch">
Function <code>thrust::mismatch</code>
</h3>

<code class="doxybook">
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename InputIterator1,</span>
<span>&nbsp;&nbsp;typename InputIterator2,</span>
<span>&nbsp;&nbsp;typename BinaryPredicate&gt;</span>
<span>__host__ __device__ <a href="{{ site.baseurl }}/api/classes/structthrust_1_1pair.html">thrust::pair</a>< InputIterator1, InputIterator2 > </span><span><b>mismatch</b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;InputIterator1 first1,</span>
<span>&nbsp;&nbsp;InputIterator1 last1,</span>
<span>&nbsp;&nbsp;InputIterator2 first2,</span>
<span>&nbsp;&nbsp;BinaryPredicate pred);</span></code>
<code>mismatch</code> finds the first position where the two ranges <code>[first1, last1)</code> and <code>[first2, first2 + (last1 - first1))</code> differ. The two versions of <code>mismatch</code> use different tests for whether elements differ.

This version of <code>mismatch</code> finds the first iterator <code>i</code> in <code>[first1, last1)</code> such that <code>pred(&#42;i, &#42;(first2 + (i - first1))</code> is <code>false</code>. The return value is a <code>pair</code> whose first element is <code>i</code> and whose second element is <code>&#42;(first2 + (i - first1))</code>. If no such iterator <code>i</code> exists, the return value is a <code>pair</code> whose first element is <code>last1</code> and whose second element is <code>&#42;(first2 + (last1 - first1))</code>.

The algorithm's execution is parallelized as determined by <code>exec</code>.



```cpp
#include <thrust/mismatch.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
...
thrust::device_vector<int> vec1(4);
thrust::device_vector<int> vec2(4);

vec1[0] = 0;  vec2[0] = 0; 
vec1[1] = 5;  vec2[1] = 5;
vec1[2] = 3;  vec2[2] = 8;
vec1[3] = 7;  vec2[3] = 7;

typedef thrust::device_vector<int>::iterator Iterator;
thrust::pair<Iterator,Iterator> result;

result = thrust::mismatch(thrust::device, vec1.begin(), vec1.end(), vec2.begin(), thrust::equal_to<int>());

// result.first  is vec1.begin() + 2
// result.second is vec2.begin() + 2
```

**Template Parameters**:
* **`DerivedPolicy`** The name of the derived execution policy. 
* **`InputIterator1`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>. 
* **`InputIterator2`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>. 
* **`Predicate`** is a model of <a href="https://en.cppreference.com/w/cpp/named_req/BinaryPredicate">Input Iterator</a>.

**Function Parameters**:
* **`exec`** The execution policy to use for parallelization. 
* **`first1`** The beginning of the first sequence. 
* **`last1`** The end of the first sequence. 
* **`first2`** The beginning of the second sequence. 
* **`pred`** The binary predicate to compare elements. 

**Returns**:
The first position where the sequences differ.

**See**:
* find 
* find_if 

<h3 id="function-mismatch">
Function <code>thrust::mismatch</code>
</h3>

<code class="doxybook">
<span>template &lt;typename InputIterator1,</span>
<span>&nbsp;&nbsp;typename InputIterator2,</span>
<span>&nbsp;&nbsp;typename BinaryPredicate&gt;</span>
<span><a href="{{ site.baseurl }}/api/classes/structthrust_1_1pair.html">thrust::pair</a>< InputIterator1, InputIterator2 > </span><span><b>mismatch</b>(InputIterator1 first1,</span>
<span>&nbsp;&nbsp;InputIterator1 last1,</span>
<span>&nbsp;&nbsp;InputIterator2 first2,</span>
<span>&nbsp;&nbsp;BinaryPredicate pred);</span></code>
<code>mismatch</code> finds the first position where the two ranges <code>[first1, last1)</code> and <code>[first2, first2 + (last1 - first1))</code> differ. The two versions of <code>mismatch</code> use different tests for whether elements differ.

This version of <code>mismatch</code> finds the first iterator <code>i</code> in <code>[first1, last1)</code> such that <code>pred(&#42;i, &#42;(first2 + (i - first1))</code> is <code>false</code>. The return value is a <code>pair</code> whose first element is <code>i</code> and whose second element is <code>&#42;(first2 + (i - first1))</code>. If no such iterator <code>i</code> exists, the return value is a <code>pair</code> whose first element is <code>last1</code> and whose second element is <code>&#42;(first2 + (last1 - first1))</code>.



```cpp
#include <thrust/mismatch.h>
#include <thrust/device_vector.h>
...
thrust::device_vector<int> vec1(4);
thrust::device_vector<int> vec2(4);

vec1[0] = 0;  vec2[0] = 0; 
vec1[1] = 5;  vec2[1] = 5;
vec1[2] = 3;  vec2[2] = 8;
vec1[3] = 7;  vec2[3] = 7;

typedef thrust::device_vector<int>::iterator Iterator;
thrust::pair<Iterator,Iterator> result;

result = thrust::mismatch(vec1.begin(), vec1.end(), vec2.begin(), thrust::equal_to<int>());

// result.first  is vec1.begin() + 2
// result.second is vec2.begin() + 2
```

**Template Parameters**:
* **`InputIterator1`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>. 
* **`InputIterator2`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>. 
* **`Predicate`** is a model of <a href="https://en.cppreference.com/w/cpp/named_req/BinaryPredicate">Input Iterator</a>.

**Function Parameters**:
* **`first1`** The beginning of the first sequence. 
* **`last1`** The end of the first sequence. 
* **`first2`** The beginning of the second sequence. 
* **`pred`** The binary predicate to compare elements. 

**Returns**:
The first position where the sequences differ.

**See**:
* find 
* find_if 

<h3 id="function-partition-point">
Function <code>thrust::partition&#95;point</code>
</h3>

<code class="doxybook">
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename ForwardIterator,</span>
<span>&nbsp;&nbsp;typename Predicate&gt;</span>
<span>__host__ __device__ ForwardIterator </span><span><b>partition_point</b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;ForwardIterator first,</span>
<span>&nbsp;&nbsp;ForwardIterator last,</span>
<span>&nbsp;&nbsp;Predicate pred);</span></code>
<code>partition&#95;point</code> returns an iterator pointing to the end of the true partition of a partitioned range. <code>partition&#95;point</code> requires the input range <code>[first,last)</code> to be a partition; that is, all elements which satisfy <code>pred</code> shall appear before those that do not.

The algorithm's execution is parallelized as determined by <code>exec</code>.



```cpp
#include <thrust/partition.h>
#include <thrust/execution_policy.h>

struct is_even
{
  __host__ __device__
  bool operator()(const int &x)
  {
    return (x % 2) == 0;
  }
};

...

int A[] = {2, 4, 6, 8, 10, 1, 3, 5, 7, 9};
int * B = thrust::partition_point(thrust::host, A, A + 10, is_even());
// B - A is 5
// [A, B) contains only even values
```

**Note**:
Though similar, <code>partition&#95;point</code> is not redundant with <code>find&#95;if&#95;not</code>. <code>partition&#95;point's</code> precondition provides an opportunity for a faster implemention.

**Template Parameters**:
* **`DerivedPolicy`** The name of the derived execution policy. 
* **`ForwardIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a>, and <code>ForwardIterator's</code><code>value&#95;type</code> is convertible to <code>Predicate's</code><code>argument&#95;type</code>. 
* **`Predicate`** is a model of <a href="https://en.cppreference.com/w/cpp/concepts/predicate">Predicate</a>.

**Function Parameters**:
* **`exec`** The execution policy to use for parallelization. 
* **`first`** The beginning of the range to consider. 
* **`last`** The end of the range to consider. 
* **`pred`** A function object which decides to which partition each element of the range <code>[first, last)</code> belongs. 

**Preconditions**:
The range <code>[first, last)</code> shall be partitioned by <code>pred</code>.

**Returns**:
An iterator <code>mid</code> such that <code>all&#95;of(first, mid, pred)</code> and <code>none&#95;of(mid, last, pred)</code> are both true.

**See**:
* <code>partition</code>
* <code>find&#95;if&#95;not</code>

<h3 id="function-partition-point">
Function <code>thrust::partition&#95;point</code>
</h3>

<code class="doxybook">
<span>template &lt;typename ForwardIterator,</span>
<span>&nbsp;&nbsp;typename Predicate&gt;</span>
<span>ForwardIterator </span><span><b>partition_point</b>(ForwardIterator first,</span>
<span>&nbsp;&nbsp;ForwardIterator last,</span>
<span>&nbsp;&nbsp;Predicate pred);</span></code>
<code>partition&#95;point</code> returns an iterator pointing to the end of the true partition of a partitioned range. <code>partition&#95;point</code> requires the input range <code>[first,last)</code> to be a partition; that is, all elements which satisfy <code>pred</code> shall appear before those that do not. 

```cpp
#include <thrust/partition.h>

struct is_even
{
  __host__ __device__
  bool operator()(const int &x)
  {
    return (x % 2) == 0;
  }
};

...

int A[] = {2, 4, 6, 8, 10, 1, 3, 5, 7, 9};
int * B = thrust::partition_point(A, A + 10, is_even());
// B - A is 5
// [A, B) contains only even values
```

**Note**:
Though similar, <code>partition&#95;point</code> is not redundant with <code>find&#95;if&#95;not</code>. <code>partition&#95;point's</code> precondition provides an opportunity for a faster implemention.

**Template Parameters**:
* **`ForwardIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a>, and <code>ForwardIterator's</code><code>value&#95;type</code> is convertible to <code>Predicate's</code><code>argument&#95;type</code>. 
* **`Predicate`** is a model of <a href="https://en.cppreference.com/w/cpp/concepts/predicate">Predicate</a>.

**Function Parameters**:
* **`first`** The beginning of the range to consider. 
* **`last`** The end of the range to consider. 
* **`pred`** A function object which decides to which partition each element of the range <code>[first, last)</code> belongs. 

**Preconditions**:
The range <code>[first, last)</code> shall be partitioned by <code>pred</code>.

**Returns**:
An iterator <code>mid</code> such that <code>all&#95;of(first, mid, pred)</code> and <code>none&#95;of(mid, last, pred)</code> are both true.

**See**:
* <code>partition</code>
* <code>find&#95;if&#95;not</code>


