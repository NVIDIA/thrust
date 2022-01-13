---
title: Replacing
parent: Transformations
grand_parent: Algorithms
nav_exclude: false
has_children: true
has_toc: false
---

# Replacing

<code class="doxybook">
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename ForwardIterator,</span>
<span>&nbsp;&nbsp;typename T&gt;</span>
<span>__host__ __device__ void </span><span><b><a href="{{ site.baseurl }}/api/groups/group__replacing.html#function-replace">thrust::replace</a></b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;ForwardIterator first,</span>
<span>&nbsp;&nbsp;ForwardIterator last,</span>
<span>&nbsp;&nbsp;const T & old_value,</span>
<span>&nbsp;&nbsp;const T & new_value);</span>
<br>
<span>template &lt;typename ForwardIterator,</span>
<span>&nbsp;&nbsp;typename T&gt;</span>
<span>void </span><span><b><a href="{{ site.baseurl }}/api/groups/group__replacing.html#function-replace">thrust::replace</a></b>(ForwardIterator first,</span>
<span>&nbsp;&nbsp;ForwardIterator last,</span>
<span>&nbsp;&nbsp;const T & old_value,</span>
<span>&nbsp;&nbsp;const T & new_value);</span>
<br>
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename ForwardIterator,</span>
<span>&nbsp;&nbsp;typename Predicate,</span>
<span>&nbsp;&nbsp;typename T&gt;</span>
<span>__host__ __device__ void </span><span><b><a href="{{ site.baseurl }}/api/groups/group__replacing.html#function-replace-if">thrust::replace&#95;if</a></b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;ForwardIterator first,</span>
<span>&nbsp;&nbsp;ForwardIterator last,</span>
<span>&nbsp;&nbsp;Predicate pred,</span>
<span>&nbsp;&nbsp;const T & new_value);</span>
<br>
<span>template &lt;typename ForwardIterator,</span>
<span>&nbsp;&nbsp;typename Predicate,</span>
<span>&nbsp;&nbsp;typename T&gt;</span>
<span>void </span><span><b><a href="{{ site.baseurl }}/api/groups/group__replacing.html#function-replace-if">thrust::replace&#95;if</a></b>(ForwardIterator first,</span>
<span>&nbsp;&nbsp;ForwardIterator last,</span>
<span>&nbsp;&nbsp;Predicate pred,</span>
<span>&nbsp;&nbsp;const T & new_value);</span>
<br>
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename ForwardIterator,</span>
<span>&nbsp;&nbsp;typename InputIterator,</span>
<span>&nbsp;&nbsp;typename Predicate,</span>
<span>&nbsp;&nbsp;typename T&gt;</span>
<span>__host__ __device__ void </span><span><b><a href="{{ site.baseurl }}/api/groups/group__replacing.html#function-replace-if">thrust::replace&#95;if</a></b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;ForwardIterator first,</span>
<span>&nbsp;&nbsp;ForwardIterator last,</span>
<span>&nbsp;&nbsp;InputIterator stencil,</span>
<span>&nbsp;&nbsp;Predicate pred,</span>
<span>&nbsp;&nbsp;const T & new_value);</span>
<br>
<span>template &lt;typename ForwardIterator,</span>
<span>&nbsp;&nbsp;typename InputIterator,</span>
<span>&nbsp;&nbsp;typename Predicate,</span>
<span>&nbsp;&nbsp;typename T&gt;</span>
<span>void </span><span><b><a href="{{ site.baseurl }}/api/groups/group__replacing.html#function-replace-if">thrust::replace&#95;if</a></b>(ForwardIterator first,</span>
<span>&nbsp;&nbsp;ForwardIterator last,</span>
<span>&nbsp;&nbsp;InputIterator stencil,</span>
<span>&nbsp;&nbsp;Predicate pred,</span>
<span>&nbsp;&nbsp;const T & new_value);</span>
<br>
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename InputIterator,</span>
<span>&nbsp;&nbsp;typename OutputIterator,</span>
<span>&nbsp;&nbsp;typename T&gt;</span>
<span>__host__ __device__ OutputIterator </span><span><b><a href="{{ site.baseurl }}/api/groups/group__replacing.html#function-replace-copy">thrust::replace&#95;copy</a></b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;InputIterator first,</span>
<span>&nbsp;&nbsp;InputIterator last,</span>
<span>&nbsp;&nbsp;OutputIterator result,</span>
<span>&nbsp;&nbsp;const T & old_value,</span>
<span>&nbsp;&nbsp;const T & new_value);</span>
<br>
<span>template &lt;typename InputIterator,</span>
<span>&nbsp;&nbsp;typename OutputIterator,</span>
<span>&nbsp;&nbsp;typename T&gt;</span>
<span>OutputIterator </span><span><b><a href="{{ site.baseurl }}/api/groups/group__replacing.html#function-replace-copy">thrust::replace&#95;copy</a></b>(InputIterator first,</span>
<span>&nbsp;&nbsp;InputIterator last,</span>
<span>&nbsp;&nbsp;OutputIterator result,</span>
<span>&nbsp;&nbsp;const T & old_value,</span>
<span>&nbsp;&nbsp;const T & new_value);</span>
<br>
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename InputIterator,</span>
<span>&nbsp;&nbsp;typename OutputIterator,</span>
<span>&nbsp;&nbsp;typename Predicate,</span>
<span>&nbsp;&nbsp;typename T&gt;</span>
<span>__host__ __device__ OutputIterator </span><span><b><a href="{{ site.baseurl }}/api/groups/group__replacing.html#function-replace-copy-if">thrust::replace&#95;copy&#95;if</a></b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;InputIterator first,</span>
<span>&nbsp;&nbsp;InputIterator last,</span>
<span>&nbsp;&nbsp;OutputIterator result,</span>
<span>&nbsp;&nbsp;Predicate pred,</span>
<span>&nbsp;&nbsp;const T & new_value);</span>
<br>
<span>template &lt;typename InputIterator,</span>
<span>&nbsp;&nbsp;typename OutputIterator,</span>
<span>&nbsp;&nbsp;typename Predicate,</span>
<span>&nbsp;&nbsp;typename T&gt;</span>
<span>OutputIterator </span><span><b><a href="{{ site.baseurl }}/api/groups/group__replacing.html#function-replace-copy-if">thrust::replace&#95;copy&#95;if</a></b>(InputIterator first,</span>
<span>&nbsp;&nbsp;InputIterator last,</span>
<span>&nbsp;&nbsp;OutputIterator result,</span>
<span>&nbsp;&nbsp;Predicate pred,</span>
<span>&nbsp;&nbsp;const T & new_value);</span>
<br>
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename InputIterator1,</span>
<span>&nbsp;&nbsp;typename InputIterator2,</span>
<span>&nbsp;&nbsp;typename OutputIterator,</span>
<span>&nbsp;&nbsp;typename Predicate,</span>
<span>&nbsp;&nbsp;typename T&gt;</span>
<span>__host__ __device__ OutputIterator </span><span><b><a href="{{ site.baseurl }}/api/groups/group__replacing.html#function-replace-copy-if">thrust::replace&#95;copy&#95;if</a></b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;InputIterator1 first,</span>
<span>&nbsp;&nbsp;InputIterator1 last,</span>
<span>&nbsp;&nbsp;InputIterator2 stencil,</span>
<span>&nbsp;&nbsp;OutputIterator result,</span>
<span>&nbsp;&nbsp;Predicate pred,</span>
<span>&nbsp;&nbsp;const T & new_value);</span>
<br>
<span>template &lt;typename InputIterator1,</span>
<span>&nbsp;&nbsp;typename InputIterator2,</span>
<span>&nbsp;&nbsp;typename OutputIterator,</span>
<span>&nbsp;&nbsp;typename Predicate,</span>
<span>&nbsp;&nbsp;typename T&gt;</span>
<span>OutputIterator </span><span><b><a href="{{ site.baseurl }}/api/groups/group__replacing.html#function-replace-copy-if">thrust::replace&#95;copy&#95;if</a></b>(InputIterator1 first,</span>
<span>&nbsp;&nbsp;InputIterator1 last,</span>
<span>&nbsp;&nbsp;InputIterator2 stencil,</span>
<span>&nbsp;&nbsp;OutputIterator result,</span>
<span>&nbsp;&nbsp;Predicate pred,</span>
<span>&nbsp;&nbsp;const T & new_value);</span>
</code>

## Functions

<h3 id="function-replace">
Function <code>thrust::replace</code>
</h3>

<code class="doxybook">
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename ForwardIterator,</span>
<span>&nbsp;&nbsp;typename T&gt;</span>
<span>__host__ __device__ void </span><span><b>replace</b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;ForwardIterator first,</span>
<span>&nbsp;&nbsp;ForwardIterator last,</span>
<span>&nbsp;&nbsp;const T & old_value,</span>
<span>&nbsp;&nbsp;const T & new_value);</span></code>
<code>replace</code> replaces every element in the range [first, last) equal to <code>old&#95;value</code> with <code>new&#95;value</code>. That is: for every iterator <code>i</code>, if <code>&#42;i == old&#95;value</code> then it performs the <code>assignment &#42;i = new&#95;value</code>.

The algorithm's execution is parallelized as determined by <code>exec</code>.


The following code snippet demonstrates how to use <code>replace</code> to replace a value of interest in a <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1device__vector.html">device&#95;vector</a></code> with another using the <code>thrust::device</code> execution policy for parallelization:



```cpp
#include <thrust/replace.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>

...

thrust::device_vector<int> A(4);
A[0] = 1;
A[1] = 2;
A[2] = 3;
A[3] = 1;

thrust::replace(thrust::device, A.begin(), A.end(), 1, 99);

// A contains [99, 2, 3, 99]
```

**Template Parameters**:
* **`DerivedPolicy`** The name of the derived execution policy. 
* **`ForwardIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a>, and <code>ForwardIterator</code> is mutable. 
* **`T`** is a model of <a href="https://en.cppreference.com/w/cpp/named_req/CopyAssignable>Assignable">Assignable</a>, <code>T</code> is a model of <a href="https://en.cppreference.com/w/cpp/concepts/equality_comparable">EqualityComparable</a>, objects of <code>T</code> may be compared for equality with objects of <code>ForwardIterator's</code><code>value&#95;type</code>, and <code>T</code> is convertible to <code>ForwardIterator's</code><code>value&#95;type</code>.

**Function Parameters**:
* **`exec`** The execution policy to use for parallelization. 
* **`first`** The beginning of the sequence of interest. 
* **`last`** The end of the sequence of interest. 
* **`old_value`** The value to replace. 
* **`new_value`** The new value to replace <code>old&#95;value</code>.

**See**:
* <a href="https://en.cppreference.com/w/cpp/algorithm/replace">https://en.cppreference.com/w/cpp/algorithm/replace</a>
* <code>replace&#95;if</code>
* <code>replace&#95;copy</code>
* <code>replace&#95;copy&#95;if</code>

<h3 id="function-replace">
Function <code>thrust::replace</code>
</h3>

<code class="doxybook">
<span>template &lt;typename ForwardIterator,</span>
<span>&nbsp;&nbsp;typename T&gt;</span>
<span>void </span><span><b>replace</b>(ForwardIterator first,</span>
<span>&nbsp;&nbsp;ForwardIterator last,</span>
<span>&nbsp;&nbsp;const T & old_value,</span>
<span>&nbsp;&nbsp;const T & new_value);</span></code>
<code>replace</code> replaces every element in the range [first, last) equal to <code>old&#95;value</code> with <code>new&#95;value</code>. That is: for every iterator <code>i</code>, if <code>&#42;i == old&#95;value</code> then it performs the <code>assignment &#42;i = new&#95;value</code>.


The following code snippet demonstrates how to use <code>replace</code> to replace a value of interest in a <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1device__vector.html">device&#95;vector</a></code> with another.



```cpp
#include <thrust/replace.h>
#include <thrust/device_vector.h>

...

thrust::device_vector<int> A(4);
A[0] = 1;
A[1] = 2;
A[2] = 3;
A[3] = 1;

thrust::replace(A.begin(), A.end(), 1, 99);

// A contains [99, 2, 3, 99]
```

**Template Parameters**:
* **`ForwardIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a>, and <code>ForwardIterator</code> is mutable. 
* **`T`** is a model of <a href="https://en.cppreference.com/w/cpp/named_req/CopyAssignable>Assignable">Assignable</a>, <code>T</code> is a model of <a href="https://en.cppreference.com/w/cpp/concepts/equality_comparable">EqualityComparable</a>, objects of <code>T</code> may be compared for equality with objects of <code>ForwardIterator's</code><code>value&#95;type</code>, and <code>T</code> is convertible to <code>ForwardIterator's</code><code>value&#95;type</code>.

**Function Parameters**:
* **`first`** The beginning of the sequence of interest. 
* **`last`** The end of the sequence of interest. 
* **`old_value`** The value to replace. 
* **`new_value`** The new value to replace <code>old&#95;value</code>.

**See**:
* <a href="https://en.cppreference.com/w/cpp/algorithm/replace">https://en.cppreference.com/w/cpp/algorithm/replace</a>
* <code>replace&#95;if</code>
* <code>replace&#95;copy</code>
* <code>replace&#95;copy&#95;if</code>

<h3 id="function-replace-if">
Function <code>thrust::replace&#95;if</code>
</h3>

<code class="doxybook">
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename ForwardIterator,</span>
<span>&nbsp;&nbsp;typename Predicate,</span>
<span>&nbsp;&nbsp;typename T&gt;</span>
<span>__host__ __device__ void </span><span><b>replace_if</b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;ForwardIterator first,</span>
<span>&nbsp;&nbsp;ForwardIterator last,</span>
<span>&nbsp;&nbsp;Predicate pred,</span>
<span>&nbsp;&nbsp;const T & new_value);</span></code>
<code>replace&#95;if</code> replaces every element in the range <code>[first, last)</code> for which <code>pred</code> returns <code>true</code> with <code>new&#95;value</code>. That is: for every iterator <code>i</code>, if <code>pred(&#42;i)</code> is <code>true</code> then it performs the assignment <code>&#42;i = new&#95;value</code>.

The algorithm's execution is parallelized as determined by <code>exec</code>.


The following code snippet demonstrates how to use <code>replace&#95;if</code> to replace a <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1device__vector.html">device&#95;vector</a>'s</code> negative elements with <code>0</code> using the <code>thrust::device</code> execution policy for parallelization:



```cpp
#include <thrust/replace.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
...
struct is_less_than_zero
{
  __host__ __device__
  bool operator()(int x)
  {
    return x < 0;
  }
};

...

thrust::device_vector<int> A(4);
A[0] =  1;
A[1] = -3;
A[2] =  2;
A[3] = -1;

is_less_than_zero pred;

thrust::replace_if(thrust::device, A.begin(), A.end(), pred, 0);

// A contains [1, 0, 2, 0]
```

**Template Parameters**:
* **`DerivedPolicy`** The name of the derived execution policy. 
* **`ForwardIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a>, <code>ForwardIterator</code> is mutable, and <code>ForwardIterator's</code><code>value&#95;type</code> is convertible to <code>Predicate's</code><code>argument&#95;type</code>. 
* **`Predicate`** is a model of <a href="https://en.cppreference.com/w/cpp/concepts/predicate">Predicate</a>. 
* **`T`** is a model of <a href="https://en.cppreference.com/w/cpp/named_req/CopyAssignable">Assignable</a>, and <code>T</code> is convertible to <code>ForwardIterator's</code><code>value&#95;type</code>.

**Function Parameters**:
* **`exec`** The execution policy to use for parallelization. 
* **`first`** The beginning of the sequence of interest. 
* **`last`** The end of the sequence of interest. 
* **`pred`** The predicate to test on every value of the range <code>[first,last)</code>. 
* **`new_value`** The new value to replace elements which <code>pred(&#42;i)</code> evaluates to <code>true</code>.

**See**:
* <a href="https://en.cppreference.com/w/cpp/algorithm/replace">https://en.cppreference.com/w/cpp/algorithm/replace</a>
* <code>replace</code>
* <code>replace&#95;copy</code>
* <code>replace&#95;copy&#95;if</code>

<h3 id="function-replace-if">
Function <code>thrust::replace&#95;if</code>
</h3>

<code class="doxybook">
<span>template &lt;typename ForwardIterator,</span>
<span>&nbsp;&nbsp;typename Predicate,</span>
<span>&nbsp;&nbsp;typename T&gt;</span>
<span>void </span><span><b>replace_if</b>(ForwardIterator first,</span>
<span>&nbsp;&nbsp;ForwardIterator last,</span>
<span>&nbsp;&nbsp;Predicate pred,</span>
<span>&nbsp;&nbsp;const T & new_value);</span></code>
<code>replace&#95;if</code> replaces every element in the range <code>[first, last)</code> for which <code>pred</code> returns <code>true</code> with <code>new&#95;value</code>. That is: for every iterator <code>i</code>, if <code>pred(&#42;i)</code> is <code>true</code> then it performs the assignment <code>&#42;i = new&#95;value</code>.


The following code snippet demonstrates how to use <code>replace&#95;if</code> to replace a <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1device__vector.html">device&#95;vector</a>'s</code> negative elements with <code>0</code>.



```cpp
#include <thrust/replace.h>
#include <thrust/device_vector.h>
...
struct is_less_than_zero
{
  __host__ __device__
  bool operator()(int x)
  {
    return x < 0;
  }
};

...

thrust::device_vector<int> A(4);
A[0] =  1;
A[1] = -3;
A[2] =  2;
A[3] = -1;

is_less_than_zero pred;

thrust::replace_if(A.begin(), A.end(), pred, 0);

// A contains [1, 0, 2, 0]
```

**Template Parameters**:
* **`ForwardIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a>, <code>ForwardIterator</code> is mutable, and <code>ForwardIterator's</code><code>value&#95;type</code> is convertible to <code>Predicate's</code><code>argument&#95;type</code>. 
* **`Predicate`** is a model of <a href="https://en.cppreference.com/w/cpp/concepts/predicate">Predicate</a>. 
* **`T`** is a model of <a href="https://en.cppreference.com/w/cpp/named_req/CopyAssignable">Assignable</a>, and <code>T</code> is convertible to <code>ForwardIterator's</code><code>value&#95;type</code>.

**Function Parameters**:
* **`first`** The beginning of the sequence of interest. 
* **`last`** The end of the sequence of interest. 
* **`pred`** The predicate to test on every value of the range <code>[first,last)</code>. 
* **`new_value`** The new value to replace elements which <code>pred(&#42;i)</code> evaluates to <code>true</code>.

**See**:
* <a href="https://en.cppreference.com/w/cpp/algorithm/replace">https://en.cppreference.com/w/cpp/algorithm/replace</a>
* <code>replace</code>
* <code>replace&#95;copy</code>
* <code>replace&#95;copy&#95;if</code>

<h3 id="function-replace-if">
Function <code>thrust::replace&#95;if</code>
</h3>

<code class="doxybook">
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename ForwardIterator,</span>
<span>&nbsp;&nbsp;typename InputIterator,</span>
<span>&nbsp;&nbsp;typename Predicate,</span>
<span>&nbsp;&nbsp;typename T&gt;</span>
<span>__host__ __device__ void </span><span><b>replace_if</b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;ForwardIterator first,</span>
<span>&nbsp;&nbsp;ForwardIterator last,</span>
<span>&nbsp;&nbsp;InputIterator stencil,</span>
<span>&nbsp;&nbsp;Predicate pred,</span>
<span>&nbsp;&nbsp;const T & new_value);</span></code>
<code>replace&#95;if</code> replaces every element in the range <code>[first, last)</code> for which <code>pred(&#42;s)</code> returns <code>true</code> with <code>new&#95;value</code>. That is: for every iterator <code>i</code> in the range <code>[first, last)</code>, and <code>s</code> in the range <code>[stencil, stencil + (last - first))</code>, if <code>pred(&#42;s)</code> is <code>true</code> then it performs the assignment <code>&#42;i = new&#95;value</code>.

The algorithm's execution is parallelized as determined by <code>exec</code>.


The following code snippet demonstrates how to use <code>replace&#95;if</code> to replace a <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1device__vector.html">device&#95;vector</a>'s</code> element with <code>0</code> when its corresponding stencil element is less than zero using the <code>thrust::device</code> execution policy for parallelization:



```cpp
#include <thrust/replace.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>

struct is_less_than_zero
{
  __host__ __device__
  bool operator()(int x)
  {
    return x < 0;
  }
};

...

thrust::device_vector<int> A(4);
A[0] =  10;
A[1] =  20;
A[2] =  30;
A[3] =  40;

thrust::device_vector<int> S(4);
S[0] = -1;
S[1] =  0;
S[2] = -1;
S[3] =  0;

is_less_than_zero pred;
thrust::replace_if(thrust::device, A.begin(), A.end(), S.begin(), pred, 0);

// A contains [0, 20, 0, 40]
```

**Template Parameters**:
* **`DerivedPolicy`** The name of the derived execution policy. 
* **`ForwardIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a>, and <code>ForwardIterator</code> is mutable. 
* **`InputIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>, and <code>InputIterator's</code><code>value&#95;type</code> is convertible to <code>Predicate's</code><code>argument&#95;type</code>. 
* **`Predicate`** is a model of <a href="https://en.cppreference.com/w/cpp/concepts/predicate">Predicate</a>. 
* **`T`** is a model of <a href="https://en.cppreference.com/w/cpp/named_req/CopyAssignable">Assignable</a>, and <code>T</code> is convertible to <code>ForwardIterator's</code><code>value&#95;type</code>.

**Function Parameters**:
* **`exec`** The execution policy to use for parallelization. 
* **`first`** The beginning of the sequence of interest. 
* **`last`** The end of the sequence of interest. 
* **`stencil`** The beginning of the stencil sequence. 
* **`pred`** The predicate to test on every value of the range <code>[first,last)</code>. 
* **`new_value`** The new value to replace elements which <code>pred(&#42;i)</code> evaluates to <code>true</code>.

**See**:
* <a href="https://en.cppreference.com/w/cpp/algorithm/replace">https://en.cppreference.com/w/cpp/algorithm/replace</a>
* <code>replace</code>
* <code>replace&#95;copy</code>
* <code>replace&#95;copy&#95;if</code>

<h3 id="function-replace-if">
Function <code>thrust::replace&#95;if</code>
</h3>

<code class="doxybook">
<span>template &lt;typename ForwardIterator,</span>
<span>&nbsp;&nbsp;typename InputIterator,</span>
<span>&nbsp;&nbsp;typename Predicate,</span>
<span>&nbsp;&nbsp;typename T&gt;</span>
<span>void </span><span><b>replace_if</b>(ForwardIterator first,</span>
<span>&nbsp;&nbsp;ForwardIterator last,</span>
<span>&nbsp;&nbsp;InputIterator stencil,</span>
<span>&nbsp;&nbsp;Predicate pred,</span>
<span>&nbsp;&nbsp;const T & new_value);</span></code>
<code>replace&#95;if</code> replaces every element in the range <code>[first, last)</code> for which <code>pred(&#42;s)</code> returns <code>true</code> with <code>new&#95;value</code>. That is: for every iterator <code>i</code> in the range <code>[first, last)</code>, and <code>s</code> in the range <code>[stencil, stencil + (last - first))</code>, if <code>pred(&#42;s)</code> is <code>true</code> then it performs the assignment <code>&#42;i = new&#95;value</code>.


The following code snippet demonstrates how to use <code>replace&#95;if</code> to replace a <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1device__vector.html">device&#95;vector</a>'s</code> element with <code>0</code> when its corresponding stencil element is less than zero.



```cpp
#include <thrust/replace.h>
#include <thrust/device_vector.h>

struct is_less_than_zero
{
  __host__ __device__
  bool operator()(int x)
  {
    return x < 0;
  }
};

...

thrust::device_vector<int> A(4);
A[0] =  10;
A[1] =  20;
A[2] =  30;
A[3] =  40;

thrust::device_vector<int> S(4);
S[0] = -1;
S[1] =  0;
S[2] = -1;
S[3] =  0;

is_less_than_zero pred;
thrust::replace_if(A.begin(), A.end(), S.begin(), pred, 0);

// A contains [0, 20, 0, 40]
```

**Template Parameters**:
* **`ForwardIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a>, and <code>ForwardIterator</code> is mutable. 
* **`InputIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>, and <code>InputIterator's</code><code>value&#95;type</code> is convertible to <code>Predicate's</code><code>argument&#95;type</code>. 
* **`Predicate`** is a model of <a href="https://en.cppreference.com/w/cpp/concepts/predicate">Predicate</a>. 
* **`T`** is a model of <a href="https://en.cppreference.com/w/cpp/named_req/CopyAssignable">Assignable</a>, and <code>T</code> is convertible to <code>ForwardIterator's</code><code>value&#95;type</code>.

**Function Parameters**:
* **`first`** The beginning of the sequence of interest. 
* **`last`** The end of the sequence of interest. 
* **`stencil`** The beginning of the stencil sequence. 
* **`pred`** The predicate to test on every value of the range <code>[first,last)</code>. 
* **`new_value`** The new value to replace elements which <code>pred(&#42;i)</code> evaluates to <code>true</code>.

**See**:
* <a href="https://en.cppreference.com/w/cpp/algorithm/replace">https://en.cppreference.com/w/cpp/algorithm/replace</a>
* <code>replace</code>
* <code>replace&#95;copy</code>
* <code>replace&#95;copy&#95;if</code>

<h3 id="function-replace-copy">
Function <code>thrust::replace&#95;copy</code>
</h3>

<code class="doxybook">
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename InputIterator,</span>
<span>&nbsp;&nbsp;typename OutputIterator,</span>
<span>&nbsp;&nbsp;typename T&gt;</span>
<span>__host__ __device__ OutputIterator </span><span><b>replace_copy</b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;InputIterator first,</span>
<span>&nbsp;&nbsp;InputIterator last,</span>
<span>&nbsp;&nbsp;OutputIterator result,</span>
<span>&nbsp;&nbsp;const T & old_value,</span>
<span>&nbsp;&nbsp;const T & new_value);</span></code>
<code>replace&#95;copy</code> copies elements from the range <code>[first, last)</code> to the range <code>[result, result + (last-first))</code>, except that any element equal to <code>old&#95;value</code> is not copied; <code>new&#95;value</code> is copied instead.

More precisely, for every integer <code>n</code> such that <code>0 &lt;= n &lt; last-first</code>, <code>replace&#95;copy</code> performs the assignment <code>&#42;(result+n) = new&#95;value</code> if <code>&#42;(first+n) == old&#95;value</code>, and <code>&#42;(result+n) = &#42;(first+n)</code> otherwise.

The algorithm's execution is parallelized as determined by <code>exec</code>.



```cpp
#include <thrust/replace.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
...
thrust::device_vector<int> A(4);
A[0] = 1;
A[1] = 2;
A[2] = 3;
A[3] = 1;

thrust::device_vector<int> B(4);

thrust::replace_copy(thrust::device, A.begin(), A.end(), B.begin(), 1, 99);

// B contains [99, 2, 3, 99]
```

**Template Parameters**:
* **`DerivedPolicy`** The name of the derived execution policy. 
* **`InputIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>. 
* **`OutputIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>. 
* **`T`** is a model of <a href="https://en.cppreference.com/w/cpp/named_req/CopyAssignable">Assignable</a>, <code>T</code> is a model of <a href="https://en.cppreference.com/w/cpp/concepts/equality_comparable">Equality Comparable</a>, <code>T</code> may be compared for equality with <code>InputIterator's</code><code>value&#95;type</code>, and <code>T</code> is convertible to <code>OutputIterator's</code><code>value&#95;type</code>.

**Function Parameters**:
* **`exec`** The execution policy to use for parallelization. 
* **`first`** The beginning of the sequence to copy from. 
* **`last`** The end of the sequence to copy from. 
* **`result`** The beginning of the sequence to copy to. 
* **`old_value`** The value to replace. 
* **`new_value`** The replacement value for which <code>&#42;i == old&#95;value</code> evaluates to <code>true</code>. 

**Preconditions**:
<code>first</code> may equal <code>result</code>, but the ranges <code>[first, last)</code> and <code>[result, result + (last - first))</code> shall not overlap otherwise.

**Returns**:
<code>result + (last-first)</code>

**See**:
* <a href="https://en.cppreference.com/w/cpp/algorithm/replace_copy">https://en.cppreference.com/w/cpp/algorithm/replace_copy</a>
* <code>copy</code>
* <code>replace</code>
* <code>replace&#95;if</code>
* <code>replace&#95;copy&#95;if</code>

<h3 id="function-replace-copy">
Function <code>thrust::replace&#95;copy</code>
</h3>

<code class="doxybook">
<span>template &lt;typename InputIterator,</span>
<span>&nbsp;&nbsp;typename OutputIterator,</span>
<span>&nbsp;&nbsp;typename T&gt;</span>
<span>OutputIterator </span><span><b>replace_copy</b>(InputIterator first,</span>
<span>&nbsp;&nbsp;InputIterator last,</span>
<span>&nbsp;&nbsp;OutputIterator result,</span>
<span>&nbsp;&nbsp;const T & old_value,</span>
<span>&nbsp;&nbsp;const T & new_value);</span></code>
<code>replace&#95;copy</code> copies elements from the range <code>[first, last)</code> to the range <code>[result, result + (last-first))</code>, except that any element equal to <code>old&#95;value</code> is not copied; <code>new&#95;value</code> is copied instead.

More precisely, for every integer <code>n</code> such that <code>0 &lt;= n &lt; last-first</code>, <code>replace&#95;copy</code> performs the assignment <code>&#42;(result+n) = new&#95;value</code> if <code>&#42;(first+n) == old&#95;value</code>, and <code>&#42;(result+n) = &#42;(first+n)</code> otherwise.



```cpp
#include <thrust/replace.h>
#include <thrust/device_vector.h>
...
thrust::device_vector<int> A(4);
A[0] = 1;
A[1] = 2;
A[2] = 3;
A[3] = 1;

thrust::device_vector<int> B(4);

thrust::replace_copy(A.begin(), A.end(), B.begin(), 1, 99);

// B contains [99, 2, 3, 99]
```

**Template Parameters**:
* **`InputIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>. 
* **`OutputIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>. 
* **`T`** is a model of <a href="https://en.cppreference.com/w/cpp/named_req/CopyAssignable">Assignable</a>, <code>T</code> is a model of <a href="https://en.cppreference.com/w/cpp/concepts/equality_comparable">Equality Comparable</a>, <code>T</code> may be compared for equality with <code>InputIterator's</code><code>value&#95;type</code>, and <code>T</code> is convertible to <code>OutputIterator's</code><code>value&#95;type</code>.

**Function Parameters**:
* **`first`** The beginning of the sequence to copy from. 
* **`last`** The end of the sequence to copy from. 
* **`result`** The beginning of the sequence to copy to. 
* **`old_value`** The value to replace. 
* **`new_value`** The replacement value for which <code>&#42;i == old&#95;value</code> evaluates to <code>true</code>. 

**Preconditions**:
<code>first</code> may equal <code>result</code>, but the ranges <code>[first, last)</code> and <code>[result, result + (last - first))</code> shall not overlap otherwise.

**Returns**:
<code>result + (last-first)</code>

**See**:
* <a href="https://en.cppreference.com/w/cpp/algorithm/replace_copy">https://en.cppreference.com/w/cpp/algorithm/replace_copy</a>
* <code>copy</code>
* <code>replace</code>
* <code>replace&#95;if</code>
* <code>replace&#95;copy&#95;if</code>

<h3 id="function-replace-copy-if">
Function <code>thrust::replace&#95;copy&#95;if</code>
</h3>

<code class="doxybook">
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename InputIterator,</span>
<span>&nbsp;&nbsp;typename OutputIterator,</span>
<span>&nbsp;&nbsp;typename Predicate,</span>
<span>&nbsp;&nbsp;typename T&gt;</span>
<span>__host__ __device__ OutputIterator </span><span><b>replace_copy_if</b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;InputIterator first,</span>
<span>&nbsp;&nbsp;InputIterator last,</span>
<span>&nbsp;&nbsp;OutputIterator result,</span>
<span>&nbsp;&nbsp;Predicate pred,</span>
<span>&nbsp;&nbsp;const T & new_value);</span></code>
<code>replace&#95;copy&#95;if</code> copies elements from the range <code>[first, last)</code> to the range <code>[result, result + (last-first))</code>, except that any element for which <code>pred</code> is <code>true</code> is not copied; <code>new&#95;value</code> is copied instead.

More precisely, for every integer <code>n</code> such that 0 <= n < last-first, <code>replace&#95;copy&#95;if</code> performs the assignment <code>&#42;(result+n) = new&#95;value</code> if <code>pred(&#42;(first+n))</code>, and <code>&#42;(result+n) = &#42;(first+n)</code> otherwise.

The algorithm's execution is parallelized as determined by <code>exec</code>.



```cpp
#include <thrust/replace.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>

struct is_less_than_zero
{
  __host__ __device__
  bool operator()(int x)
  {
    return x < 0;
  }
};

...

thrust::device_vector<int> A(4);
A[0] =  1;
A[1] = -3;
A[2] =  2;
A[3] = -1;

thrust::device_vector<int> B(4);
is_less_than_zero pred;

thrust::replace_copy_if(thrust::device, A.begin(), A.end(), B.begin(), pred, 0);

// B contains [1, 0, 2, 0]
```

**Template Parameters**:
* **`DerivedPolicy`** The name of the derived execution policy. 
* **`InputIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>, and <code>InputIterator's</code><code>value&#95;type</code> is convertible to <code>Predicate's</code><code>argument&#95;type</code>. 
* **`OutputIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>. 
* **`Predicate`** is a model of <a href="https://en.cppreference.com/w/cpp/concepts/predicate">Predicate</a>. 
* **`T`** is a model of <a href="https://en.cppreference.com/w/cpp/named_req/CopyAssignable">Assignable</a>, and <code>T</code> is convertible to <code>OutputIterator's</code><code>value&#95;type</code>.

**Function Parameters**:
* **`exec`** The execution policy to use for parallelization. 
* **`first`** The beginning of the sequence to copy from. 
* **`last`** The end of the sequence to copy from. 
* **`result`** The beginning of the sequence to copy to. 
* **`pred`** The predicate to test on every value of the range <code>[first,last)</code>. 
* **`new_value`** The replacement value to assign <code>pred(&#42;i)</code> evaluates to <code>true</code>. 

**Preconditions**:
<code>first</code> may equal <code>result</code>, but the ranges <code>[first, last)</code> and <code>[result, result + (last - first))</code> shall not overlap otherwise.

**Returns**:
<code>result + (last-first)</code>

**See**:
* <a href="https://en.cppreference.com/w/cpp/algorithm/replace_copy">https://en.cppreference.com/w/cpp/algorithm/replace_copy</a>
* <code>replace</code>
* <code>replace&#95;if</code>
* <code>replace&#95;copy</code>

<h3 id="function-replace-copy-if">
Function <code>thrust::replace&#95;copy&#95;if</code>
</h3>

<code class="doxybook">
<span>template &lt;typename InputIterator,</span>
<span>&nbsp;&nbsp;typename OutputIterator,</span>
<span>&nbsp;&nbsp;typename Predicate,</span>
<span>&nbsp;&nbsp;typename T&gt;</span>
<span>OutputIterator </span><span><b>replace_copy_if</b>(InputIterator first,</span>
<span>&nbsp;&nbsp;InputIterator last,</span>
<span>&nbsp;&nbsp;OutputIterator result,</span>
<span>&nbsp;&nbsp;Predicate pred,</span>
<span>&nbsp;&nbsp;const T & new_value);</span></code>
<code>replace&#95;copy&#95;if</code> copies elements from the range <code>[first, last)</code> to the range <code>[result, result + (last-first))</code>, except that any element for which <code>pred</code> is <code>true</code> is not copied; <code>new&#95;value</code> is copied instead.

More precisely, for every integer <code>n</code> such that 0 <= n < last-first, <code>replace&#95;copy&#95;if</code> performs the assignment <code>&#42;(result+n) = new&#95;value</code> if <code>pred(&#42;(first+n))</code>, and <code>&#42;(result+n) = &#42;(first+n)</code> otherwise.



```cpp
#include <thrust/replace.h>
#include <thrust/device_vector.h>

struct is_less_than_zero
{
  __host__ __device__
  bool operator()(int x)
  {
    return x < 0;
  }
};

...

thrust::device_vector<int> A(4);
A[0] =  1;
A[1] = -3;
A[2] =  2;
A[3] = -1;

thrust::device_vector<int> B(4);
is_less_than_zero pred;

thrust::replace_copy_if(A.begin(), A.end(), B.begin(), pred, 0);

// B contains [1, 0, 2, 0]
```

**Template Parameters**:
* **`InputIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>, and <code>InputIterator's</code><code>value&#95;type</code> is convertible to <code>Predicate's</code><code>argument&#95;type</code>. 
* **`OutputIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>. 
* **`Predicate`** is a model of <a href="https://en.cppreference.com/w/cpp/concepts/predicate">Predicate</a>. 
* **`T`** is a model of <a href="https://en.cppreference.com/w/cpp/named_req/CopyAssignable">Assignable</a>, and <code>T</code> is convertible to <code>OutputIterator's</code><code>value&#95;type</code>.

**Function Parameters**:
* **`first`** The beginning of the sequence to copy from. 
* **`last`** The end of the sequence to copy from. 
* **`result`** The beginning of the sequence to copy to. 
* **`pred`** The predicate to test on every value of the range <code>[first,last)</code>. 
* **`new_value`** The replacement value to assign <code>pred(&#42;i)</code> evaluates to <code>true</code>. 

**Preconditions**:
<code>first</code> may equal <code>result</code>, but the ranges <code>[first, last)</code> and <code>[result, result + (last - first))</code> shall not overlap otherwise.

**Returns**:
<code>result + (last-first)</code>

**See**:
* <a href="https://en.cppreference.com/w/cpp/algorithm/replace_copy">https://en.cppreference.com/w/cpp/algorithm/replace_copy</a>
* <code>replace</code>
* <code>replace&#95;if</code>
* <code>replace&#95;copy</code>

<h3 id="function-replace-copy-if">
Function <code>thrust::replace&#95;copy&#95;if</code>
</h3>

<code class="doxybook">
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename InputIterator1,</span>
<span>&nbsp;&nbsp;typename InputIterator2,</span>
<span>&nbsp;&nbsp;typename OutputIterator,</span>
<span>&nbsp;&nbsp;typename Predicate,</span>
<span>&nbsp;&nbsp;typename T&gt;</span>
<span>__host__ __device__ OutputIterator </span><span><b>replace_copy_if</b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;InputIterator1 first,</span>
<span>&nbsp;&nbsp;InputIterator1 last,</span>
<span>&nbsp;&nbsp;InputIterator2 stencil,</span>
<span>&nbsp;&nbsp;OutputIterator result,</span>
<span>&nbsp;&nbsp;Predicate pred,</span>
<span>&nbsp;&nbsp;const T & new_value);</span></code>
This version of <code>replace&#95;copy&#95;if</code> copies elements from the range <code>[first, last)</code> to the range <code>[result, result + (last-first))</code>, except that any element whose corresponding stencil element causes <code>pred</code> to be <code>true</code> is not copied; <code>new&#95;value</code> is copied instead.

More precisely, for every integer <code>n</code> such that <code>0 &lt;= n &lt; last-first</code>, <code>replace&#95;copy&#95;if</code> performs the assignment <code>&#42;(result+n) = new&#95;value</code> if <code>pred(&#42;(stencil+n))</code>, and <code>&#42;(result+n) = &#42;(first+n)</code> otherwise.

The algorithm's execution is parallelized as determined by <code>exec</code>.



```cpp
#include <thrust/replace.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>

struct is_less_than_zero
{
  __host__ __device__
  bool operator()(int x)
  {
    return x < 0;
  }
};

...

thrust::device_vector<int> A(4);
A[0] =  10;
A[1] =  20;
A[2] =  30;
A[3] =  40;

thrust::device_vector<int> S(4);
S[0] = -1;
S[1] =  0;
S[2] = -1;
S[3] =  0;

thrust::device_vector<int> B(4);
is_less_than_zero pred;

thrust::replace_if(thrust::device, A.begin(), A.end(), S.begin(), B.begin(), pred, 0);

// B contains [0, 20, 0, 40]
```

**Template Parameters**:
* **`DerivedPolicy`** The name of the derived execution policy. 
* **`InputIterator1`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>. 
* **`InputIterator2`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a> and <code>InputIterator2's</code><code>value&#95;type</code> is convertible to <code>Predicate's</code><code>argument&#95;type</code>. 
* **`OutputIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>. 
* **`Predicate`** is a model of <a href="https://en.cppreference.com/w/cpp/concepts/predicate">Predicate</a>. 
* **`T`** is a model of <a href="https://en.cppreference.com/w/cpp/named_req/CopyAssignable">Assignable</a>, and <code>T</code> is convertible to <code>OutputIterator's</code><code>value&#95;type</code>.

**Function Parameters**:
* **`exec`** The execution policy to use for parallelization. 
* **`first`** The beginning of the sequence to copy from. 
* **`last`** The end of the sequence to copy from. 
* **`stencil`** The beginning of the stencil sequence. 
* **`result`** The beginning of the sequence to copy to. 
* **`pred`** The predicate to test on every value of the range <code>[stencil, stencil + (last - first))</code>. 
* **`new_value`** The replacement value to assign when <code>pred(&#42;s)</code> evaluates to <code>true</code>. 

**Preconditions**:
* <code>first</code> may equal <code>result</code>, but the ranges <code>[first, last)</code> and <code>[result, result + (last - first))</code> shall not overlap otherwise. 
* <code>stencil</code> may equal <code>result</code>, but the ranges <code>[stencil, stencil + (last - first))</code> and <code>[result, result + (last - first))</code> shall not overlap otherwise.

**Returns**:
<code>result + (last-first)</code>

**See**:
* <code>replace&#95;copy</code>
* <code>replace&#95;if</code>

<h3 id="function-replace-copy-if">
Function <code>thrust::replace&#95;copy&#95;if</code>
</h3>

<code class="doxybook">
<span>template &lt;typename InputIterator1,</span>
<span>&nbsp;&nbsp;typename InputIterator2,</span>
<span>&nbsp;&nbsp;typename OutputIterator,</span>
<span>&nbsp;&nbsp;typename Predicate,</span>
<span>&nbsp;&nbsp;typename T&gt;</span>
<span>OutputIterator </span><span><b>replace_copy_if</b>(InputIterator1 first,</span>
<span>&nbsp;&nbsp;InputIterator1 last,</span>
<span>&nbsp;&nbsp;InputIterator2 stencil,</span>
<span>&nbsp;&nbsp;OutputIterator result,</span>
<span>&nbsp;&nbsp;Predicate pred,</span>
<span>&nbsp;&nbsp;const T & new_value);</span></code>
This version of <code>replace&#95;copy&#95;if</code> copies elements from the range <code>[first, last)</code> to the range <code>[result, result + (last-first))</code>, except that any element whose corresponding stencil element causes <code>pred</code> to be <code>true</code> is not copied; <code>new&#95;value</code> is copied instead.

More precisely, for every integer <code>n</code> such that <code>0 &lt;= n &lt; last-first</code>, <code>replace&#95;copy&#95;if</code> performs the assignment <code>&#42;(result+n) = new&#95;value</code> if <code>pred(&#42;(stencil+n))</code>, and <code>&#42;(result+n) = &#42;(first+n)</code> otherwise.



```cpp
#include <thrust/replace.h>
#include <thrust/device_vector.h>

struct is_less_than_zero
{
  __host__ __device__
  bool operator()(int x)
  {
    return x < 0;
  }
};

...

thrust::device_vector<int> A(4);
A[0] =  10;
A[1] =  20;
A[2] =  30;
A[3] =  40;

thrust::device_vector<int> S(4);
S[0] = -1;
S[1] =  0;
S[2] = -1;
S[3] =  0;

thrust::device_vector<int> B(4);
is_less_than_zero pred;

thrust::replace_if(A.begin(), A.end(), S.begin(), B.begin(), pred, 0);

// B contains [0, 20, 0, 40]
```

**Template Parameters**:
* **`InputIterator1`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>. 
* **`InputIterator2`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a> and <code>InputIterator2's</code><code>value&#95;type</code> is convertible to <code>Predicate's</code><code>argument&#95;type</code>. 
* **`OutputIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>. 
* **`Predicate`** is a model of <a href="https://en.cppreference.com/w/cpp/concepts/predicate">Predicate</a>. 
* **`T`** is a model of <a href="https://en.cppreference.com/w/cpp/named_req/CopyAssignable">Assignable</a>, and <code>T</code> is convertible to <code>OutputIterator's</code><code>value&#95;type</code>.

**Function Parameters**:
* **`first`** The beginning of the sequence to copy from. 
* **`last`** The end of the sequence to copy from. 
* **`stencil`** The beginning of the stencil sequence. 
* **`result`** The beginning of the sequence to copy to. 
* **`pred`** The predicate to test on every value of the range <code>[stencil, stencil + (last - first))</code>. 
* **`new_value`** The replacement value to assign when <code>pred(&#42;s)</code> evaluates to <code>true</code>. 

**Preconditions**:
* <code>first</code> may equal <code>result</code>, but the ranges <code>[first, last)</code> and <code>[result, result + (last - first))</code> shall not overlap otherwise. 
* <code>stencil</code> may equal <code>result</code>, but the ranges <code>[stencil, stencil + (last - first))</code> and <code>[result, result + (last - first))</code> shall not overlap otherwise.

**Returns**:
<code>result + (last-first)</code>

**See**:
* <code>replace&#95;copy</code>
* <code>replace&#95;if</code>


