---
title: Logical
parent: Reductions
grand_parent: Algorithms
nav_exclude: false
has_children: true
has_toc: false
---

# Logical

<code class="doxybook">
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename InputIterator,</span>
<span>&nbsp;&nbsp;typename Predicate&gt;</span>
<span>__host__ __device__ bool </span><span><b><a href="{{ site.baseurl }}/api/groups/group__logical.html#function-all-of">thrust::all&#95;of</a></b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;InputIterator first,</span>
<span>&nbsp;&nbsp;InputIterator last,</span>
<span>&nbsp;&nbsp;Predicate pred);</span>
<br>
<span>template &lt;typename InputIterator,</span>
<span>&nbsp;&nbsp;typename Predicate&gt;</span>
<span>bool </span><span><b><a href="{{ site.baseurl }}/api/groups/group__logical.html#function-all-of">thrust::all&#95;of</a></b>(InputIterator first,</span>
<span>&nbsp;&nbsp;InputIterator last,</span>
<span>&nbsp;&nbsp;Predicate pred);</span>
<br>
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename InputIterator,</span>
<span>&nbsp;&nbsp;typename Predicate&gt;</span>
<span>__host__ __device__ bool </span><span><b><a href="{{ site.baseurl }}/api/groups/group__logical.html#function-any-of">thrust::any&#95;of</a></b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;InputIterator first,</span>
<span>&nbsp;&nbsp;InputIterator last,</span>
<span>&nbsp;&nbsp;Predicate pred);</span>
<br>
<span>template &lt;typename InputIterator,</span>
<span>&nbsp;&nbsp;typename Predicate&gt;</span>
<span>bool </span><span><b><a href="{{ site.baseurl }}/api/groups/group__logical.html#function-any-of">thrust::any&#95;of</a></b>(InputIterator first,</span>
<span>&nbsp;&nbsp;InputIterator last,</span>
<span>&nbsp;&nbsp;Predicate pred);</span>
<br>
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename InputIterator,</span>
<span>&nbsp;&nbsp;typename Predicate&gt;</span>
<span>__host__ __device__ bool </span><span><b><a href="{{ site.baseurl }}/api/groups/group__logical.html#function-none-of">thrust::none&#95;of</a></b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;InputIterator first,</span>
<span>&nbsp;&nbsp;InputIterator last,</span>
<span>&nbsp;&nbsp;Predicate pred);</span>
<br>
<span>template &lt;typename InputIterator,</span>
<span>&nbsp;&nbsp;typename Predicate&gt;</span>
<span>bool </span><span><b><a href="{{ site.baseurl }}/api/groups/group__logical.html#function-none-of">thrust::none&#95;of</a></b>(InputIterator first,</span>
<span>&nbsp;&nbsp;InputIterator last,</span>
<span>&nbsp;&nbsp;Predicate pred);</span>
</code>

## Functions

<h3 id="function-all-of">
Function <code>thrust::all&#95;of</code>
</h3>

<code class="doxybook">
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename InputIterator,</span>
<span>&nbsp;&nbsp;typename Predicate&gt;</span>
<span>__host__ __device__ bool </span><span><b>all_of</b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;InputIterator first,</span>
<span>&nbsp;&nbsp;InputIterator last,</span>
<span>&nbsp;&nbsp;Predicate pred);</span></code>
<code>all&#95;of</code> determines whether all elements in a range satify a predicate. Specifically, <code>all&#95;of</code> returns <code>true</code> if <code>pred(&#42;i)</code> is <code>true</code> for every iterator <code>i</code> in the range <code>[first, last)</code> and <code>false</code> otherwise.

The algorithm's execution is parallelized as determined by <code>exec</code>.



```cpp
#include <thrust/logical.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>
...
bool A[3] = {true, true, false};

thrust::all_of(thrust::host, A, A + 2, thrust::identity<bool>()); // returns true
thrust::all_of(thrust::host, A, A + 3, thrust::identity<bool>()); // returns false

// empty range
thrust::all_of(thrust::host, A, A, thrust::identity<bool>()); // returns false
```

**Template Parameters**:
* **`DerivedPolicy`** The name of the derived execution policy. 
* **`InputIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>, 
* **`Predicate`** must be a model of <a href="https://en.cppreference.com/w/cpp/concepts/predicate">Predicate</a>.

**Function Parameters**:
* **`exec`** The execution policy to use for parallelization. 
* **`first`** The beginning of the sequence. 
* **`last`** The end of the sequence. 
* **`pred`** A predicate used to test range elements. 

**Returns**:
<code>true</code>, if all elements satisfy the predicate; <code>false</code>, otherwise.

**See**:
* any_of 
* none_of 
* transform_reduce 

<h3 id="function-all-of">
Function <code>thrust::all&#95;of</code>
</h3>

<code class="doxybook">
<span>template &lt;typename InputIterator,</span>
<span>&nbsp;&nbsp;typename Predicate&gt;</span>
<span>bool </span><span><b>all_of</b>(InputIterator first,</span>
<span>&nbsp;&nbsp;InputIterator last,</span>
<span>&nbsp;&nbsp;Predicate pred);</span></code>
<code>all&#95;of</code> determines whether all elements in a range satify a predicate. Specifically, <code>all&#95;of</code> returns <code>true</code> if <code>pred(&#42;i)</code> is <code>true</code> for every iterator <code>i</code> in the range <code>[first, last)</code> and <code>false</code> otherwise.



```cpp
#include <thrust/logical.h>
#include <thrust/functional.h>
...
bool A[3] = {true, true, false};

thrust::all_of(A, A + 2, thrust::identity<bool>()); // returns true
thrust::all_of(A, A + 3, thrust::identity<bool>()); // returns false

// empty range
thrust::all_of(A, A, thrust::identity<bool>()); // returns false
```

**Template Parameters**:
* **`InputIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>, 
* **`Predicate`** must be a model of <a href="https://en.cppreference.com/w/cpp/concepts/predicate">Predicate</a>.

**Function Parameters**:
* **`first`** The beginning of the sequence. 
* **`last`** The end of the sequence. 
* **`pred`** A predicate used to test range elements. 

**Returns**:
<code>true</code>, if all elements satisfy the predicate; <code>false</code>, otherwise.

**See**:
* any_of 
* none_of 
* transform_reduce 

<h3 id="function-any-of">
Function <code>thrust::any&#95;of</code>
</h3>

<code class="doxybook">
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename InputIterator,</span>
<span>&nbsp;&nbsp;typename Predicate&gt;</span>
<span>__host__ __device__ bool </span><span><b>any_of</b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;InputIterator first,</span>
<span>&nbsp;&nbsp;InputIterator last,</span>
<span>&nbsp;&nbsp;Predicate pred);</span></code>
<code>any&#95;of</code> determines whether any element in a range satifies a predicate. Specifically, <code>any&#95;of</code> returns <code>true</code> if <code>pred(&#42;i)</code> is <code>true</code> for any iterator <code>i</code> in the range <code>[first, last)</code> and <code>false</code> otherwise.

The algorithm's execution is parallelized as determined by <code>exec</code>.



```cpp
#include <thrust/logical.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>
...
bool A[3] = {true, true, false};

thrust::any_of(thrust::host, A, A + 2, thrust::identity<bool>()); // returns true
thrust::any_of(thrust::host, A, A + 3, thrust::identity<bool>()); // returns true

thrust::any_of(thrust::host, A + 2, A + 3, thrust::identity<bool>()); // returns false

// empty range
thrust::any_of(thrust::host, A, A, thrust::identity<bool>()); // returns false
```

**Template Parameters**:
* **`DerivedPolicy`** The name of the derived execution policy. 
* **`InputIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>, 
* **`Predicate`** must be a model of <a href="https://en.cppreference.com/w/cpp/concepts/predicate">Predicate</a>.

**Function Parameters**:
* **`exec`** The execution policy to use for parallelization. 
* **`first`** The beginning of the sequence. 
* **`last`** The end of the sequence. 
* **`pred`** A predicate used to test range elements. 

**Returns**:
<code>true</code>, if any element satisfies the predicate; <code>false</code>, otherwise.

**See**:
* all_of 
* none_of 
* transform_reduce 

<h3 id="function-any-of">
Function <code>thrust::any&#95;of</code>
</h3>

<code class="doxybook">
<span>template &lt;typename InputIterator,</span>
<span>&nbsp;&nbsp;typename Predicate&gt;</span>
<span>bool </span><span><b>any_of</b>(InputIterator first,</span>
<span>&nbsp;&nbsp;InputIterator last,</span>
<span>&nbsp;&nbsp;Predicate pred);</span></code>
<code>any&#95;of</code> determines whether any element in a range satifies a predicate. Specifically, <code>any&#95;of</code> returns <code>true</code> if <code>pred(&#42;i)</code> is <code>true</code> for any iterator <code>i</code> in the range <code>[first, last)</code> and <code>false</code> otherwise.



```cpp
#include <thrust/logical.h>
#include <thrust/functional.h>
...
bool A[3] = {true, true, false};

thrust::any_of(A, A + 2, thrust::identity<bool>()); // returns true
thrust::any_of(A, A + 3, thrust::identity<bool>()); // returns true

thrust::any_of(A + 2, A + 3, thrust::identity<bool>()); // returns false

// empty range
thrust::any_of(A, A, thrust::identity<bool>()); // returns false
```

**Template Parameters**:
* **`InputIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>, 
* **`Predicate`** must be a model of <a href="https://en.cppreference.com/w/cpp/concepts/predicate">Predicate</a>.

**Function Parameters**:
* **`first`** The beginning of the sequence. 
* **`last`** The end of the sequence. 
* **`pred`** A predicate used to test range elements. 

**Returns**:
<code>true</code>, if any element satisfies the predicate; <code>false</code>, otherwise.

**See**:
* all_of 
* none_of 
* transform_reduce 

<h3 id="function-none-of">
Function <code>thrust::none&#95;of</code>
</h3>

<code class="doxybook">
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename InputIterator,</span>
<span>&nbsp;&nbsp;typename Predicate&gt;</span>
<span>__host__ __device__ bool </span><span><b>none_of</b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;InputIterator first,</span>
<span>&nbsp;&nbsp;InputIterator last,</span>
<span>&nbsp;&nbsp;Predicate pred);</span></code>
<code>none&#95;of</code> determines whether no element in a range satifies a predicate. Specifically, <code>none&#95;of</code> returns <code>true</code> if there is no iterator <code>i</code> in the range <code>[first, last)</code> such that <code>pred(&#42;i)</code> is <code>true</code>, and <code>false</code> otherwise.

The algorithm's execution is parallelized as determined by <code>exec</code>.



```cpp
#include <thrust/logical.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>
...
bool A[3] = {true, true, false};

thrust::none_of(thrust::host, A, A + 2, thrust::identity<bool>()); // returns false
thrust::none_of(thrust::host, A, A + 3, thrust::identity<bool>()); // returns false

thrust::none_of(thrust::host, A + 2, A + 3, thrust::identity<bool>()); // returns true

// empty range
thrust::none_of(thrust::host, A, A, thrust::identity<bool>()); // returns true
```

**Template Parameters**:
* **`DerivedPolicy`** The name of the derived execution policy. 
* **`InputIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>, 
* **`Predicate`** must be a model of <a href="https://en.cppreference.com/w/cpp/concepts/predicate">Predicate</a>.

**Function Parameters**:
* **`exec`** The execution policy to use for parallelization. 
* **`first`** The beginning of the sequence. 
* **`last`** The end of the sequence. 
* **`pred`** A predicate used to test range elements. 

**Returns**:
<code>true</code>, if no element satisfies the predicate; <code>false</code>, otherwise.

**See**:
* all_of 
* any_of 
* transform_reduce 

<h3 id="function-none-of">
Function <code>thrust::none&#95;of</code>
</h3>

<code class="doxybook">
<span>template &lt;typename InputIterator,</span>
<span>&nbsp;&nbsp;typename Predicate&gt;</span>
<span>bool </span><span><b>none_of</b>(InputIterator first,</span>
<span>&nbsp;&nbsp;InputIterator last,</span>
<span>&nbsp;&nbsp;Predicate pred);</span></code>
<code>none&#95;of</code> determines whether no element in a range satifies a predicate. Specifically, <code>none&#95;of</code> returns <code>true</code> if there is no iterator <code>i</code> in the range <code>[first, last)</code> such that <code>pred(&#42;i)</code> is <code>true</code>, and <code>false</code> otherwise.



```cpp
#include <thrust/logical.h>
#include <thrust/functional.h>
...
bool A[3] = {true, true, false};

thrust::none_of(A, A + 2, thrust::identity<bool>()); // returns false
thrust::none_of(A, A + 3, thrust::identity<bool>()); // returns false

thrust::none_of(A + 2, A + 3, thrust::identity<bool>()); // returns true

// empty range
thrust::none_of(A, A, thrust::identity<bool>()); // returns true
```

**Template Parameters**:
* **`InputIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>, 
* **`Predicate`** must be a model of <a href="https://en.cppreference.com/w/cpp/concepts/predicate">Predicate</a>.

**Function Parameters**:
* **`first`** The beginning of the sequence. 
* **`last`** The end of the sequence. 
* **`pred`** A predicate used to test range elements. 

**Returns**:
<code>true</code>, if no element satisfies the predicate; <code>false</code>, otherwise.

**See**:
* all_of 
* any_of 
* transform_reduce 


