---
title: Swap
parent: Utility
grand_parent: API
nav_exclude: false
has_children: true
has_toc: false
---

# Swap

<code class="doxybook">
<span>template &lt;typename Assignable1,</span>
<span>&nbsp;&nbsp;typename Assignable2&gt;</span>
<span>__host__ __device__ void </span><span><b><a href="{{ site.baseurl }}/api/groups/group__swap.html#function-swap">thrust::swap</a></b>(Assignable1 & a,</span>
<span>&nbsp;&nbsp;Assignable2 & b);</span>
</code>

## Functions

<h3 id="function-swap">
Function <code>thrust::swap</code>
</h3>

<code class="doxybook">
<span>template &lt;typename Assignable1,</span>
<span>&nbsp;&nbsp;typename Assignable2&gt;</span>
<span>__host__ __device__ void </span><span><b>swap</b>(Assignable1 & a,</span>
<span>&nbsp;&nbsp;Assignable2 & b);</span></code>
<code>swap</code> assigns the contents of <code>a</code> to <code>b</code> and the contents of <code>b</code> to <code>a</code>. This is used as a primitive operation by many other algorithms.


The following code snippet demonstrates how to use <code>swap</code> to swap the contents of two variables.



```cpp
#include <thrust/swap.h>
...
int x = 1;
int y = 2;
thrust::swap(x,h);

// x == 2, y == 1
```

**Template Parameters**:
**`Assignable`**: is a model of <a href="https://en.cppreference.com/w/cpp/named_req/CopyAssignable">Assignable</a>.

**Function Parameters**:
* **`a`** The first value of interest. After completion, the value of b will be returned here. 
* **`b`** The second value of interest. After completion, the value of a will be returned here.


