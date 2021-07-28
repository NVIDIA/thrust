---
title: identity
parent: Generalized Identity Operations
grand_parent: Predefined Function Objects
nav_exclude: true
has_children: true
has_toc: false
---

# Struct `identity`

<code>identity</code> is a Unary Function that represents the identity function: it takes a single argument <code>x</code>, and returns <code>x</code>.


The following code snippet demonstrates that <code>identity</code> returns its argument.



```cpp
#include <thrust/functional.h>
#include <assert.h>
...
int x = 137;
thrust::identity<int> id;
assert(x == id(x));
```

**Template Parameters**:
**`T`**: No requirements on <code>T</code>.

**See**:
* <a href="https://en.cppreference.com/w/cpp/utility/functional/identity">https://en.cppreference.com/w/cpp/utility/functional/identity</a>
* <a href="/api/classes/structunary__function.html">unary_function</a>

<code class="doxybook">
<span>#include <thrust/functional.h></span><br>
<span>template &lt;typename T = void&gt;</span>
<span>struct identity {</span>
<span>public:</span><span class="doxybook-comment">&nbsp;&nbsp;/* The type of the function object's first argument.  */</span><span>&nbsp;&nbsp;typedef <i>see below</i> <b><a href="/api/classes/structidentity.html#typedef-argument_type">argument&#95;type</a></b>;</span>
<br>
<span class="doxybook-comment">&nbsp;&nbsp;/* The type of the function object's result;.  */</span><span>&nbsp;&nbsp;typedef <i>see below</i> <b><a href="/api/classes/structidentity.html#typedef-result_type">result&#95;type</a></b>;</span>
<br>
<span>&nbsp;&nbsp;__thrust_exec_check_disable__ __host__ constexpr const __device__ T & </span><span>&nbsp;&nbsp;<b><a href="/api/classes/structidentity.html#function-operator()">operator()</a></b>(const T & x) const;</span>
<span>};</span>
</code>

## Member Types

<h3 id="typedef-argument_type">
Typedef <code>identity::argument&#95;type</code>
</h3>

<code class="doxybook">
<span>typedef T<b>argument_type</b>;</span></code>
The type of the function object's first argument. 

<h3 id="typedef-result_type">
Typedef <code>identity::result&#95;type</code>
</h3>

<code class="doxybook">
<span>typedef T<b>result_type</b>;</span></code>
The type of the function object's result;. 


## Member Functions

<h3 id="function-operator()">
Function <code>identity::&gt;::operator()</code>
</h3>

<code class="doxybook">
<span>__thrust_exec_check_disable__ __host__ constexpr const __device__ T & </span><span><b>operator()</b>(const T & x) const;</span></code>
Function call operator. The return value is <code>x</code>. 


