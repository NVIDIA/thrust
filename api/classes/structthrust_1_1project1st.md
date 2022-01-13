---
title: thrust::project1st
parent: Generalized Identity Operations
grand_parent: Predefined Function Objects
nav_exclude: true
has_children: true
has_toc: false
---

# Struct `thrust::project1st`

<code><a href="{{ site.baseurl }}/api/classes/structthrust_1_1project1st.html">project1st</a></code> is a function object that takes two arguments and returns its first argument; the second argument is unused. It is essentially a generalization of identity to the case of a Binary Function.



```cpp
#include <thrust/functional.h>
#include <assert.h>
...
int x =  137;
int y = -137;
thrust::project1st<int> pj1;
assert(x == pj1(x,y));
```

**See**:
* <a href="{{ site.baseurl }}/api/classes/structthrust_1_1identity.html">identity</a>
* <a href="{{ site.baseurl }}/api/classes/structthrust_1_1project2nd.html">project2nd</a>
* <a href="{{ site.baseurl }}/api/classes/structthrust_1_1binary__function.html">binary_function</a>

<code class="doxybook">
<span>#include <thrust/functional.h></span><br>
<span>template &lt;typename T1 = void,</span>
<span>&nbsp;&nbsp;typename T2 = void&gt;</span>
<span>struct thrust::project1st {</span>
<span>public:</span><span class="doxybook-comment"><code>&nbsp;&nbsp;</code>
/* The type of the function object's first argument.  */</span><span>&nbsp;&nbsp;typedef <i>see below</i> <b><a href="{{ site.baseurl }}/api/classes/structthrust_1_1project1st.html#typedef-first-argument-type">first&#95;argument&#95;type</a></b>;</span>
<br>
<span class="doxybook-comment"><code>&nbsp;&nbsp;</code>
/* The type of the function object's second argument.  */</span><span>&nbsp;&nbsp;typedef <i>see below</i> <b><a href="{{ site.baseurl }}/api/classes/structthrust_1_1project1st.html#typedef-second-argument-type">second&#95;argument&#95;type</a></b>;</span>
<br>
<span class="doxybook-comment"><code>&nbsp;&nbsp;</code>
/* The type of the function object's result;.  */</span><span>&nbsp;&nbsp;typedef <i>see below</i> <b><a href="{{ site.baseurl }}/api/classes/structthrust_1_1project1st.html#typedef-result-type">result&#95;type</a></b>;</span>
<br>
<span>&nbsp;&nbsp;__host__ __device__ constexpr const T1 & </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/structthrust_1_1project1st.html#function-operator()">operator()</a></b>(const T1 & lhs,</span>
<span>&nbsp;&nbsp;&nbsp;&nbsp;const T2 &) const;</span>
<span>};</span>
</code>

## Member Types

<h3 id="typedef-first-argument-type">
Typedef <code>thrust::project1st::first&#95;argument&#95;type</code>
</h3>

<code class="doxybook">
<span>typedef T1<b>first_argument_type</b>;</span></code>
The type of the function object's first argument. 

<h3 id="typedef-second-argument-type">
Typedef <code>thrust::project1st::second&#95;argument&#95;type</code>
</h3>

<code class="doxybook">
<span>typedef T2<b>second_argument_type</b>;</span></code>
The type of the function object's second argument. 

<h3 id="typedef-result-type">
Typedef <code>thrust::project1st::result&#95;type</code>
</h3>

<code class="doxybook">
<span>typedef T1<b>result_type</b>;</span></code>
The type of the function object's result;. 


## Member Functions

<h3 id="function-operator()">
Function <code>thrust::project1st::operator()</code>
</h3>

<code class="doxybook">
<span>__host__ __device__ constexpr const T1 & </span><span><b>operator()</b>(const T1 & lhs,</span>
<span>&nbsp;&nbsp;const T2 &) const;</span></code>
Function call operator. The return value is <code>lhs</code>. 


