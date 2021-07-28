---
title: logical_not
parent: Logical Operations
grand_parent: Predefined Function Objects
nav_exclude: true
has_children: true
has_toc: false
---

# Struct `logical_not`

<code><a href="/api/classes/structlogical__not.html">logical&#95;not</a></code> is a function object. Specifically, it is an Adaptable Predicate, which means it is a function object that tests the truth or falsehood of some condition. If <code>f</code> is an object of class <code>logical&#95;not&lt;T&gt;</code> and <code>x</code> is an object of class <code>T</code> (where <code>T</code> is convertible to <code>bool</code>) then <code>f(x)</code> returns <code>true</code> if and only if <code>x</code> is <code>false</code>.


The following code snippet demonstrates how to use <code><a href="/api/classes/structlogical__not.html">logical&#95;not</a></code> to transform a <a href="/api/classes/classdevice__vector.html">device_vector</a> of <code>bools</code> into its logical complement.



```cpp
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
...
thrust::device_vector<bool> V;
...
thrust::transform(V.begin(), V.end(), V.begin(), thrust::logical_not<bool>());
// The elements of V are now the logical complement of what they were prior
```

**Template Parameters**:
**`T`**: must be convertible to <code>bool</code>.

**See**:
* <a href="https://en.cppreference.com/w/cpp/utility/functional/logical_not">https://en.cppreference.com/w/cpp/utility/functional/logical_not</a>
* <a href="/api/classes/structunary__function.html">unary_function</a>

<code class="doxybook">
<span>#include <thrust/functional.h></span><br>
<span>template &lt;typename T = void&gt;</span>
<span>struct logical&#95;not {</span>
<span>public:</span><span class="doxybook-comment">&nbsp;&nbsp;/* The type of the function object's first argument.  */</span><span>&nbsp;&nbsp;typedef <i>see below</i> <b><a href="/api/classes/structlogical__not.html#typedef-first_argument_type">first&#95;argument&#95;type</a></b>;</span>
<br>
<span class="doxybook-comment">&nbsp;&nbsp;/* The type of the function object's second argument.  */</span><span>&nbsp;&nbsp;typedef <i>see below</i> <b><a href="/api/classes/structlogical__not.html#typedef-second_argument_type">second&#95;argument&#95;type</a></b>;</span>
<br>
<span class="doxybook-comment">&nbsp;&nbsp;/* The type of the function object's result;.  */</span><span>&nbsp;&nbsp;typedef <i>see below</i> <b><a href="/api/classes/structlogical__not.html#typedef-result_type">result&#95;type</a></b>;</span>
<br>
<span>&nbsp;&nbsp;__thrust_exec_check_disable__ __host__ constexpr __device__ bool </span><span>&nbsp;&nbsp;<b><a href="/api/classes/structlogical__not.html#function-operator()">operator()</a></b>(const T & x) const;</span>
<span>};</span>
</code>

## Member Types

<h3 id="typedef-first_argument_type">
Typedef <code>logical&#95;not::first&#95;argument&#95;type</code>
</h3>

<code class="doxybook">
<span>typedef T<b>first_argument_type</b>;</span></code>
The type of the function object's first argument. 

<h3 id="typedef-second_argument_type">
Typedef <code>logical&#95;not::second&#95;argument&#95;type</code>
</h3>

<code class="doxybook">
<span>typedef T<b>second_argument_type</b>;</span></code>
The type of the function object's second argument. 

<h3 id="typedef-result_type">
Typedef <code>logical&#95;not::result&#95;type</code>
</h3>

<code class="doxybook">
<span>typedef bool<b>result_type</b>;</span></code>
The type of the function object's result;. 


## Member Functions

<h3 id="function-operator()">
Function <code>logical&#95;not::&gt;::operator()</code>
</h3>

<code class="doxybook">
<span>__thrust_exec_check_disable__ __host__ constexpr __device__ bool </span><span><b>operator()</b>(const T & x) const;</span></code>
Function call operator. The return value is <code>!x</code>. 


