---
title: square
parent: Arithmetic Operations
grand_parent: Predefined Function Objects
nav_exclude: true
has_children: true
has_toc: false
---

# Struct `square`

<code>square</code> is a function object. Specifically, it is an Adaptable Unary Function. If <code>f</code> is an object of class <code>square&lt;T&gt;</code>, and <code>x</code> is an object of class <code>T</code>, then <code>f(x)</code> returns <code>x&#42;x</code>.


The following code snippet demonstrates how to use <code>square</code> to square the elements of a <a href="/thrust/api/classes/classdevice__vector.html">device_vector</a> of <code>floats</code>.



```cpp
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>
...
const int N = 1000;
thrust::device_vector<float> V1(N);
thrust::device_vector<float> V2(N);

thrust::sequence(V1.begin(), V1.end(), 1);

thrust::transform(V1.begin(), V1.end(), V2.begin(),
                  thrust::square<float>());
// V2 is now {1, 4, 9, ..., 1000000}
```

**Template Parameters**:
**`T`**: is a model of <a href="https://en.cppreference.com/w/cpp/named_req/CopyAssignable">Assignable</a>, and if <code>x</code> is an object of type <code>T</code>, then <code>x&#42;x</code> must be defined and must have a return type that is convertible to <code>T</code>.

**See**:
<a href="/thrust/api/classes/structunary__function.html">unary_function</a>

<code class="doxybook">
<span>#include <thrust/functional.h></span><br>
<span>template &lt;typename T = void&gt;</span>
<span>struct square {</span>
<span>public:</span><span class="doxybook-comment">&nbsp;&nbsp;/* The type of the function object's argument.  */</span><span>&nbsp;&nbsp;typedef <i>see below</i> <b><a href="/thrust/api/classes/structsquare.html#typedef-argument_type">argument&#95;type</a></b>;</span>
<br>
<span class="doxybook-comment">&nbsp;&nbsp;/* The type of the function object's result;.  */</span><span>&nbsp;&nbsp;typedef <i>see below</i> <b><a href="/thrust/api/classes/structsquare.html#typedef-result_type">result&#95;type</a></b>;</span>
<br>
<span>&nbsp;&nbsp;__thrust_exec_check_disable__ __host__ constexpr __device__ T </span><span>&nbsp;&nbsp;<b><a href="/thrust/api/classes/structsquare.html#function-operator()">operator()</a></b>(const T & x) const;</span>
<span>};</span>
</code>

## Member Types

<h3 id="typedef-argument_type">
Typedef <code>square::argument&#95;type</code>
</h3>

<code class="doxybook">
<span>typedef T<b>argument_type</b>;</span></code>
The type of the function object's argument. 

<h3 id="typedef-result_type">
Typedef <code>square::result&#95;type</code>
</h3>

<code class="doxybook">
<span>typedef T<b>result_type</b>;</span></code>
The type of the function object's result;. 


## Member Functions

<h3 id="function-operator()">
Function <code>square::&gt;::operator()</code>
</h3>

<code class="doxybook">
<span>__thrust_exec_check_disable__ __host__ constexpr __device__ T </span><span><b>operator()</b>(const T & x) const;</span></code>
Function call operator. The return value is <code>x&#42;x</code>. 


