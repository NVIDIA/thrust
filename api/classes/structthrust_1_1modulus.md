---
title: thrust::modulus
parent: Arithmetic Operations
grand_parent: Predefined Function Objects
nav_exclude: true
has_children: true
has_toc: false
---

# Struct `thrust::modulus`

<code>modulus</code> is a function object. Specifically, it is an Adaptable Binary Function. If <code>f</code> is an object of class <code>modulus&lt;T&gt;</code>, and <code>x</code> and <code>y</code> are objects of class <code>T</code>, then <code>f(x,y)</code> returns <code>x % y</code>.


The following code snippet demonstrates how to use <code>modulus</code> to take the modulus of one device_vectors of <code>floats</code> by another.



```cpp
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/sequence.h>
#include <thrust/fill.h>
#include <thrust/transform.h>
...
const int N = 1000;
thrust::device_vector<float> V1(N);
thrust::device_vector<float> V2(N);
thrust::device_vector<float> V3(N);

thrust::sequence(V1.begin(), V1.end(), 1);
thrust::fill(V2.begin(), V2.end(), 75);

thrust::transform(V1.begin(), V1.end(), V2.begin(), V3.begin(),
                  thrust::modulus<int>());
// V3 is now {1%75, 2%75, 3%75, ..., 1000%75}
```

**Template Parameters**:
**`T`**: is a model of <a href="https://en.cppreference.com/w/cpp/named_req/CopyAssignable">Assignable</a>, and if <code>x</code> and <code>y</code> are objects of type <code>T</code>, then <code>x % y</code> must be defined and must have a return type that is convertible to <code>T</code>.

**See**:
* <a href="https://en.cppreference.com/w/cpp/utility/functional/modulus">https://en.cppreference.com/w/cpp/utility/functional/modulus</a>
* <a href="{{ site.baseurl }}/api/classes/structthrust_1_1binary__function.html">binary_function</a>

<code class="doxybook">
<span>#include <thrust/functional.h></span><br>
<span>template &lt;typename T = void&gt;</span>
<span>struct thrust::modulus {</span>
<span>public:</span><span class="doxybook-comment"><code>&nbsp;&nbsp;</code>
/* The type of the function object's first argument.  */</span><span>&nbsp;&nbsp;typedef <i>see below</i> <b><a href="{{ site.baseurl }}/api/classes/structthrust_1_1modulus.html#typedef-first-argument-type">first&#95;argument&#95;type</a></b>;</span>
<br>
<span class="doxybook-comment"><code>&nbsp;&nbsp;</code>
/* The type of the function object's second argument.  */</span><span>&nbsp;&nbsp;typedef <i>see below</i> <b><a href="{{ site.baseurl }}/api/classes/structthrust_1_1modulus.html#typedef-second-argument-type">second&#95;argument&#95;type</a></b>;</span>
<br>
<span class="doxybook-comment"><code>&nbsp;&nbsp;</code>
/* The type of the function object's result;.  */</span><span>&nbsp;&nbsp;typedef <i>see below</i> <b><a href="{{ site.baseurl }}/api/classes/structthrust_1_1modulus.html#typedef-result-type">result&#95;type</a></b>;</span>
<br>
<span>&nbsp;&nbsp;__thrust_exec_check_disable__ __host__ __device__ constexpr T </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/structthrust_1_1modulus.html#function-operator()">operator()</a></b>(const T & lhs,</span>
<span>&nbsp;&nbsp;&nbsp;&nbsp;const T & rhs) const;</span>
<span>};</span>
</code>

## Member Types

<h3 id="typedef-first-argument-type">
Typedef <code>thrust::modulus::first&#95;argument&#95;type</code>
</h3>

<code class="doxybook">
<span>typedef T<b>first_argument_type</b>;</span></code>
The type of the function object's first argument. 

<h3 id="typedef-second-argument-type">
Typedef <code>thrust::modulus::second&#95;argument&#95;type</code>
</h3>

<code class="doxybook">
<span>typedef T<b>second_argument_type</b>;</span></code>
The type of the function object's second argument. 

<h3 id="typedef-result-type">
Typedef <code>thrust::modulus::result&#95;type</code>
</h3>

<code class="doxybook">
<span>typedef T<b>result_type</b>;</span></code>
The type of the function object's result;. 


## Member Functions

<h3 id="function-operator()">
Function <code>thrust::modulus::operator()</code>
</h3>

<code class="doxybook">
<span>__thrust_exec_check_disable__ __host__ __device__ constexpr T </span><span><b>operator()</b>(const T & lhs,</span>
<span>&nbsp;&nbsp;const T & rhs) const;</span></code>
Function call operator. The return value is <code>lhs % rhs</code>. 


