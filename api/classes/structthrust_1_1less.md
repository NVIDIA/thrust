---
title: thrust::less
parent: Comparison Operations
grand_parent: Predefined Function Objects
nav_exclude: true
has_children: true
has_toc: false
---

# Struct `thrust::less`

<code>less</code> is a function object. Specifically, it is an Adaptable Binary Predicate, which means it is a function object that tests the truth or falsehood of some condition. If <code>f</code> is an object of class <code>less&lt;T&gt;</code> and <code>x</code> and <code>y</code> are objects of class <code>T</code>, then <code>f(x,y)</code> returns <code>true</code> if <code>x &lt; y</code> and <code>false</code> otherwise.

**Template Parameters**:
**`T`**: is a model of <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a>.

**See**:
* <a href="https://en.cppreference.com/w/cpp/utility/functional/less">https://en.cppreference.com/w/cpp/utility/functional/less</a>
* <a href="{{ site.baseurl }}/api/classes/structthrust_1_1binary__function.html">binary_function</a>

<code class="doxybook">
<span>#include <thrust/functional.h></span><br>
<span>template &lt;typename T = void&gt;</span>
<span>struct thrust::less {</span>
<span>public:</span><span class="doxybook-comment"><code>&nbsp;&nbsp;</code>
/* The type of the function object's first argument.  */</span><span>&nbsp;&nbsp;typedef <i>see below</i> <b><a href="{{ site.baseurl }}/api/classes/structthrust_1_1less.html#typedef-first-argument-type">first&#95;argument&#95;type</a></b>;</span>
<br>
<span class="doxybook-comment"><code>&nbsp;&nbsp;</code>
/* The type of the function object's second argument.  */</span><span>&nbsp;&nbsp;typedef <i>see below</i> <b><a href="{{ site.baseurl }}/api/classes/structthrust_1_1less.html#typedef-second-argument-type">second&#95;argument&#95;type</a></b>;</span>
<br>
<span class="doxybook-comment"><code>&nbsp;&nbsp;</code>
/* The type of the function object's result;.  */</span><span>&nbsp;&nbsp;typedef <i>see below</i> <b><a href="{{ site.baseurl }}/api/classes/structthrust_1_1less.html#typedef-result-type">result&#95;type</a></b>;</span>
<br>
<span>&nbsp;&nbsp;__thrust_exec_check_disable__ __host__ __device__ constexpr bool </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/structthrust_1_1less.html#function-operator()">operator()</a></b>(const T & lhs,</span>
<span>&nbsp;&nbsp;&nbsp;&nbsp;const T & rhs) const;</span>
<span>};</span>
</code>

## Member Types

<h3 id="typedef-first-argument-type">
Typedef <code>thrust::less::first&#95;argument&#95;type</code>
</h3>

<code class="doxybook">
<span>typedef T<b>first_argument_type</b>;</span></code>
The type of the function object's first argument. 

<h3 id="typedef-second-argument-type">
Typedef <code>thrust::less::second&#95;argument&#95;type</code>
</h3>

<code class="doxybook">
<span>typedef T<b>second_argument_type</b>;</span></code>
The type of the function object's second argument. 

<h3 id="typedef-result-type">
Typedef <code>thrust::less::result&#95;type</code>
</h3>

<code class="doxybook">
<span>typedef bool<b>result_type</b>;</span></code>
The type of the function object's result;. 


## Member Functions

<h3 id="function-operator()">
Function <code>thrust::less::operator()</code>
</h3>

<code class="doxybook">
<span>__thrust_exec_check_disable__ __host__ __device__ constexpr bool </span><span><b>operator()</b>(const T & lhs,</span>
<span>&nbsp;&nbsp;const T & rhs) const;</span></code>
Function call operator. The return value is <code>lhs &lt; rhs</code>. 


