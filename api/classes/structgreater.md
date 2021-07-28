---
title: greater
parent: Comparison Operations
grand_parent: Predefined Function Objects
nav_exclude: true
has_children: true
has_toc: false
---

# Struct `greater`

<code>greater</code> is a function object. Specifically, it is an Adaptable Binary Predicate, which means it is a function object that tests the truth or falsehood of some condition. If <code>f</code> is an object of class <code>greater&lt;T&gt;</code> and <code>x</code> and <code>y</code> are objects of class <code>T</code>, then <code>f(x,y)</code> returns <code>true</code> if <code>x &gt; y</code> and <code>false</code> otherwise.

**Template Parameters**:
**`T`**: is a model of <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a>.

**See**:
* <a href="https://en.cppreference.com/w/cpp/utility/functional/greater">https://en.cppreference.com/w/cpp/utility/functional/greater</a>
* <a href="/api/classes/structbinary__function.html">binary_function</a>

<code class="doxybook">
<span>#include <thrust/functional.h></span><br>
<span>template &lt;typename T = void&gt;</span>
<span>struct greater {</span>
<span>public:</span><span class="doxybook-comment">&nbsp;&nbsp;/* The type of the function object's first argument.  */</span><span>&nbsp;&nbsp;typedef <i>see below</i> <b><a href="/api/classes/structgreater.html#typedef-first_argument_type">first&#95;argument&#95;type</a></b>;</span>
<br>
<span class="doxybook-comment">&nbsp;&nbsp;/* The type of the function object's second argument.  */</span><span>&nbsp;&nbsp;typedef <i>see below</i> <b><a href="/api/classes/structgreater.html#typedef-second_argument_type">second&#95;argument&#95;type</a></b>;</span>
<br>
<span class="doxybook-comment">&nbsp;&nbsp;/* The type of the function object's result;.  */</span><span>&nbsp;&nbsp;typedef <i>see below</i> <b><a href="/api/classes/structgreater.html#typedef-result_type">result&#95;type</a></b>;</span>
<br>
<span>&nbsp;&nbsp;__thrust_exec_check_disable__ __host__ constexpr __device__ bool </span><span>&nbsp;&nbsp;<b><a href="/api/classes/structgreater.html#function-operator()">operator()</a></b>(const T & lhs,</span>
<span>&nbsp;&nbsp;&nbsp;&nbsp;const T & rhs) const;</span>
<span>};</span>
</code>

## Member Types

<h3 id="typedef-first_argument_type">
Typedef <code>greater::first&#95;argument&#95;type</code>
</h3>

<code class="doxybook">
<span>typedef T<b>first_argument_type</b>;</span></code>
The type of the function object's first argument. 

<h3 id="typedef-second_argument_type">
Typedef <code>greater::second&#95;argument&#95;type</code>
</h3>

<code class="doxybook">
<span>typedef T<b>second_argument_type</b>;</span></code>
The type of the function object's second argument. 

<h3 id="typedef-result_type">
Typedef <code>greater::result&#95;type</code>
</h3>

<code class="doxybook">
<span>typedef bool<b>result_type</b>;</span></code>
The type of the function object's result;. 


## Member Functions

<h3 id="function-operator()">
Function <code>greater::&gt;::operator()</code>
</h3>

<code class="doxybook">
<span>__thrust_exec_check_disable__ __host__ constexpr __device__ bool </span><span><b>operator()</b>(const T & lhs,</span>
<span>&nbsp;&nbsp;const T & rhs) const;</span></code>
Function call operator. The return value is <code>lhs &gt; rhs</code>. 


