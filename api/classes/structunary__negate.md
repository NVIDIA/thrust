---
title: unary_negate
parent: Function Object Adaptors
grand_parent: Function Objects
nav_exclude: true
has_children: true
has_toc: false
---

# Struct `unary_negate`

<code><a href="/api/classes/structunary__negate.html">unary&#95;negate</a></code> is a function object adaptor: it is an Adaptable Predicate that represents the logical negation of some other Adaptable Predicate. That is: if <code>f</code> is an object of class <code>unary&#95;negate&lt;AdaptablePredicate&gt;</code>, then there exists an object <code>pred</code> of class <code>AdaptablePredicate</code> such that <code>f(x)</code> always returns the same value as <code>!pred(x)</code>. There is rarely any reason to construct a <code><a href="/api/classes/structunary__negate.html">unary&#95;negate</a></code> directly; it is almost always easier to use the helper function not1.

**Inherits From**:
`thrust::unary_function< Predicate::argument_type, bool >`

**See**:
* <a href="https://en.cppreference.com/w/cpp/utility/functional/unary_negate">https://en.cppreference.com/w/cpp/utility/functional/unary_negate</a>
* <a href="/api/groups/group__function__object__adaptors.html#function-not1">not1</a>

<code class="doxybook">
<span>#include <thrust/functional.h></span><br>
<span>template &lt;typename Predicate&gt;</span>
<span>struct unary&#95;negate {</span>
<span>public:</span><span>&nbsp;&nbsp;explicit __host__ __device__ </span><span>&nbsp;&nbsp;<b><a href="/api/classes/structunary__negate.html#function-unary_negate">unary&#95;negate</a></b>(Predicate p);</span>
<br>
<span>&nbsp;&nbsp;__thrust_exec_check_disable__ __host__ __device__ bool </span><span>&nbsp;&nbsp;<b><a href="/api/classes/structunary__negate.html#function-operator()">operator()</a></b>(const typename Predicate::argument_type & x);</span>
<span>};</span>
</code>

## Member Functions

<h3 id="function-unary_negate">
Function <code>unary&#95;negate::&gt;::unary&#95;negate</code>
</h3>

<code class="doxybook">
<span>explicit __host__ __device__ </span><span><b>unary_negate</b>(Predicate p);</span></code>
Constructor takes a <code>Predicate</code> object to negate. 

**Function Parameters**:
**`p`**: The <code>Predicate</code> object to negate. 

<h3 id="function-operator()">
Function <code>unary&#95;negate::&gt;::operator()</code>
</h3>

<code class="doxybook">
<span>__thrust_exec_check_disable__ __host__ __device__ bool </span><span><b>operator()</b>(const typename Predicate::argument_type & x);</span></code>
Function call operator. The return value is <code>!pred(x)</code>. 


