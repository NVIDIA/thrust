---
title: thrust::unary_negate
parent: Function Object Adaptors
grand_parent: Function Objects
nav_exclude: true
has_children: true
has_toc: false
---

# Struct `thrust::unary_negate`

<code><a href="{{ site.baseurl }}/api/classes/structthrust_1_1unary__negate.html">unary&#95;negate</a></code> is a function object adaptor: it is an Adaptable Predicate that represents the logical negation of some other Adaptable Predicate. That is: if <code>f</code> is an object of class <code>unary&#95;negate&lt;AdaptablePredicate&gt;</code>, then there exists an object <code>pred</code> of class <code>AdaptablePredicate</code> such that <code>f(x)</code> always returns the same value as <code>!pred(x)</code>. There is rarely any reason to construct a <code><a href="{{ site.baseurl }}/api/classes/structthrust_1_1unary__negate.html">unary&#95;negate</a></code> directly; it is almost always easier to use the helper function not1.

**Inherits From**:
[`thrust::unary_function< Predicate::argument_type, bool >`]({{ site.baseurl }}/api/classes/structthrust_1_1unary__function.html)

**See**:
* <a href="https://en.cppreference.com/w/cpp/utility/functional/unary_negate">https://en.cppreference.com/w/cpp/utility/functional/unary_negate</a>
* not1 

<code class="doxybook">
<span>#include <thrust/functional.h></span><br>
<span>template &lt;typename Predicate&gt;</span>
<span>struct thrust::unary&#95;negate {</span>
<span>public:</span><span class="doxybook-comment"><code>&nbsp;&nbsp;</code>
/* Inherited from <code><b><a href="{{ site.baseurl }}/api/classes/structthrust_1_1unary__function.html">thrust::unary&#95;function&lt; Predicate::argument&#95;type, bool &gt;</a></b></code> */</span><span class="doxybook-comment"><code>&nbsp;&nbsp;</code>
/* The type of the function object's argument.  */</span><span>&nbsp;&nbsp;typedef <i>see below</i> <b><a href="{{ site.baseurl }}/api/classes/structthrust_1_1unary__function.html#typedef-argument-type">argument&#95;type</a></b>;</span>
<br>
<span class="doxybook-comment"><code>&nbsp;&nbsp;</code>
/* Inherited from <code><b><a href="{{ site.baseurl }}/api/classes/structthrust_1_1unary__function.html">thrust::unary&#95;function&lt; Predicate::argument&#95;type, bool &gt;</a></b></code> */</span><span class="doxybook-comment"><code>&nbsp;&nbsp;</code>
/* The type of the function object's result.  */</span><span>&nbsp;&nbsp;typedef <i>see below</i> <b><a href="{{ site.baseurl }}/api/classes/structthrust_1_1unary__function.html#typedef-result-type">result&#95;type</a></b>;</span>
<br>
<span>&nbsp;&nbsp;explicit __host__ __device__ </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/structthrust_1_1unary__negate.html#function-unary-negate">unary&#95;negate</a></b>(Predicate p);</span>
<br>
<span>&nbsp;&nbsp;__thrust_exec_check_disable__ __host__ __device__ bool </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/structthrust_1_1unary__negate.html#function-operator()">operator()</a></b>(const typename Predicate::argument_type & x);</span>
<span>};</span>
</code>

## Member Functions

<h3 id="function-unary-negate">
Function <code>thrust::unary&#95;negate::unary&#95;negate</code>
</h3>

<code class="doxybook">
<span>explicit __host__ __device__ </span><span><b>unary_negate</b>(Predicate p);</span></code>
Constructor takes a <code>Predicate</code> object to negate. 

**Function Parameters**:
**`p`**: The <code>Predicate</code> object to negate. 

<h3 id="function-operator()">
Function <code>thrust::unary&#95;negate::operator()</code>
</h3>

<code class="doxybook">
<span>__thrust_exec_check_disable__ __host__ __device__ bool </span><span><b>operator()</b>(const typename Predicate::argument_type & x);</span></code>
Function call operator. The return value is <code>!pred(x)</code>. 


