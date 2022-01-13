---
title: thrust::binary_negate
parent: Function Object Adaptors
grand_parent: Function Objects
nav_exclude: true
has_children: true
has_toc: false
---

# Struct `thrust::binary_negate`

<code><a href="{{ site.baseurl }}/api/classes/structthrust_1_1binary__negate.html">binary&#95;negate</a></code> is a function object adaptor: it is an Adaptable Binary Predicate that represents the logical negation of some other Adaptable Binary Predicate. That is: if <code>f</code> is an object of class <code>binary&#95;negate&lt;AdaptablePredicate&gt;</code>, then there exists an object <code>pred</code> of class <code>AdaptableBinaryPredicate</code> such that <code>f(x,y)</code> always returns the same value as <code>!pred(x,y)</code>. There is rarely any reason to construct a <code><a href="{{ site.baseurl }}/api/classes/structthrust_1_1binary__negate.html">binary&#95;negate</a></code> directly; it is almost always easier to use the helper function not2.

**Inherits From**:
[`thrust::binary_function< Predicate::first_argument_type, Predicate::second_argument_type, bool >`]({{ site.baseurl }}/api/classes/structthrust_1_1binary__function.html)

**See**:
<a href="https://en.cppreference.com/w/cpp/utility/functional/binary_negate">https://en.cppreference.com/w/cpp/utility/functional/binary_negate</a>

<code class="doxybook">
<span>#include <thrust/functional.h></span><br>
<span>template &lt;typename Predicate&gt;</span>
<span>struct thrust::binary&#95;negate {</span>
<span>public:</span><span class="doxybook-comment"><code>&nbsp;&nbsp;</code>
/* Inherited from <code><b><a href="{{ site.baseurl }}/api/classes/structthrust_1_1binary__function.html">thrust::binary&#95;function&lt; Predicate::first&#95;argument&#95;type, Predicate::second&#95;argument&#95;type, bool &gt;</a></b></code> */</span><span class="doxybook-comment"><code>&nbsp;&nbsp;</code>
/* The type of the function object's first argument.  */</span><span>&nbsp;&nbsp;typedef <i>see below</i> <b><a href="{{ site.baseurl }}/api/classes/structthrust_1_1binary__function.html#typedef-first-argument-type">first&#95;argument&#95;type</a></b>;</span>
<br>
<span class="doxybook-comment"><code>&nbsp;&nbsp;</code>
/* Inherited from <code><b><a href="{{ site.baseurl }}/api/classes/structthrust_1_1binary__function.html">thrust::binary&#95;function&lt; Predicate::first&#95;argument&#95;type, Predicate::second&#95;argument&#95;type, bool &gt;</a></b></code> */</span><span class="doxybook-comment"><code>&nbsp;&nbsp;</code>
/* The type of the function object's second argument.  */</span><span>&nbsp;&nbsp;typedef <i>see below</i> <b><a href="{{ site.baseurl }}/api/classes/structthrust_1_1binary__function.html#typedef-second-argument-type">second&#95;argument&#95;type</a></b>;</span>
<br>
<span class="doxybook-comment"><code>&nbsp;&nbsp;</code>
/* Inherited from <code><b><a href="{{ site.baseurl }}/api/classes/structthrust_1_1binary__function.html">thrust::binary&#95;function&lt; Predicate::first&#95;argument&#95;type, Predicate::second&#95;argument&#95;type, bool &gt;</a></b></code> */</span><span class="doxybook-comment"><code>&nbsp;&nbsp;</code>
/* The type of the function object's result;.  */</span><span>&nbsp;&nbsp;typedef <i>see below</i> <b><a href="{{ site.baseurl }}/api/classes/structthrust_1_1binary__function.html#typedef-result-type">result&#95;type</a></b>;</span>
<br>
<span>&nbsp;&nbsp;explicit __host__ __device__ </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/structthrust_1_1binary__negate.html#function-binary-negate">binary&#95;negate</a></b>(Predicate p);</span>
<br>
<span>&nbsp;&nbsp;__thrust_exec_check_disable__ __host__ __device__ bool </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/structthrust_1_1binary__negate.html#function-operator()">operator()</a></b>(const typename Predicate::first_argument_type & x,</span>
<span>&nbsp;&nbsp;&nbsp;&nbsp;const typename Predicate::second_argument_type & y);</span>
<span>};</span>
</code>

## Member Functions

<h3 id="function-binary-negate">
Function <code>thrust::binary&#95;negate::binary&#95;negate</code>
</h3>

<code class="doxybook">
<span>explicit __host__ __device__ </span><span><b>binary_negate</b>(Predicate p);</span></code>
Constructor takes a <code>Predicate</code> object to negate. 

**Function Parameters**:
**`p`**: The <code>Predicate</code> object to negate. 

<h3 id="function-operator()">
Function <code>thrust::binary&#95;negate::operator()</code>
</h3>

<code class="doxybook">
<span>__thrust_exec_check_disable__ __host__ __device__ bool </span><span><b>operator()</b>(const typename Predicate::first_argument_type & x,</span>
<span>&nbsp;&nbsp;const typename Predicate::second_argument_type & y);</span></code>
Function call operator. The return value is <code>!pred(x,y)</code>. 


