---
title: Function Object Adaptors
parent: Function Objects
grand_parent: API
nav_exclude: false
has_children: true
has_toc: false
---

# Function Object Adaptors

<code class="doxybook">
<span class="doxybook-comment">/* \exclude  */</span><span>namespace <b><a href="{{ site.baseurl }}/api/namespaces/namespacethrust_1_1detail.html">thrust::detail</a></b> { <i>…</i> }</span>
<br>
<span>template &lt;typename Argument,</span>
<span>&nbsp;&nbsp;typename Result&gt;</span>
<span>struct <b><a href="{{ site.baseurl }}/api/classes/structthrust_1_1unary__function.html">thrust::unary&#95;function</a></b>;</span>
<br>
<span>template &lt;typename Argument1,</span>
<span>&nbsp;&nbsp;typename Argument2,</span>
<span>&nbsp;&nbsp;typename Result&gt;</span>
<span>struct <b><a href="{{ site.baseurl }}/api/classes/structthrust_1_1binary__function.html">thrust::binary&#95;function</a></b>;</span>
<br>
<span>template &lt;typename Predicate&gt;</span>
<span>struct <b><a href="{{ site.baseurl }}/api/classes/structthrust_1_1unary__negate.html">thrust::unary&#95;negate</a></b>;</span>
<br>
<span>template &lt;typename Predicate&gt;</span>
<span>struct <b><a href="{{ site.baseurl }}/api/classes/structthrust_1_1binary__negate.html">thrust::binary&#95;negate</a></b>;</span>
<br>
<span>template &lt;typename Function&gt;</span>
<span>class <b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1zip__function.html">thrust::zip&#95;function</a></b>;</span>
<br>
<span>template &lt;typename Predicate&gt;</span>
<span>__host__ __device__ unary_negate< Predicate > </span><span><b><a href="{{ site.baseurl }}/api/groups/group__function__object__adaptors.html#function-not1">thrust::not1</a></b>(const Predicate & pred);</span>
<br>
<span>template &lt;typename BinaryPredicate&gt;</span>
<span>__host__ __device__ binary_negate< BinaryPredicate > </span><span><b><a href="{{ site.baseurl }}/api/groups/group__function__object__adaptors.html#function-not2">thrust::not2</a></b>(const BinaryPredicate & pred);</span>
<br>
<span>template &lt;typename Function&gt;</span>
<span>__host__ __device__ zip_function< typename std::decay< Function >::type > </span><span><b><a href="{{ site.baseurl }}/api/groups/group__function__object__adaptors.html#function-make-zip-function">thrust::make&#95;zip&#95;function</a></b>(Function && fun);</span>
</code>

## Member Classes

<h3 id="struct-thrustunary-function">
<a href="{{ site.baseurl }}/api/classes/structthrust_1_1unary__function.html">Struct <code>thrust::unary&#95;function</code>
</a>
</h3>

<h3 id="struct-thrustbinary-function">
<a href="{{ site.baseurl }}/api/classes/structthrust_1_1binary__function.html">Struct <code>thrust::binary&#95;function</code>
</a>
</h3>

<h3 id="struct-thrustunary-negate">
<a href="{{ site.baseurl }}/api/classes/structthrust_1_1unary__negate.html">Struct <code>thrust::unary&#95;negate</code>
</a>
</h3>

**Inherits From**:
[`thrust::unary_function< Predicate::argument_type, bool >`]({{ site.baseurl }}/api/classes/structthrust_1_1unary__function.html)

<h3 id="struct-thrustbinary-negate">
<a href="{{ site.baseurl }}/api/classes/structthrust_1_1binary__negate.html">Struct <code>thrust::binary&#95;negate</code>
</a>
</h3>

**Inherits From**:
[`thrust::binary_function< Predicate::first_argument_type, Predicate::second_argument_type, bool >`]({{ site.baseurl }}/api/classes/structthrust_1_1binary__function.html)

<h3 id="class-thrustzip-function">
<a href="{{ site.baseurl }}/api/classes/classthrust_1_1zip__function.html">Class <code>thrust::zip&#95;function</code>
</a>
</h3>


## Functions

<h3 id="function-not1">
Function <code>thrust::not1</code>
</h3>

<code class="doxybook">
<span>template &lt;typename Predicate&gt;</span>
<span>__host__ __device__ unary_negate< Predicate > </span><span><b>not1</b>(const Predicate & pred);</span></code>
<code>not1</code> is a helper function to simplify the creation of Adaptable Predicates: it takes an Adaptable Predicate <code>pred</code> as an argument and returns a new Adaptable Predicate that represents the negation of <code>pred</code>. That is: if <code>pred</code> is an object of a type which models Adaptable Predicate, then the the type of the result <code>npred</code> of <code>not1(pred)</code> is also a model of Adaptable Predicate and <code>npred(x)</code> always returns the same value as <code>!pred(x)</code>.

**Template Parameters**:
**`Predicate`**: is a model of <a href="https://en.cppreference.com/w/cpp/utility/functional/unary_negate">Adaptable Predicate</a>.

**Function Parameters**:
**`pred`**: The Adaptable Predicate to negate. 

**Returns**:
A new object, <code>npred</code> such that <code>npred(x)</code> always returns the same value as <code>!pred(x)</code>.

**See**:
* <a href="{{ site.baseurl }}/api/classes/structthrust_1_1unary__negate.html">unary_negate</a>
* not2 

<h3 id="function-not2">
Function <code>thrust::not2</code>
</h3>

<code class="doxybook">
<span>template &lt;typename BinaryPredicate&gt;</span>
<span>__host__ __device__ binary_negate< BinaryPredicate > </span><span><b>not2</b>(const BinaryPredicate & pred);</span></code>
<code>not2</code> is a helper function to simplify the creation of Adaptable Binary Predicates: it takes an Adaptable Binary Predicate <code>pred</code> as an argument and returns a new Adaptable Binary Predicate that represents the negation of <code>pred</code>. That is: if <code>pred</code> is an object of a type which models Adaptable Binary Predicate, then the the type of the result <code>npred</code> of <code>not2(pred)</code> is also a model of Adaptable Binary Predicate and <code>npred(x,y)</code> always returns the same value as <code>!pred(x,y)</code>.

**Template Parameters**:
**`Binary`**: Predicate is a model of <a href="https://en.cppreference.com/w/cpp/utility/functional/AdaptableBinaryPredicate">Adaptable Binary Predicate</a>.

**Function Parameters**:
**`pred`**: The Adaptable Binary Predicate to negate. 

**Returns**:
A new object, <code>npred</code> such that <code>npred(x,y)</code> always returns the same value as <code>!pred(x,y)</code>.

**See**:
* <a href="{{ site.baseurl }}/api/classes/structthrust_1_1binary__negate.html">binary_negate</a>
* not1 

<h3 id="function-make-zip-function">
Function <code>thrust::make&#95;zip&#95;function</code>
</h3>

<code class="doxybook">
<span>template &lt;typename Function&gt;</span>
<span>__host__ __device__ zip_function< typename std::decay< Function >::type > </span><span><b>make_zip_function</b>(Function && fun);</span></code>
<code>make&#95;zip&#95;function</code> creates a <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1zip__function.html">zip&#95;function</a></code> from a function object.

**Function Parameters**:
**`fun`**: The N-ary function object. 

**Returns**:
A <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1zip__function.html">zip&#95;function</a></code> that takes a N-tuple.

**See**:
<a href="{{ site.baseurl }}/api/classes/classthrust_1_1zip__function.html">zip_function</a>


