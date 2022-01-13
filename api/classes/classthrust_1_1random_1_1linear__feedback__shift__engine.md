---
title: thrust::random::linear_feedback_shift_engine
summary: A linear_feedback_shift_engine random number engine produces unsigned integer random values using a linear feedback shift random number generation algorithm. 
parent: Random Number Engine Class Templates
grand_parent: Random Number Generation
nav_exclude: true
has_children: true
has_toc: false
---

# Class `thrust::random::linear_feedback_shift_engine`

A <code>linear&#95;feedback&#95;shift&#95;engine</code> random number engine produces unsigned integer random values using a linear feedback shift random number generation algorithm. 

**Note**:
<a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1linear__feedback__shift__engine.html">linear_feedback_shift_engine</a> is based on the Boost Template Library's linear_feedback_shift. 

**Template Parameters**:
* **`UIntType`** The type of unsigned integer to produce. 
* **`w`** The word size of the produced values (<code>w &lt;= sizeof(UIntType)</code>). 
* **`k`** The k parameter of Tausworthe's 1965 algorithm. 
* **`q`** The q exponent of Tausworthe's 1965 algorithm. 
* **`s`** The step size of Tausworthe's 1965 algorithm.

<code class="doxybook">
<span>#include <thrust/random/linear_feedback_shift_engine.h></span><br>
<span>template &lt;typename UIntType,</span>
<span>&nbsp;&nbsp;size_t w,</span>
<span>&nbsp;&nbsp;size_t k,</span>
<span>&nbsp;&nbsp;size_t q,</span>
<span>&nbsp;&nbsp;size_t s&gt;</span>
<span>class thrust::random::linear&#95;feedback&#95;shift&#95;engine {</span>
<span>public:</span><span class="doxybook-comment"><code>&nbsp;&nbsp;</code>
/* The type of the unsigned integer produced by this <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1linear__feedback__shift__engine.html">linear&#95;feedback&#95;shift&#95;engine</a></code>.  */</span><span>&nbsp;&nbsp;typedef <i>see below</i> <b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1linear__feedback__shift__engine.html#typedef-result-type">result&#95;type</a></b>;</span>
<br>
<span>&nbsp;&nbsp;static const size_t <b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1linear__feedback__shift__engine.html#variable-word-size">word&#95;size</a></b> = <i>see below</i>;</span>
<br>
<span>&nbsp;&nbsp;static const size_t <b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1linear__feedback__shift__engine.html#variable-exponent1">exponent1</a></b> = <i>see below</i>;</span>
<br>
<span>&nbsp;&nbsp;static const size_t <b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1linear__feedback__shift__engine.html#variable-exponent2">exponent2</a></b> = <i>see below</i>;</span>
<br>
<span>&nbsp;&nbsp;static const size_t <b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1linear__feedback__shift__engine.html#variable-step-size">step&#95;size</a></b> = <i>see below</i>;</span>
<br>
<span>&nbsp;&nbsp;static const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1linear__feedback__shift__engine.html#typedef-result-type">result_type</a> <b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1linear__feedback__shift__engine.html#variable-min">min</a></b> = <i>see below</i>;</span>
<br>
<span>&nbsp;&nbsp;static const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1linear__feedback__shift__engine.html#typedef-result-type">result_type</a> <b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1linear__feedback__shift__engine.html#variable-max">max</a></b> = <i>see below</i>;</span>
<br>
<span>&nbsp;&nbsp;static const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1linear__feedback__shift__engine.html#typedef-result-type">result_type</a> <b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1linear__feedback__shift__engine.html#variable-default-seed">default&#95;seed</a></b> = <i>see below</i>;</span>
<br>
<span>&nbsp;&nbsp;explicit __host__ __device__ </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1linear__feedback__shift__engine.html#function-linear-feedback-shift-engine">linear&#95;feedback&#95;shift&#95;engine</a></b>(<a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1linear__feedback__shift__engine.html#typedef-result-type">result_type</a> value = default&#95;seed) = default;</span>
<br>
<span>&nbsp;&nbsp;__host__ __device__ void </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1linear__feedback__shift__engine.html#function-seed">seed</a></b>(<a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1linear__feedback__shift__engine.html#typedef-result-type">result_type</a> value = default&#95;seed) = default;</span>
<br>
<span>&nbsp;&nbsp;__host__ __device__ <a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1linear__feedback__shift__engine.html#typedef-result-type">result_type</a> </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1linear__feedback__shift__engine.html#function-operator()">operator()</a></b>(void);</span>
<br>
<span>&nbsp;&nbsp;__host__ __device__ void </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1linear__feedback__shift__engine.html#function-discard">discard</a></b>(unsigned long long z);</span>
<span>};</span>
</code>

## Member Types

<h3 id="typedef-result-type">
Typedef <code>thrust::random::linear&#95;feedback&#95;shift&#95;engine::result&#95;type</code>
</h3>

<code class="doxybook">
<span>typedef UIntType<b>result_type</b>;</span></code>
The type of the unsigned integer produced by this <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1linear__feedback__shift__engine.html">linear&#95;feedback&#95;shift&#95;engine</a></code>. 


## Member Variables

<h3 id="variable-word-size">
Variable <code>thrust::random::linear&#95;feedback&#95;shift&#95;engine::word&#95;size</code>
</h3>

<code class="doxybook">
<span>static const size_t <b>word_size</b> = w;</span></code>
The word size of the produced values. 

<h3 id="variable-exponent1">
Variable <code>thrust::random::linear&#95;feedback&#95;shift&#95;engine::exponent1</code>
</h3>

<code class="doxybook">
<span>static const size_t <b>exponent1</b> = k;</span></code>
A constant used in the generation algorithm. 

<h3 id="variable-exponent2">
Variable <code>thrust::random::linear&#95;feedback&#95;shift&#95;engine::exponent2</code>
</h3>

<code class="doxybook">
<span>static const size_t <b>exponent2</b> = q;</span></code>
A constant used in the generation algorithm. 

<h3 id="variable-step-size">
Variable <code>thrust::random::linear&#95;feedback&#95;shift&#95;engine::step&#95;size</code>
</h3>

<code class="doxybook">
<span>static const size_t <b>step_size</b> = s;</span></code>
The step size used in the generation algorithm. 

<h3 id="variable-min">
Variable <code>thrust::random::linear&#95;feedback&#95;shift&#95;engine::min</code>
</h3>

<code class="doxybook">
<span>static const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1linear__feedback__shift__engine.html#typedef-result-type">result_type</a> <b>min</b> = 0;</span></code>
The smallest value this <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1linear__feedback__shift__engine.html">linear&#95;feedback&#95;shift&#95;engine</a></code> may potentially produce. 

<h3 id="variable-max">
Variable <code>thrust::random::linear&#95;feedback&#95;shift&#95;engine::max</code>
</h3>

<code class="doxybook">
<span>static const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1linear__feedback__shift__engine.html#typedef-result-type">result_type</a> <b>max</b> = wordmask;</span></code>
The largest value this <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1linear__feedback__shift__engine.html">linear&#95;feedback&#95;shift&#95;engine</a></code> may potentially produce. 

<h3 id="variable-default-seed">
Variable <code>thrust::random::linear&#95;feedback&#95;shift&#95;engine::default&#95;seed</code>
</h3>

<code class="doxybook">
<span>static const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1linear__feedback__shift__engine.html#typedef-result-type">result_type</a> <b>default_seed</b> = 341u;</span></code>
The default seed of this <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1linear__feedback__shift__engine.html">linear&#95;feedback&#95;shift&#95;engine</a></code>. 


## Member Functions

<h3 id="function-linear-feedback-shift-engine">
Function <code>thrust::random::linear&#95;feedback&#95;shift&#95;engine::linear&#95;feedback&#95;shift&#95;engine</code>
</h3>

<code class="doxybook">
<span>explicit __host__ __device__ </span><span><b>linear_feedback_shift_engine</b>(<a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1linear__feedback__shift__engine.html#typedef-result-type">result_type</a> value = default&#95;seed) = default;</span></code>
This constructor, which optionally accepts a seed, initializes a new <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1linear__feedback__shift__engine.html">linear&#95;feedback&#95;shift&#95;engine</a></code>.

**Function Parameters**:
**`value`**: The seed used to intialize this <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1linear__feedback__shift__engine.html">linear&#95;feedback&#95;shift&#95;engine</a>'s</code> state. 

<h3 id="function-seed">
Function <code>thrust::random::linear&#95;feedback&#95;shift&#95;engine::seed</code>
</h3>

<code class="doxybook">
<span>__host__ __device__ void </span><span><b>seed</b>(<a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1linear__feedback__shift__engine.html#typedef-result-type">result_type</a> value = default&#95;seed) = default;</span></code>
This method initializes this <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1linear__feedback__shift__engine.html">linear&#95;feedback&#95;shift&#95;engine</a>'s</code> state, and optionally accepts a seed value.

**Function Parameters**:
**`value`**: The seed used to initializes this <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1linear__feedback__shift__engine.html">linear&#95;feedback&#95;shift&#95;engine</a>'s</code> state. 

<h3 id="function-operator()">
Function <code>thrust::random::linear&#95;feedback&#95;shift&#95;engine::operator()</code>
</h3>

<code class="doxybook">
<span>__host__ __device__ <a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1linear__feedback__shift__engine.html#typedef-result-type">result_type</a> </span><span><b>operator()</b>(void);</span></code>
This member function produces a new random value and updates this <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1linear__feedback__shift__engine.html">linear&#95;feedback&#95;shift&#95;engine</a>'s</code> state. 

**Returns**:
A new random number. 

<h3 id="function-discard">
Function <code>thrust::random::linear&#95;feedback&#95;shift&#95;engine::discard</code>
</h3>

<code class="doxybook">
<span>__host__ __device__ void </span><span><b>discard</b>(unsigned long long z);</span></code>
This member function advances this <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1linear__feedback__shift__engine.html">linear&#95;feedback&#95;shift&#95;engine</a>'s</code> state a given number of times and discards the results.

**Note**:
This function is provided because an implementation may be able to accelerate it. 

**Function Parameters**:
**`z`**: The number of random values to discard. 


