---
title: thrust::random::subtract_with_carry_engine
summary: A subtract_with_carry_engine random number engine produces unsigned integer random numbers using the subtract with carry algorithm of Marsaglia & Zaman. 
parent: Random Number Engine Class Templates
grand_parent: Random Number Generation
nav_exclude: true
has_children: true
has_toc: false
---

# Class `thrust::random::subtract_with_carry_engine`

A <code>subtract&#95;with&#95;carry&#95;engine</code> random number engine produces unsigned integer random numbers using the subtract with carry algorithm of Marsaglia & Zaman. 

The generation algorithm is performed as follows:

1. Let <code>Y = X&#95;{i-s}- X&#95;{i-r} - c</code>.
2. Set <code>X&#95;i</code> to <code>y = T mod m</code>. Set <code>c</code> to <code>1</code> if <code>Y &lt; 0</code>, otherwise set <code>c</code> to <code>0</code>.
This algorithm corresponds to a modular linear function of the form

<code>TA(x&#95;i) = (a &#42; x&#95;i) mod b</code>, where <code>b</code> is of the form <code>m^r - m^s + 1</code> and <code>a = b - (b-1)/m</code>.

**Note**:
Inexperienced users should not use this class template directly. Instead, use <code>ranlux24&#95;base</code> or <code>ranlux48&#95;base</code>, which are instances of <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1subtract__with__carry__engine.html">subtract&#95;with&#95;carry&#95;engine</a></code>.

**Template Parameters**:
* **`UIntType`** The type of unsigned integer to produce. 
* **`w`** The word size of the produced values (<code> w &lt;= sizeof(UIntType)</code>). 
* **`s`** The short lag of the generation algorithm. 
* **`r`** The long lag of the generation algorithm.

**See**:
* <a href="{{ site.baseurl }}/api/groups/group__predefined__random.html#typedef-ranlux24-base">thrust::random::ranlux24_base</a>
* <a href="{{ site.baseurl }}/api/groups/group__predefined__random.html#typedef-ranlux48-base">thrust::random::ranlux48_base</a>

<code class="doxybook">
<span>#include <thrust/random/subtract_with_carry_engine.h></span><br>
<span>template &lt;typename UIntType,</span>
<span>&nbsp;&nbsp;size_t w,</span>
<span>&nbsp;&nbsp;size_t s,</span>
<span>&nbsp;&nbsp;size_t r&gt;</span>
<span>class thrust::random::subtract&#95;with&#95;carry&#95;engine {</span>
<span>public:</span><span class="doxybook-comment"><code>&nbsp;&nbsp;</code>
/* The type of the unsigned integer produced by this <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1subtract__with__carry__engine.html">subtract&#95;with&#95;carry&#95;engine</a></code>.  */</span><span>&nbsp;&nbsp;typedef <i>see below</i> <b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1subtract__with__carry__engine.html#typedef-result-type">result&#95;type</a></b>;</span>
<br>
<span>&nbsp;&nbsp;static const size_t <b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1subtract__with__carry__engine.html#variable-word-size">word&#95;size</a></b> = <i>see below</i>;</span>
<br>
<span>&nbsp;&nbsp;static const size_t <b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1subtract__with__carry__engine.html#variable-short-lag">short&#95;lag</a></b> = <i>see below</i>;</span>
<br>
<span>&nbsp;&nbsp;static const size_t <b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1subtract__with__carry__engine.html#variable-long-lag">long&#95;lag</a></b> = <i>see below</i>;</span>
<br>
<span>&nbsp;&nbsp;static const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1subtract__with__carry__engine.html#typedef-result-type">result_type</a> <b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1subtract__with__carry__engine.html#variable-min">min</a></b> = <i>see below</i>;</span>
<br>
<span>&nbsp;&nbsp;static const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1subtract__with__carry__engine.html#typedef-result-type">result_type</a> <b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1subtract__with__carry__engine.html#variable-max">max</a></b> = <i>see below</i>;</span>
<br>
<span>&nbsp;&nbsp;static const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1subtract__with__carry__engine.html#typedef-result-type">result_type</a> <b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1subtract__with__carry__engine.html#variable-default-seed">default&#95;seed</a></b> = <i>see below</i>;</span>
<br>
<span>&nbsp;&nbsp;explicit __host__ __device__ </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1subtract__with__carry__engine.html#function-subtract-with-carry-engine">subtract&#95;with&#95;carry&#95;engine</a></b>(<a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1subtract__with__carry__engine.html#typedef-result-type">result_type</a> value = default&#95;seed) = default;</span>
<br>
<span>&nbsp;&nbsp;__host__ __device__ void </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1subtract__with__carry__engine.html#function-seed">seed</a></b>(<a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1subtract__with__carry__engine.html#typedef-result-type">result_type</a> value = default&#95;seed) = default;</span>
<br>
<span>&nbsp;&nbsp;__host__ __device__ <a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1subtract__with__carry__engine.html#typedef-result-type">result_type</a> </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1subtract__with__carry__engine.html#function-operator()">operator()</a></b>(void);</span>
<br>
<span>&nbsp;&nbsp;__host__ __device__ void </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1subtract__with__carry__engine.html#function-discard">discard</a></b>(unsigned long long z);</span>
<span>};</span>
</code>

## Member Types

<h3 id="typedef-result-type">
Typedef <code>thrust::random::subtract&#95;with&#95;carry&#95;engine::result&#95;type</code>
</h3>

<code class="doxybook">
<span>typedef UIntType<b>result_type</b>;</span></code>
The type of the unsigned integer produced by this <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1subtract__with__carry__engine.html">subtract&#95;with&#95;carry&#95;engine</a></code>. 


## Member Variables

<h3 id="variable-word-size">
Variable <code>thrust::random::subtract&#95;with&#95;carry&#95;engine::word&#95;size</code>
</h3>

<code class="doxybook">
<span>static const size_t <b>word_size</b> = w;</span></code>
The word size of the produced values. 

<h3 id="variable-short-lag">
Variable <code>thrust::random::subtract&#95;with&#95;carry&#95;engine::short&#95;lag</code>
</h3>

<code class="doxybook">
<span>static const size_t <b>short_lag</b> = s;</span></code>
The size of the short lag used in the generation algorithm. 

<h3 id="variable-long-lag">
Variable <code>thrust::random::subtract&#95;with&#95;carry&#95;engine::long&#95;lag</code>
</h3>

<code class="doxybook">
<span>static const size_t <b>long_lag</b> = r;</span></code>
The size of the long lag used in the generation algorithm. 

<h3 id="variable-min">
Variable <code>thrust::random::subtract&#95;with&#95;carry&#95;engine::min</code>
</h3>

<code class="doxybook">
<span>static const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1subtract__with__carry__engine.html#typedef-result-type">result_type</a> <b>min</b> = 0;</span></code>
The smallest value this <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1subtract__with__carry__engine.html">subtract&#95;with&#95;carry&#95;engine</a></code> may potentially produce. 

<h3 id="variable-max">
Variable <code>thrust::random::subtract&#95;with&#95;carry&#95;engine::max</code>
</h3>

<code class="doxybook">
<span>static const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1subtract__with__carry__engine.html#typedef-result-type">result_type</a> <b>max</b> = modulus - 1;</span></code>
The largest value this <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1subtract__with__carry__engine.html">subtract&#95;with&#95;carry&#95;engine</a></code> may potentially produce. 

<h3 id="variable-default-seed">
Variable <code>thrust::random::subtract&#95;with&#95;carry&#95;engine::default&#95;seed</code>
</h3>

<code class="doxybook">
<span>static const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1subtract__with__carry__engine.html#typedef-result-type">result_type</a> <b>default_seed</b> = 19780503u;</span></code>
The default seed of this <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1subtract__with__carry__engine.html">subtract&#95;with&#95;carry&#95;engine</a></code>. 


## Member Functions

<h3 id="function-subtract-with-carry-engine">
Function <code>thrust::random::subtract&#95;with&#95;carry&#95;engine::subtract&#95;with&#95;carry&#95;engine</code>
</h3>

<code class="doxybook">
<span>explicit __host__ __device__ </span><span><b>subtract_with_carry_engine</b>(<a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1subtract__with__carry__engine.html#typedef-result-type">result_type</a> value = default&#95;seed) = default;</span></code>
This constructor, which optionally accepts a seed, initializes a new <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1subtract__with__carry__engine.html">subtract&#95;with&#95;carry&#95;engine</a></code>.

**Function Parameters**:
**`value`**: The seed used to intialize this <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1subtract__with__carry__engine.html">subtract&#95;with&#95;carry&#95;engine</a>'s</code> state. 

<h3 id="function-seed">
Function <code>thrust::random::subtract&#95;with&#95;carry&#95;engine::seed</code>
</h3>

<code class="doxybook">
<span>__host__ __device__ void </span><span><b>seed</b>(<a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1subtract__with__carry__engine.html#typedef-result-type">result_type</a> value = default&#95;seed) = default;</span></code>
This method initializes this <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1subtract__with__carry__engine.html">subtract&#95;with&#95;carry&#95;engine</a>'s</code> state, and optionally accepts a seed value.

**Function Parameters**:
**`value`**: The seed used to initializes this <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1subtract__with__carry__engine.html">subtract&#95;with&#95;carry&#95;engine</a>'s</code> state. 

<h3 id="function-operator()">
Function <code>thrust::random::subtract&#95;with&#95;carry&#95;engine::operator()</code>
</h3>

<code class="doxybook">
<span>__host__ __device__ <a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1subtract__with__carry__engine.html#typedef-result-type">result_type</a> </span><span><b>operator()</b>(void);</span></code>
This member function produces a new random value and updates this <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1subtract__with__carry__engine.html">subtract&#95;with&#95;carry&#95;engine</a>'s</code> state. 

**Returns**:
A new random number. 

<h3 id="function-discard">
Function <code>thrust::random::subtract&#95;with&#95;carry&#95;engine::discard</code>
</h3>

<code class="doxybook">
<span>__host__ __device__ void </span><span><b>discard</b>(unsigned long long z);</span></code>
This member function advances this <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1subtract__with__carry__engine.html">subtract&#95;with&#95;carry&#95;engine</a>'s</code> state a given number of times and discards the results.

**Note**:
This function is provided because an implementation may be able to accelerate it. 

**Function Parameters**:
**`z`**: The number of random values to discard. 


