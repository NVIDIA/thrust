---
title: thrust::random::xor_combine_engine
summary: An xor_combine_engine adapts two existing base random number engines and produces random values by combining the values produced by each. 
parent: Random Number Engine Adaptor Class Templates
grand_parent: Random Number Generation
nav_exclude: true
has_children: true
has_toc: false
---

# Class `thrust::random::xor_combine_engine`

An <code>xor&#95;combine&#95;engine</code> adapts two existing base random number engines and produces random values by combining the values produced by each. 


The following code snippet shows an example of using an <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1xor__combine__engine.html">xor&#95;combine&#95;engine</a></code> instance:



```cpp
#include <thrust/random/linear_congruential_engine.h>
#include <thrust/random/xor_combine_engine.h>
#include <iostream>

int main(void)
{
  // create an xor_combine_engine from minstd_rand and minstd_rand0
  // use a shift of 0 for each
  thrust::xor_combine_engine<thrust::minstd_rand,0,thrust::minstd_rand0,0> rng;

  // print a random number to standard output
  std::cout << rng() << std::endl;

  return 0;
}
```

**Template Parameters**:
* **`Engine1`** The type of the first base random number engine to adapt. 
* **`s1`** The size of the first shift to use in the generation algorithm. 
* **`Engine2`** The type of the second base random number engine to adapt. 
* **`s2`** The second of the second shift to use in the generation algorithm. Defaults to <code>0</code>.

<code class="doxybook">
<span>#include <thrust/random/xor_combine_engine.h></span><br>
<span>template &lt;typename Engine1,</span>
<span>&nbsp;&nbsp;size_t s1,</span>
<span>&nbsp;&nbsp;typename Engine2,</span>
<span>&nbsp;&nbsp;size_t s2 = 0u&gt;</span>
<span>class thrust::random::xor&#95;combine&#95;engine {</span>
<span>public:</span><span class="doxybook-comment"><code>&nbsp;&nbsp;</code>
/* The type of the first adapted base random number engine.  */</span><span>&nbsp;&nbsp;typedef <i>see below</i> <b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1xor__combine__engine.html#typedef-base1-type">base1&#95;type</a></b>;</span>
<br>
<span class="doxybook-comment"><code>&nbsp;&nbsp;</code>
/* The type of the second adapted base random number engine.  */</span><span>&nbsp;&nbsp;typedef <i>see below</i> <b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1xor__combine__engine.html#typedef-base2-type">base2&#95;type</a></b>;</span>
<br>
<span class="doxybook-comment"><code>&nbsp;&nbsp;</code>
/* The type of the unsigned integer produced by this <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1xor__combine__engine.html">xor&#95;combine&#95;engine</a></code>.  */</span><span>&nbsp;&nbsp;typedef <i>see below</i> <b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1xor__combine__engine.html#typedef-result-type">result&#95;type</a></b>;</span>
<br>
<span>&nbsp;&nbsp;static const size_t <b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1xor__combine__engine.html#variable-shift1">shift1</a></b> = <i>see below</i>;</span>
<br>
<span>&nbsp;&nbsp;static const size_t <b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1xor__combine__engine.html#variable-shift2">shift2</a></b> = <i>see below</i>;</span>
<br>
<span>&nbsp;&nbsp;static const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1xor__combine__engine.html#typedef-result-type">result_type</a> <b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1xor__combine__engine.html#variable-min">min</a></b> = <i>see below</i>;</span>
<br>
<span>&nbsp;&nbsp;static const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1xor__combine__engine.html#typedef-result-type">result_type</a> <b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1xor__combine__engine.html#variable-max">max</a></b> = <i>see below</i>;</span>
<br>
<span>&nbsp;&nbsp;__host__ __device__ </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1xor__combine__engine.html#function-xor-combine-engine">xor&#95;combine&#95;engine</a></b>(void);</span>
<br>
<span>&nbsp;&nbsp;__host__ __device__ </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1xor__combine__engine.html#function-xor-combine-engine">xor&#95;combine&#95;engine</a></b>(const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1xor__combine__engine.html#typedef-base1-type">base1_type</a> & urng1,</span>
<span>&nbsp;&nbsp;&nbsp;&nbsp;const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1xor__combine__engine.html#typedef-base2-type">base2_type</a> & urng2);</span>
<br>
<span>&nbsp;&nbsp;__host__ __device__ </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1xor__combine__engine.html#function-xor-combine-engine">xor&#95;combine&#95;engine</a></b>(<a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1xor__combine__engine.html#typedef-result-type">result_type</a> s);</span>
<br>
<span>&nbsp;&nbsp;__host__ __device__ void </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1xor__combine__engine.html#function-seed">seed</a></b>(void);</span>
<br>
<span>&nbsp;&nbsp;__host__ __device__ void </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1xor__combine__engine.html#function-seed">seed</a></b>(<a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1xor__combine__engine.html#typedef-result-type">result_type</a> s);</span>
<br>
<span>&nbsp;&nbsp;__host__ __device__ <a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1xor__combine__engine.html#typedef-result-type">result_type</a> </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1xor__combine__engine.html#function-operator()">operator()</a></b>(void);</span>
<br>
<span>&nbsp;&nbsp;__host__ __device__ void </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1xor__combine__engine.html#function-discard">discard</a></b>(unsigned long long z);</span>
<br>
<span>&nbsp;&nbsp;__host__ __device__ const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1xor__combine__engine.html#typedef-base1-type">base1_type</a> & </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1xor__combine__engine.html#function-base1">base1</a></b>(void) const;</span>
<br>
<span>&nbsp;&nbsp;__host__ __device__ const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1xor__combine__engine.html#typedef-base2-type">base2_type</a> & </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1xor__combine__engine.html#function-base2">base2</a></b>(void) const;</span>
<span>};</span>
</code>

## Member Types

<h3 id="typedef-base1-type">
Typedef <code>thrust::random::xor&#95;combine&#95;engine::base1&#95;type</code>
</h3>

<code class="doxybook">
<span>typedef Engine1<b>base1_type</b>;</span></code>
The type of the first adapted base random number engine. 

<h3 id="typedef-base2-type">
Typedef <code>thrust::random::xor&#95;combine&#95;engine::base2&#95;type</code>
</h3>

<code class="doxybook">
<span>typedef Engine2<b>base2_type</b>;</span></code>
The type of the second adapted base random number engine. 

<h3 id="typedef-result-type">
Typedef <code>thrust::random::xor&#95;combine&#95;engine::result&#95;type</code>
</h3>

<code class="doxybook">
<span>typedef thrust::detail::eval_if<(sizeof(typenamebase2_type::result_type)>sizeof(typenamebase1_type::result_type)), thrust::detail::identity_< typenamebase2_type::result_type >, thrust::detail::identity_< typenamebase1_type::result_type > >::type<b>result_type</b>;</span></code>
The type of the unsigned integer produced by this <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1xor__combine__engine.html">xor&#95;combine&#95;engine</a></code>. 


## Member Variables

<h3 id="variable-shift1">
Variable <code>thrust::random::xor&#95;combine&#95;engine::shift1</code>
</h3>

<code class="doxybook">
<span>static const size_t <b>shift1</b> = s1;</span></code>
The size of the first shift used in the generation algorithm. 

<h3 id="variable-shift2">
Variable <code>thrust::random::xor&#95;combine&#95;engine::shift2</code>
</h3>

<code class="doxybook">
<span>static const size_t <b>shift2</b> = s2;</span></code>
The size of the second shift used in the generation algorithm. 

<h3 id="variable-min">
Variable <code>thrust::random::xor&#95;combine&#95;engine::min</code>
</h3>

<code class="doxybook">
<span>static const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1xor__combine__engine.html#typedef-result-type">result_type</a> <b>min</b> = 0;</span></code>
The smallest value this <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1xor__combine__engine.html">xor&#95;combine&#95;engine</a></code> may potentially produce. 

<h3 id="variable-max">
Variable <code>thrust::random::xor&#95;combine&#95;engine::max</code>
</h3>

<code class="doxybook">
<span>static const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1xor__combine__engine.html#typedef-result-type">result_type</a> <b>max</b> =
      detail::xor&#95;combine&#95;engine&#95;max&lt;
        Engine1, s1, Engine2, s2, result&#95;type
      &gt;::value;</span></code>
The largest value this <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1xor__combine__engine.html">xor&#95;combine&#95;engine</a></code> may potentially produce. 


## Member Functions

<h3 id="function-xor-combine-engine">
Function <code>thrust::random::xor&#95;combine&#95;engine::xor&#95;combine&#95;engine</code>
</h3>

<code class="doxybook">
<span>__host__ __device__ </span><span><b>xor_combine_engine</b>(void);</span></code>
This constructor constructs a new <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1xor__combine__engine.html">xor&#95;combine&#95;engine</a></code> and constructs its adapted engines using their null constructors. 

<h3 id="function-xor-combine-engine">
Function <code>thrust::random::xor&#95;combine&#95;engine::xor&#95;combine&#95;engine</code>
</h3>

<code class="doxybook">
<span>__host__ __device__ </span><span><b>xor_combine_engine</b>(const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1xor__combine__engine.html#typedef-base1-type">base1_type</a> & urng1,</span>
<span>&nbsp;&nbsp;const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1xor__combine__engine.html#typedef-base2-type">base2_type</a> & urng2);</span></code>
This constructor constructs a new <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1xor__combine__engine.html">xor&#95;combine&#95;engine</a></code> using given <code>base1&#95;type</code> and <code>base2&#95;type</code> engines to initialize its adapted base engines.

**Function Parameters**:
* **`urng1`** A <code>base1&#95;type</code> to use to initialize this <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1xor__combine__engine.html">xor&#95;combine&#95;engine</a>'s</code> first adapted base engine. 
* **`urng2`** A <code>base2&#95;type</code> to use to initialize this <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1xor__combine__engine.html">xor&#95;combine&#95;engine</a>'s</code> first adapted base engine. 

<h3 id="function-xor-combine-engine">
Function <code>thrust::random::xor&#95;combine&#95;engine::xor&#95;combine&#95;engine</code>
</h3>

<code class="doxybook">
<span>__host__ __device__ </span><span><b>xor_combine_engine</b>(<a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1xor__combine__engine.html#typedef-result-type">result_type</a> s);</span></code>
This constructor initializes a new <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1xor__combine__engine.html">xor&#95;combine&#95;engine</a></code> with a given seed.

**Function Parameters**:
**`s`**: The seed used to intialize this <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1xor__combine__engine.html">xor&#95;combine&#95;engine</a>'s</code> adapted base engines. 

<h3 id="function-seed">
Function <code>thrust::random::xor&#95;combine&#95;engine::seed</code>
</h3>

<code class="doxybook">
<span>__host__ __device__ void </span><span><b>seed</b>(void);</span></code>
This method initializes the state of this <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1xor__combine__engine.html">xor&#95;combine&#95;engine</a>'s</code> adapted base engines by using their <code>default&#95;seed</code> values. 

<h3 id="function-seed">
Function <code>thrust::random::xor&#95;combine&#95;engine::seed</code>
</h3>

<code class="doxybook">
<span>__host__ __device__ void </span><span><b>seed</b>(<a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1xor__combine__engine.html#typedef-result-type">result_type</a> s);</span></code>
This method initializes the state of this <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1xor__combine__engine.html">xor&#95;combine&#95;engine</a>'s</code> adapted base engines by using the given seed.

**Function Parameters**:
**`s`**: The seed with which to intialize this <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1xor__combine__engine.html">xor&#95;combine&#95;engine</a>'s</code> adapted base engines. 

<h3 id="function-operator()">
Function <code>thrust::random::xor&#95;combine&#95;engine::operator()</code>
</h3>

<code class="doxybook">
<span>__host__ __device__ <a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1xor__combine__engine.html#typedef-result-type">result_type</a> </span><span><b>operator()</b>(void);</span></code>
This member function produces a new random value and updates this <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1xor__combine__engine.html">xor&#95;combine&#95;engine</a>'s</code> state. 

**Returns**:
A new random number. 

<h3 id="function-discard">
Function <code>thrust::random::xor&#95;combine&#95;engine::discard</code>
</h3>

<code class="doxybook">
<span>__host__ __device__ void </span><span><b>discard</b>(unsigned long long z);</span></code>
This member function advances this <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1xor__combine__engine.html">xor&#95;combine&#95;engine</a>'s</code> state a given number of times and discards the results.

**Note**:
This function is provided because an implementation may be able to accelerate it. 

**Function Parameters**:
**`z`**: The number of random values to discard. 

<h3 id="function-base1">
Function <code>thrust::random::xor&#95;combine&#95;engine::base1</code>
</h3>

<code class="doxybook">
<span>__host__ __device__ const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1xor__combine__engine.html#typedef-base1-type">base1_type</a> & </span><span><b>base1</b>(void) const;</span></code>
This member function returns a const reference to this <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1xor__combine__engine.html">xor&#95;combine&#95;engine</a>'s</code> first adapted base engine.

**Returns**:
A const reference to the first base engine this <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1xor__combine__engine.html">xor&#95;combine&#95;engine</a></code> adapts. 

<h3 id="function-base2">
Function <code>thrust::random::xor&#95;combine&#95;engine::base2</code>
</h3>

<code class="doxybook">
<span>__host__ __device__ const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1xor__combine__engine.html#typedef-base2-type">base2_type</a> & </span><span><b>base2</b>(void) const;</span></code>
This member function returns a const reference to this <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1xor__combine__engine.html">xor&#95;combine&#95;engine</a>'s</code> second adapted base engine.

**Returns**:
A const reference to the second base engine this <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1xor__combine__engine.html">xor&#95;combine&#95;engine</a></code> adapts. 


