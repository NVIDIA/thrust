---
title: thrust::random::linear_congruential_engine
summary: A linear_congruential_engine random number engine produces unsigned integer random numbers using a linear congruential random number generation algorithm. 
parent: Random Number Engine Class Templates
grand_parent: Random Number Generation
nav_exclude: true
has_children: true
has_toc: false
---

# Class `thrust::random::linear_congruential_engine`

A <code>linear&#95;congruential&#95;engine</code> random number engine produces unsigned integer random numbers using a linear congruential random number generation algorithm. 

The generation algorithm has the form <code>x&#95;i = (a &#42; x&#95;{i-1} + c) mod m</code>.


The following code snippet shows examples of use of a <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1linear__congruential__engine.html">linear&#95;congruential&#95;engine</a></code> instance:



```cpp
#include <thrust/random/linear_congruential_engine.h>
#include <iostream>

int main(void)
{
  // create a minstd_rand object, which is an instance of linear_congruential_engine
  thrust::minstd_rand rng1;

  // output some random values to cout
  std::cout << rng1() << std::endl;

  // a random value is printed

  // create a new minstd_rand from a seed
  thrust::minstd_rand rng2(13);

  // discard some random values
  rng2.discard(13);

  // stream the object to an iostream
  std::cout << rng2 << std::endl;

  // rng2's current state is printed

  // print the minimum and maximum values that minstd_rand can produce
  std::cout << thrust::minstd_rand::min << std::endl;
  std::cout << thrust::minstd_rand::max << std::endl;

  // the range of minstd_rand is printed

  // save the state of rng2 to a different object
  thrust::minstd_rand rng3 = rng2;

  // compare rng2 and rng3
  std::cout << (rng2 == rng3) << std::endl;

  // 1 is printed

  // re-seed rng2 with a different seed
  rng2.seed(7);

  // compare rng2 and rng3
  std::cout << (rng2 == rng3) << std::endl;

  // 0 is printed

  return 0;
}
```

**Note**:
Inexperienced users should not use this class template directly. Instead, use <code>minstd&#95;rand</code> or <code>minstd&#95;rand0</code>.

**Template Parameters**:
* **`UIntType`** The type of unsigned integer to produce. 
* **`a`** The multiplier used in the generation algorithm. 
* **`c`** The increment used in the generation algorithm. 
* **`m`** The modulus used in the generation algorithm.

**See**:
* <a href="{{ site.baseurl }}/api/groups/group__predefined__random.html#typedef-minstd-rand">thrust::random::minstd_rand</a>
* <a href="{{ site.baseurl }}/api/groups/group__predefined__random.html#typedef-minstd-rand0">thrust::random::minstd_rand0</a>

<code class="doxybook">
<span>#include <thrust/random/linear_congruential_engine.h></span><br>
<span>template &lt;typename UIntType,</span>
<span>&nbsp;&nbsp;UIntType a,</span>
<span>&nbsp;&nbsp;UIntType c,</span>
<span>&nbsp;&nbsp;UIntType m&gt;</span>
<span>class thrust::random::linear&#95;congruential&#95;engine {</span>
<span>public:</span><span class="doxybook-comment"><code>&nbsp;&nbsp;</code>
/* The type of the unsigned integer produced by this <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1linear__congruential__engine.html">linear&#95;congruential&#95;engine</a></code>.  */</span><span>&nbsp;&nbsp;typedef <i>see below</i> <b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1linear__congruential__engine.html#typedef-result-type">result&#95;type</a></b>;</span>
<br>
<span>&nbsp;&nbsp;static const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1linear__congruential__engine.html#typedef-result-type">result_type</a> <b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1linear__congruential__engine.html#variable-multiplier">multiplier</a></b> = <i>see below</i>;</span>
<br>
<span>&nbsp;&nbsp;static const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1linear__congruential__engine.html#typedef-result-type">result_type</a> <b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1linear__congruential__engine.html#variable-increment">increment</a></b> = <i>see below</i>;</span>
<br>
<span>&nbsp;&nbsp;static const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1linear__congruential__engine.html#typedef-result-type">result_type</a> <b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1linear__congruential__engine.html#variable-modulus">modulus</a></b> = <i>see below</i>;</span>
<br>
<span>&nbsp;&nbsp;static const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1linear__congruential__engine.html#typedef-result-type">result_type</a> <b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1linear__congruential__engine.html#variable-min">min</a></b> = <i>see below</i>;</span>
<br>
<span>&nbsp;&nbsp;static const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1linear__congruential__engine.html#typedef-result-type">result_type</a> <b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1linear__congruential__engine.html#variable-max">max</a></b> = <i>see below</i>;</span>
<br>
<span>&nbsp;&nbsp;static const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1linear__congruential__engine.html#typedef-result-type">result_type</a> <b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1linear__congruential__engine.html#variable-default-seed">default&#95;seed</a></b> = <i>see below</i>;</span>
<br>
<span>&nbsp;&nbsp;explicit __host__ __device__ </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1linear__congruential__engine.html#function-linear-congruential-engine">linear&#95;congruential&#95;engine</a></b>(<a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1linear__congruential__engine.html#typedef-result-type">result_type</a> s = default&#95;seed) = default;</span>
<br>
<span>&nbsp;&nbsp;__host__ __device__ void </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1linear__congruential__engine.html#function-seed">seed</a></b>(<a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1linear__congruential__engine.html#typedef-result-type">result_type</a> s = default&#95;seed) = default;</span>
<br>
<span>&nbsp;&nbsp;__host__ __device__ <a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1linear__congruential__engine.html#typedef-result-type">result_type</a> </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1linear__congruential__engine.html#function-operator()">operator()</a></b>(void);</span>
<br>
<span>&nbsp;&nbsp;__host__ __device__ void </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1linear__congruential__engine.html#function-discard">discard</a></b>(unsigned long long z);</span>
<span>};</span>
</code>

## Member Types

<h3 id="typedef-result-type">
Typedef <code>thrust::random::linear&#95;congruential&#95;engine::result&#95;type</code>
</h3>

<code class="doxybook">
<span>typedef UIntType<b>result_type</b>;</span></code>
The type of the unsigned integer produced by this <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1linear__congruential__engine.html">linear&#95;congruential&#95;engine</a></code>. 


## Member Variables

<h3 id="variable-multiplier">
Variable <code>thrust::random::linear&#95;congruential&#95;engine::multiplier</code>
</h3>

<code class="doxybook">
<span>static const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1linear__congruential__engine.html#typedef-result-type">result_type</a> <b>multiplier</b> = a;</span></code>
The multiplier used in the generation algorithm. 

<h3 id="variable-increment">
Variable <code>thrust::random::linear&#95;congruential&#95;engine::increment</code>
</h3>

<code class="doxybook">
<span>static const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1linear__congruential__engine.html#typedef-result-type">result_type</a> <b>increment</b> = c;</span></code>
The increment used in the generation algorithm. 

<h3 id="variable-modulus">
Variable <code>thrust::random::linear&#95;congruential&#95;engine::modulus</code>
</h3>

<code class="doxybook">
<span>static const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1linear__congruential__engine.html#typedef-result-type">result_type</a> <b>modulus</b> = m;</span></code>
The modulus used in the generation algorithm. 

<h3 id="variable-min">
Variable <code>thrust::random::linear&#95;congruential&#95;engine::min</code>
</h3>

<code class="doxybook">
<span>static const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1linear__congruential__engine.html#typedef-result-type">result_type</a> <b>min</b> = c == 0u ? 1u : 0u;</span></code>
The smallest value this <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1linear__congruential__engine.html">linear&#95;congruential&#95;engine</a></code> may potentially produce. 

<h3 id="variable-max">
Variable <code>thrust::random::linear&#95;congruential&#95;engine::max</code>
</h3>

<code class="doxybook">
<span>static const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1linear__congruential__engine.html#typedef-result-type">result_type</a> <b>max</b> = m - 1u;</span></code>
The largest value this <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1linear__congruential__engine.html">linear&#95;congruential&#95;engine</a></code> may potentially produce. 

<h3 id="variable-default-seed">
Variable <code>thrust::random::linear&#95;congruential&#95;engine::default&#95;seed</code>
</h3>

<code class="doxybook">
<span>static const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1linear__congruential__engine.html#typedef-result-type">result_type</a> <b>default_seed</b> = 1u;</span></code>
The default seed of this <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1linear__congruential__engine.html">linear&#95;congruential&#95;engine</a></code>. 


## Member Functions

<h3 id="function-linear-congruential-engine">
Function <code>thrust::random::linear&#95;congruential&#95;engine::linear&#95;congruential&#95;engine</code>
</h3>

<code class="doxybook">
<span>explicit __host__ __device__ </span><span><b>linear_congruential_engine</b>(<a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1linear__congruential__engine.html#typedef-result-type">result_type</a> s = default&#95;seed) = default;</span></code>
This constructor, which optionally accepts a seed, initializes a new <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1linear__congruential__engine.html">linear&#95;congruential&#95;engine</a></code>.

**Function Parameters**:
**`s`**: The seed used to intialize this <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1linear__congruential__engine.html">linear&#95;congruential&#95;engine</a>'s</code> state. 

<h3 id="function-seed">
Function <code>thrust::random::linear&#95;congruential&#95;engine::seed</code>
</h3>

<code class="doxybook">
<span>__host__ __device__ void </span><span><b>seed</b>(<a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1linear__congruential__engine.html#typedef-result-type">result_type</a> s = default&#95;seed) = default;</span></code>
This method initializes this <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1linear__congruential__engine.html">linear&#95;congruential&#95;engine</a>'s</code> state, and optionally accepts a seed value.

**Function Parameters**:
**`s`**: The seed used to initializes this <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1linear__congruential__engine.html">linear&#95;congruential&#95;engine</a>'s</code> state. 

<h3 id="function-operator()">
Function <code>thrust::random::linear&#95;congruential&#95;engine::operator()</code>
</h3>

<code class="doxybook">
<span>__host__ __device__ <a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1linear__congruential__engine.html#typedef-result-type">result_type</a> </span><span><b>operator()</b>(void);</span></code>
This member function produces a new random value and updates this <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1linear__congruential__engine.html">linear&#95;congruential&#95;engine</a>'s</code> state. 

**Returns**:
A new random number. 

<h3 id="function-discard">
Function <code>thrust::random::linear&#95;congruential&#95;engine::discard</code>
</h3>

<code class="doxybook">
<span>__host__ __device__ void </span><span><b>discard</b>(unsigned long long z);</span></code>
This member function advances this <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1linear__congruential__engine.html">linear&#95;congruential&#95;engine</a>'s</code> state a given number of times and discards the results.

**Note**:
This function is provided because an implementation may be able to accelerate it. 

**Function Parameters**:
**`z`**: The number of random values to discard. 


