---
title: thrust::random::discard_block_engine
summary: A discard_block_engine adapts an existing base random number engine and produces random values by discarding some of the values returned by its base engine. Each cycle of the compound engine begins by returning r values successively produced by the base engine and ends by discarding p-r such values. The engine's state is the state of its base engine followed by the number of calls to operator() that have occurred since the beginning of the current cycle. 
parent: Random Number Engine Adaptor Class Templates
grand_parent: Random Number Generation
nav_exclude: true
has_children: true
has_toc: false
---

# Class `thrust::random::discard_block_engine`

A <code>discard&#95;block&#95;engine</code> adapts an existing base random number engine and produces random values by discarding some of the values returned by its base engine. Each cycle of the compound engine begins by returning <code>r</code> values successively produced by the base engine and ends by discarding <code>p-r</code> such values. The engine's state is the state of its base engine followed by the number of calls to <code>operator()</code> that have occurred since the beginning of the current cycle. 


The following code snippet shows an example of using a <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1discard__block__engine.html">discard&#95;block&#95;engine</a></code> instance:



```cpp
#include <thrust/random/linear_congruential_engine.h>
#include <thrust/random/discard_block_engine.h>
#include <iostream>

int main(void)
{
  // create a discard_block_engine from minstd_rand, with a cycle length of 13
  // keep every first 10 values, and discard the next 3
  thrust::discard_block_engine<thrust::minstd_rand, 13, 10> rng;

  // print a random number to standard output
  std::cout << rng() << std::endl;

  return 0;
}
```

**Template Parameters**:
* **`Engine`** The type of the base random number engine to adapt. 
* **`p`** The discard cycle length. 
* **`r`** The number of values to return of the base engine. Because <code>p-r</code> will be discarded, <code>r &lt;= p</code>.

<code class="doxybook">
<span>#include <thrust/random/discard_block_engine.h></span><br>
<span>template &lt;typename Engine,</span>
<span>&nbsp;&nbsp;size_t p,</span>
<span>&nbsp;&nbsp;size_t r&gt;</span>
<span>class thrust::random::discard&#95;block&#95;engine {</span>
<span>public:</span><span class="doxybook-comment"><code>&nbsp;&nbsp;</code>
/* The type of the adapted base random number engine.  */</span><span>&nbsp;&nbsp;typedef <i>see below</i> <b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1discard__block__engine.html#typedef-base-type">base&#95;type</a></b>;</span>
<br>
<span class="doxybook-comment"><code>&nbsp;&nbsp;</code>
/* The type of the unsigned integer produced by this <code>linear&#95;congruential&#95;engine</code>.  */</span><span>&nbsp;&nbsp;typedef <i>see below</i> <b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1discard__block__engine.html#typedef-result-type">result&#95;type</a></b>;</span>
<br>
<span>&nbsp;&nbsp;static const size_t <b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1discard__block__engine.html#variable-block-size">block&#95;size</a></b> = <i>see below</i>;</span>
<br>
<span>&nbsp;&nbsp;static const size_t <b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1discard__block__engine.html#variable-used-block">used&#95;block</a></b> = <i>see below</i>;</span>
<br>
<span>&nbsp;&nbsp;static const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1discard__block__engine.html#typedef-result-type">result_type</a> <b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1discard__block__engine.html#variable-min">min</a></b> = <i>see below</i>;</span>
<br>
<span>&nbsp;&nbsp;static const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1discard__block__engine.html#typedef-result-type">result_type</a> <b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1discard__block__engine.html#variable-max">max</a></b> = <i>see below</i>;</span>
<br>
<span>&nbsp;&nbsp;__host__ __device__ </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1discard__block__engine.html#function-discard-block-engine">discard&#95;block&#95;engine</a></b>();</span>
<br>
<span>&nbsp;&nbsp;explicit __host__ __device__ </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1discard__block__engine.html#function-discard-block-engine">discard&#95;block&#95;engine</a></b>(const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1discard__block__engine.html#typedef-base-type">base_type</a> & urng);</span>
<br>
<span>&nbsp;&nbsp;explicit __host__ __device__ </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1discard__block__engine.html#function-discard-block-engine">discard&#95;block&#95;engine</a></b>(<a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1discard__block__engine.html#typedef-result-type">result_type</a> s);</span>
<br>
<span>&nbsp;&nbsp;__host__ __device__ void </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1discard__block__engine.html#function-seed">seed</a></b>(void);</span>
<br>
<span>&nbsp;&nbsp;__host__ __device__ void </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1discard__block__engine.html#function-seed">seed</a></b>(<a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1discard__block__engine.html#typedef-result-type">result_type</a> s);</span>
<br>
<span>&nbsp;&nbsp;__host__ __device__ <a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1discard__block__engine.html#typedef-result-type">result_type</a> </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1discard__block__engine.html#function-operator()">operator()</a></b>(void);</span>
<br>
<span>&nbsp;&nbsp;__host__ __device__ void </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1discard__block__engine.html#function-discard">discard</a></b>(unsigned long long z);</span>
<br>
<span>&nbsp;&nbsp;__host__ __device__ const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1discard__block__engine.html#typedef-base-type">base_type</a> & </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1discard__block__engine.html#function-base">base</a></b>(void) const;</span>
<span>};</span>
</code>

## Member Types

<h3 id="typedef-base-type">
Typedef <code>thrust::random::discard&#95;block&#95;engine::base&#95;type</code>
</h3>

<code class="doxybook">
<span>typedef Engine<b>base_type</b>;</span></code>
The type of the adapted base random number engine. 

<h3 id="typedef-result-type">
Typedef <code>thrust::random::discard&#95;block&#95;engine::result&#95;type</code>
</h3>

<code class="doxybook">
<span>typedef base_type::result_type<b>result_type</b>;</span></code>
The type of the unsigned integer produced by this <code>linear&#95;congruential&#95;engine</code>. 


## Member Variables

<h3 id="variable-block-size">
Variable <code>thrust::random::discard&#95;block&#95;engine::block&#95;size</code>
</h3>

<code class="doxybook">
<span>static const size_t <b>block_size</b> = p;</span></code>
The length of the production cycle. 

<h3 id="variable-used-block">
Variable <code>thrust::random::discard&#95;block&#95;engine::used&#95;block</code>
</h3>

<code class="doxybook">
<span>static const size_t <b>used_block</b> = r;</span></code>
The number of used numbers per production cycle. 

<h3 id="variable-min">
Variable <code>thrust::random::discard&#95;block&#95;engine::min</code>
</h3>

<code class="doxybook">
<span>static const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1discard__block__engine.html#typedef-result-type">result_type</a> <b>min</b> = base&#95;type::min;</span></code>
The smallest value this <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1discard__block__engine.html">discard&#95;block&#95;engine</a></code> may potentially produce. 

<h3 id="variable-max">
Variable <code>thrust::random::discard&#95;block&#95;engine::max</code>
</h3>

<code class="doxybook">
<span>static const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1discard__block__engine.html#typedef-result-type">result_type</a> <b>max</b> = base&#95;type::max;</span></code>
The largest value this <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1discard__block__engine.html">discard&#95;block&#95;engine</a></code> may potentially produce. 


## Member Functions

<h3 id="function-discard-block-engine">
Function <code>thrust::random::discard&#95;block&#95;engine::discard&#95;block&#95;engine</code>
</h3>

<code class="doxybook">
<span>__host__ __device__ </span><span><b>discard_block_engine</b>();</span></code>
This constructor constructs a new <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1discard__block__engine.html">discard&#95;block&#95;engine</a></code> and constructs its <code>base&#95;type</code> engine using its null constructor. 

<h3 id="function-discard-block-engine">
Function <code>thrust::random::discard&#95;block&#95;engine::discard&#95;block&#95;engine</code>
</h3>

<code class="doxybook">
<span>explicit __host__ __device__ </span><span><b>discard_block_engine</b>(const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1discard__block__engine.html#typedef-base-type">base_type</a> & urng);</span></code>
This constructor constructs a new <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1discard__block__engine.html">discard&#95;block&#95;engine</a></code> using a given <code>base&#95;type</code> engine to initialize its adapted base engine.

**Function Parameters**:
**`urng`**: A <code>base&#95;type</code> to use to initialize this <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1discard__block__engine.html">discard&#95;block&#95;engine</a>'s</code> adapted base engine. 

<h3 id="function-discard-block-engine">
Function <code>thrust::random::discard&#95;block&#95;engine::discard&#95;block&#95;engine</code>
</h3>

<code class="doxybook">
<span>explicit __host__ __device__ </span><span><b>discard_block_engine</b>(<a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1discard__block__engine.html#typedef-result-type">result_type</a> s);</span></code>
This constructor initializes a new <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1discard__block__engine.html">discard&#95;block&#95;engine</a></code> with a given seed.

**Function Parameters**:
**`s`**: The seed used to intialize this <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1discard__block__engine.html">discard&#95;block&#95;engine</a>'s</code> adapted base engine. 

<h3 id="function-seed">
Function <code>thrust::random::discard&#95;block&#95;engine::seed</code>
</h3>

<code class="doxybook">
<span>__host__ __device__ void </span><span><b>seed</b>(void);</span></code>
This method initializes the state of this <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1discard__block__engine.html">discard&#95;block&#95;engine</a>'s</code> adapted base engine by using its <code>default&#95;seed</code> value. 

<h3 id="function-seed">
Function <code>thrust::random::discard&#95;block&#95;engine::seed</code>
</h3>

<code class="doxybook">
<span>__host__ __device__ void </span><span><b>seed</b>(<a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1discard__block__engine.html#typedef-result-type">result_type</a> s);</span></code>
This method initializes the state of this <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1discard__block__engine.html">discard&#95;block&#95;engine</a>'s</code> adapted base engine by using the given seed.

**Function Parameters**:
**`s`**: The seed with which to intialize this <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1discard__block__engine.html">discard&#95;block&#95;engine</a>'s</code> adapted base engine. 

<h3 id="function-operator()">
Function <code>thrust::random::discard&#95;block&#95;engine::operator()</code>
</h3>

<code class="doxybook">
<span>__host__ __device__ <a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1discard__block__engine.html#typedef-result-type">result_type</a> </span><span><b>operator()</b>(void);</span></code>
This member function produces a new random value and updates this <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1discard__block__engine.html">discard&#95;block&#95;engine</a>'s</code> state. 

**Returns**:
A new random number. 

<h3 id="function-discard">
Function <code>thrust::random::discard&#95;block&#95;engine::discard</code>
</h3>

<code class="doxybook">
<span>__host__ __device__ void </span><span><b>discard</b>(unsigned long long z);</span></code>
This member function advances this <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1discard__block__engine.html">discard&#95;block&#95;engine</a>'s</code> state a given number of times and discards the results.

**Note**:
This function is provided because an implementation may be able to accelerate it. 

**Function Parameters**:
**`z`**: The number of random values to discard. 

<h3 id="function-base">
Function <code>thrust::random::discard&#95;block&#95;engine::base</code>
</h3>

<code class="doxybook">
<span>__host__ __device__ const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1discard__block__engine.html#typedef-base-type">base_type</a> & </span><span><b>base</b>(void) const;</span></code>
This member function returns a const reference to this <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1discard__block__engine.html">discard&#95;block&#95;engine</a>'s</code> adapted base engine.

**Returns**:
A const reference to the base engine this <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1discard__block__engine.html">discard&#95;block&#95;engine</a></code> adapts. 


