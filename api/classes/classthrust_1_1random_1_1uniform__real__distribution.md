---
title: thrust::random::uniform_real_distribution
summary: A uniform_real_distribution random number distribution produces floating point uniform random numbers from a half-open interval. 
parent: Random Number Distributions Class Templates
grand_parent: Random Number Generation
nav_exclude: true
has_children: true
has_toc: false
---

# Class `thrust::random::uniform_real_distribution`

A <code>uniform&#95;real&#95;distribution</code> random number distribution produces floating point uniform random numbers from a half-open interval. 


The following code snippet demonstrates examples of using a <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1uniform__real__distribution.html">uniform&#95;real&#95;distribution</a></code> with a random number engine to produce random integers drawn from a given range:



```cpp
#include <thrust/random/linear_congruential_engine.h>
#include <thrust/random/uniform_real_distribution.h>

int main(void)
{
  // create a minstd_rand object to act as our source of randomness
  thrust::minstd_rand rng;

  // create a uniform_real_distribution to produce floats from [-7,13)
  thrust::uniform_real_distribution<float> dist(-7,13);

  // write a random number from the range [-7,13) to standard output
  std::cout << dist(rng) << std::endl;

  // write the range of the distribution, just in case we forgot
  std::cout << dist.min() << std::endl;

  // -7.0 is printed

  std::cout << dist.max() << std::endl;

  // 13.0 is printed

  // write the parameters of the distribution (which happen to be the bounds) to standard output
  std::cout << dist.a() << std::endl;

  // -7.0 is printed

  std::cout << dist.b() << std::endl;

  // 13.0 is printed

  return 0;
}
```

**Template Parameters**:
**`RealType`**: The type of floating point number to produce.

<code class="doxybook">
<span>#include <thrust/random/uniform_real_distribution.h></span><br>
<span>template &lt;typename RealType = double&gt;</span>
<span>class thrust::random::uniform&#95;real&#95;distribution {</span>
<span>public:</span><span class="doxybook-comment"><code>&nbsp;&nbsp;</code>
/* The type of the floating point number produced by this <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1uniform__real__distribution.html">uniform&#95;real&#95;distribution</a></code>.  */</span><span>&nbsp;&nbsp;typedef <i>see below</i> <b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1uniform__real__distribution.html#typedef-result-type">result&#95;type</a></b>;</span>
<br>
<span class="doxybook-comment"><code>&nbsp;&nbsp;</code>
/* The type of the object encapsulating this <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1uniform__real__distribution.html">uniform&#95;real&#95;distribution</a>'s</code> parameters.  */</span><span>&nbsp;&nbsp;typedef <i>see below</i> <b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1uniform__real__distribution.html#typedef-param-type">param&#95;type</a></b>;</span>
<br>
<span>&nbsp;&nbsp;explicit __host__ __device__ </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1uniform__real__distribution.html#function-uniform-real-distribution">uniform&#95;real&#95;distribution</a></b>(RealType a = 0.0,</span>
<span>&nbsp;&nbsp;&nbsp;&nbsp;RealType b = 1.0);</span>
<br>
<span>&nbsp;&nbsp;explicit __host__ __device__ </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1uniform__real__distribution.html#function-uniform-real-distribution">uniform&#95;real&#95;distribution</a></b>(const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1uniform__real__distribution.html#typedef-param-type">param_type</a> & parm);</span>
<br>
<span>&nbsp;&nbsp;__host__ __device__ void </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1uniform__real__distribution.html#function-reset">reset</a></b>(void);</span>
<br>
<span>&nbsp;&nbsp;template &lt;typename UniformRandomNumberGenerator&gt;</span>
<span>&nbsp;&nbsp;__host__ __device__ <a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1uniform__real__distribution.html#typedef-result-type">result_type</a> </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1uniform__real__distribution.html#function-operator()">operator()</a></b>(UniformRandomNumberGenerator & urng);</span>
<br>
<span>&nbsp;&nbsp;template &lt;typename UniformRandomNumberGenerator&gt;</span>
<span>&nbsp;&nbsp;__host__ __device__ <a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1uniform__real__distribution.html#typedef-result-type">result_type</a> </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1uniform__real__distribution.html#function-operator()">operator()</a></b>(UniformRandomNumberGenerator & urng,</span>
<span>&nbsp;&nbsp;&nbsp;&nbsp;const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1uniform__real__distribution.html#typedef-param-type">param_type</a> & parm);</span>
<br>
<span>&nbsp;&nbsp;__host__ __device__ <a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1uniform__real__distribution.html#typedef-result-type">result_type</a> </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1uniform__real__distribution.html#function-a">a</a></b>(void) const;</span>
<br>
<span>&nbsp;&nbsp;__host__ __device__ <a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1uniform__real__distribution.html#typedef-result-type">result_type</a> </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1uniform__real__distribution.html#function-b">b</a></b>(void) const;</span>
<br>
<span>&nbsp;&nbsp;__host__ __device__ <a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1uniform__real__distribution.html#typedef-param-type">param_type</a> </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1uniform__real__distribution.html#function-param">param</a></b>(void) const;</span>
<br>
<span>&nbsp;&nbsp;__host__ __device__ void </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1uniform__real__distribution.html#function-param">param</a></b>(const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1uniform__real__distribution.html#typedef-param-type">param_type</a> & parm);</span>
<br>
<span>&nbsp;&nbsp;__host__ __device__ <a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1uniform__real__distribution.html#typedef-result-type">result_type</a> min </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1uniform__real__distribution.html#function-thrust-prevent-macro-substitution">THRUST&#95;PREVENT&#95;MACRO&#95;SUBSTITUTION</a></b>(void) const;</span>
<br>
<span>&nbsp;&nbsp;__host__ __device__ <a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1uniform__real__distribution.html#typedef-result-type">result_type</a> max </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1uniform__real__distribution.html#function-thrust-prevent-macro-substitution">THRUST&#95;PREVENT&#95;MACRO&#95;SUBSTITUTION</a></b>(void) const;</span>
<span>};</span>
</code>

## Member Types

<h3 id="typedef-result-type">
Typedef <code>thrust::random::uniform&#95;real&#95;distribution::result&#95;type</code>
</h3>

<code class="doxybook">
<span>typedef RealType<b>result_type</b>;</span></code>
The type of the floating point number produced by this <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1uniform__real__distribution.html">uniform&#95;real&#95;distribution</a></code>. 

<h3 id="typedef-param-type">
Typedef <code>thrust::random::uniform&#95;real&#95;distribution::param&#95;type</code>
</h3>

<code class="doxybook">
<span>typedef <a href="{{ site.baseurl }}/api/classes/structthrust_1_1pair.html">thrust::pair</a>< RealType, RealType ><b>param_type</b>;</span></code>
The type of the object encapsulating this <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1uniform__real__distribution.html">uniform&#95;real&#95;distribution</a>'s</code> parameters. 


## Member Functions

<h3 id="function-uniform-real-distribution">
Function <code>thrust::random::uniform&#95;real&#95;distribution::uniform&#95;real&#95;distribution</code>
</h3>

<code class="doxybook">
<span>explicit __host__ __device__ </span><span><b>uniform_real_distribution</b>(RealType a = 0.0,</span>
<span>&nbsp;&nbsp;RealType b = 1.0);</span></code>
This constructor creates a new <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1uniform__real__distribution.html">uniform&#95;real&#95;distribution</a></code> from two values defining the half-open interval of the distribution.

**Function Parameters**:
* **`a`** The smallest floating point number to potentially produce. Defaults to <code>0.0</code>. 
* **`b`** The smallest number larger than the largest floating point number to potentially produce. Defaults to <code>1.0</code>. 

<h3 id="function-uniform-real-distribution">
Function <code>thrust::random::uniform&#95;real&#95;distribution::uniform&#95;real&#95;distribution</code>
</h3>

<code class="doxybook">
<span>explicit __host__ __device__ </span><span><b>uniform_real_distribution</b>(const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1uniform__real__distribution.html#typedef-param-type">param_type</a> & parm);</span></code>
This constructor creates a new <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1uniform__real__distribution.html">uniform&#95;real&#95;distribution</a></code> from a <code>param&#95;type</code> object encapsulating the range of the distribution.

**Function Parameters**:
**`parm`**: A <code>param&#95;type</code> object encapsulating the parameters (i.e., the range) of the distribution. 

<h3 id="function-reset">
Function <code>thrust::random::uniform&#95;real&#95;distribution::reset</code>
</h3>

<code class="doxybook">
<span>__host__ __device__ void </span><span><b>reset</b>(void);</span></code>
This does nothing. It is included to conform to the requirements of the RandomDistribution concept. 

<h3 id="function-operator()">
Function <code>thrust::random::uniform&#95;real&#95;distribution::operator()</code>
</h3>

<code class="doxybook">
<span>template &lt;typename UniformRandomNumberGenerator&gt;</span>
<span>__host__ __device__ <a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1uniform__real__distribution.html#typedef-result-type">result_type</a> </span><span><b>operator()</b>(UniformRandomNumberGenerator & urng);</span></code>
This method produces a new uniform random integer drawn from this <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1uniform__real__distribution.html">uniform&#95;real&#95;distribution</a>'s</code> range using a <code>UniformRandomNumberGenerator</code> as a source of randomness.

**Function Parameters**:
**`urng`**: The <code>UniformRandomNumberGenerator</code> to use as a source of randomness. 

<h3 id="function-operator()">
Function <code>thrust::random::uniform&#95;real&#95;distribution::operator()</code>
</h3>

<code class="doxybook">
<span>template &lt;typename UniformRandomNumberGenerator&gt;</span>
<span>__host__ __device__ <a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1uniform__real__distribution.html#typedef-result-type">result_type</a> </span><span><b>operator()</b>(UniformRandomNumberGenerator & urng,</span>
<span>&nbsp;&nbsp;const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1uniform__real__distribution.html#typedef-param-type">param_type</a> & parm);</span></code>
This method produces a new uniform random integer as if by creating a new <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1uniform__real__distribution.html">uniform&#95;real&#95;distribution</a></code> from the given <code>param&#95;type</code> object, and calling its <code>operator()</code> method with the given <code>UniformRandomNumberGenerator</code> as a source of randomness.

**Function Parameters**:
* **`urng`** The <code>UniformRandomNumberGenerator</code> to use as a source of randomness. 
* **`parm`** A <code>param&#95;type</code> object encapsulating the parameters of the <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1uniform__real__distribution.html">uniform&#95;real&#95;distribution</a></code> to draw from. 

<h3 id="function-a">
Function <code>thrust::random::uniform&#95;real&#95;distribution::a</code>
</h3>

<code class="doxybook">
<span>__host__ __device__ <a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1uniform__real__distribution.html#typedef-result-type">result_type</a> </span><span><b>a</b>(void) const;</span></code>
This method returns the value of the parameter with which this <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1uniform__real__distribution.html">uniform&#95;real&#95;distribution</a></code> was constructed.

**Returns**:
The lower bound of this <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1uniform__real__distribution.html">uniform&#95;real&#95;distribution</a>'s</code> half-open interval. 

<h3 id="function-b">
Function <code>thrust::random::uniform&#95;real&#95;distribution::b</code>
</h3>

<code class="doxybook">
<span>__host__ __device__ <a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1uniform__real__distribution.html#typedef-result-type">result_type</a> </span><span><b>b</b>(void) const;</span></code>
This method returns the value of the parameter with which this <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1uniform__real__distribution.html">uniform&#95;real&#95;distribution</a></code> was constructed.

**Returns**:
The upper bound of this <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1uniform__real__distribution.html">uniform&#95;real&#95;distribution</a>'s</code> half-open interval. 

<h3 id="function-param">
Function <code>thrust::random::uniform&#95;real&#95;distribution::param</code>
</h3>

<code class="doxybook">
<span>__host__ __device__ <a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1uniform__real__distribution.html#typedef-param-type">param_type</a> </span><span><b>param</b>(void) const;</span></code>
This method returns a <code>param&#95;type</code> object encapsulating the parameters with which this <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1uniform__real__distribution.html">uniform&#95;real&#95;distribution</a></code> was constructed.

**Returns**:
A <code>param&#95;type</code> object enapsulating the half-open interval of this <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1uniform__real__distribution.html">uniform&#95;real&#95;distribution</a></code>. 

<h3 id="function-param">
Function <code>thrust::random::uniform&#95;real&#95;distribution::param</code>
</h3>

<code class="doxybook">
<span>__host__ __device__ void </span><span><b>param</b>(const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1uniform__real__distribution.html#typedef-param-type">param_type</a> & parm);</span></code>
This method changes the parameters of this <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1uniform__real__distribution.html">uniform&#95;real&#95;distribution</a></code> using the values encapsulated in a given <code>param&#95;type</code> object.

**Function Parameters**:
**`parm`**: A <code>param&#95;type</code> object encapsulating the new half-open interval of this <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1uniform__real__distribution.html">uniform&#95;real&#95;distribution</a></code>. 

<h3 id="function-thrust-prevent-macro-substitution">
Function <code>thrust::random::uniform&#95;real&#95;distribution::THRUST&#95;PREVENT&#95;MACRO&#95;SUBSTITUTION</code>
</h3>

<code class="doxybook">
<span>__host__ __device__ <a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1uniform__real__distribution.html#typedef-result-type">result_type</a> min </span><span><b>THRUST_PREVENT_MACRO_SUBSTITUTION</b>(void) const;</span></code>
This method returns the smallest floating point number this <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1uniform__real__distribution.html">uniform&#95;real&#95;distribution</a></code> can potentially produce.

**Returns**:
The lower bound of this <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1uniform__real__distribution.html">uniform&#95;real&#95;distribution</a>'s</code> half-open interval. 

<h3 id="function-thrust-prevent-macro-substitution">
Function <code>thrust::random::uniform&#95;real&#95;distribution::THRUST&#95;PREVENT&#95;MACRO&#95;SUBSTITUTION</code>
</h3>

<code class="doxybook">
<span>__host__ __device__ <a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1uniform__real__distribution.html#typedef-result-type">result_type</a> max </span><span><b>THRUST_PREVENT_MACRO_SUBSTITUTION</b>(void) const;</span></code>
This method returns the smallest number larger than largest floating point number this <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1uniform__real__distribution.html">uniform&#95;real&#95;distribution</a></code> can potentially produce.

**Returns**:
The upper bound of this <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1uniform__real__distribution.html">uniform&#95;real&#95;distribution</a>'s</code> half-open interval. 


