---
title: thrust::binary_function
parent: Function Object Adaptors
grand_parent: Function Objects
nav_exclude: true
has_children: true
has_toc: false
---

# Struct `thrust::binary_function`

<code><a href="{{ site.baseurl }}/api/classes/structthrust_1_1binary__function.html">binary&#95;function</a></code> is an empty base class: it contains no member functions or member variables, but only type information. The only reason it exists is to make it more convenient to define types that are models of the concept Adaptable Binary Function. Specifically, any model of Adaptable Binary Function must define nested <code>typedefs</code>. Those <code>typedefs</code> are provided by the base class <code><a href="{{ site.baseurl }}/api/classes/structthrust_1_1binary__function.html">binary&#95;function</a></code>.

The following code snippet demonstrates how to construct an Adaptable Binary Function using <code><a href="{{ site.baseurl }}/api/classes/structthrust_1_1binary__function.html">binary&#95;function</a></code>.



```cpp
struct exponentiate : public thrust::binary_function<float,float,float>
{
  __host__ __device__
  float operator()(float x, float y) { return powf(x,y); }
};
```

**Note**:
Because C++11 language support makes the functionality of <code><a href="{{ site.baseurl }}/api/classes/structthrust_1_1binary__function.html">binary&#95;function</a></code> obsolete, its use is optional if C++11 language features are enabled.

**See**:
* <a href="https://en.cppreference.com/w/cpp/utility/functional/binary_function">https://en.cppreference.com/w/cpp/utility/functional/binary_function</a>
* <a href="{{ site.baseurl }}/api/classes/structthrust_1_1unary__function.html">unary_function</a>

<code class="doxybook">
<span>#include <thrust/functional.h></span><br>
<span>template &lt;typename Argument1,</span>
<span>&nbsp;&nbsp;typename Argument2,</span>
<span>&nbsp;&nbsp;typename Result&gt;</span>
<span>struct thrust::binary&#95;function {</span>
<span>public:</span><span class="doxybook-comment"><code>&nbsp;&nbsp;</code>
/* The type of the function object's first argument.  */</span><span>&nbsp;&nbsp;typedef <i>see below</i> <b><a href="{{ site.baseurl }}/api/classes/structthrust_1_1binary__function.html#typedef-first-argument-type">first&#95;argument&#95;type</a></b>;</span>
<br>
<span class="doxybook-comment"><code>&nbsp;&nbsp;</code>
/* The type of the function object's second argument.  */</span><span>&nbsp;&nbsp;typedef <i>see below</i> <b><a href="{{ site.baseurl }}/api/classes/structthrust_1_1binary__function.html#typedef-second-argument-type">second&#95;argument&#95;type</a></b>;</span>
<br>
<span class="doxybook-comment"><code>&nbsp;&nbsp;</code>
/* The type of the function object's result;.  */</span><span>&nbsp;&nbsp;typedef <i>see below</i> <b><a href="{{ site.baseurl }}/api/classes/structthrust_1_1binary__function.html#typedef-result-type">result&#95;type</a></b>;</span>
<span>};</span>
</code>

## Member Types

<h3 id="typedef-first-argument-type">
Typedef <code>thrust::binary&#95;function::first&#95;argument&#95;type</code>
</h3>

<code class="doxybook">
<span>typedef Argument1<b>first_argument_type</b>;</span></code>
The type of the function object's first argument. 

<h3 id="typedef-second-argument-type">
Typedef <code>thrust::binary&#95;function::second&#95;argument&#95;type</code>
</h3>

<code class="doxybook">
<span>typedef Argument2<b>second_argument_type</b>;</span></code>
The type of the function object's second argument. 

<h3 id="typedef-result-type">
Typedef <code>thrust::binary&#95;function::result&#95;type</code>
</h3>

<code class="doxybook">
<span>typedef Result<b>result_type</b>;</span></code>
The type of the function object's result;. 


