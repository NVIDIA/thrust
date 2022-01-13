---
title: thrust::placeholders
summary: Facilities for constructing simple functions inline. 
parent: Placeholder Objects
grand_parent: Function Objects
nav_exclude: true
has_children: true
has_toc: false
---

# Namespace `thrust::placeholders`

Facilities for constructing simple functions inline. 

Objects in the <code><a href="{{ site.baseurl }}/api/namespaces/namespacethrust_1_1placeholders.html">thrust::placeholders</a></code> namespace may be used to create simple arithmetic functions inline in an algorithm invocation. Combining placeholders such as <code>&#95;1</code> and <code>&#95;2</code> with arithmetic operations such as <code>+</code> creates an unnamed function object which applies the operation to their arguments.

The type of placeholder objects is implementation-defined.

The following code snippet demonstrates how to use the placeholders <code>&#95;1</code> and <code>&#95;2</code> with <code>thrust::transform</code> to implement the SAXPY computation:



```cpp
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/functional.h>

int main()
{
  thrust::device_vector<float> x(4), y(4);
  x[0] = 1;
  x[1] = 2;
  x[2] = 3;
  x[3] = 4;

  y[0] = 1;
  y[1] = 1;
  y[2] = 1;
  y[3] = 1;

  float a = 2.0f;

  using namespace thrust::placeholders;

  thrust::transform(x.begin(), x.end(), y.begin(), y.begin(),
    a * _1 + _2
  );

  // y is now {3, 5, 7, 9}
}
```

<code class="doxybook">
<span>namespace thrust::placeholders {</span>
<br>
<span>THRUST_INLINE_CONSTANT thrust::detail::functional::placeholder< 0 >::type <b><a href="{{ site.baseurl }}/api/namespaces/namespacethrust_1_1placeholders.html#variable--1">&#95;1</a></b>;</span>
<br>
<span>THRUST_INLINE_CONSTANT thrust::detail::functional::placeholder< 1 >::type <b><a href="{{ site.baseurl }}/api/namespaces/namespacethrust_1_1placeholders.html#variable--2">&#95;2</a></b>;</span>
<br>
<span>THRUST_INLINE_CONSTANT thrust::detail::functional::placeholder< 2 >::type <b><a href="{{ site.baseurl }}/api/namespaces/namespacethrust_1_1placeholders.html#variable--3">&#95;3</a></b>;</span>
<br>
<span>THRUST_INLINE_CONSTANT thrust::detail::functional::placeholder< 3 >::type <b><a href="{{ site.baseurl }}/api/namespaces/namespacethrust_1_1placeholders.html#variable--4">&#95;4</a></b>;</span>
<br>
<span>THRUST_INLINE_CONSTANT thrust::detail::functional::placeholder< 4 >::type <b><a href="{{ site.baseurl }}/api/namespaces/namespacethrust_1_1placeholders.html#variable--5">&#95;5</a></b>;</span>
<br>
<span>THRUST_INLINE_CONSTANT thrust::detail::functional::placeholder< 5 >::type <b><a href="{{ site.baseurl }}/api/namespaces/namespacethrust_1_1placeholders.html#variable--6">&#95;6</a></b>;</span>
<br>
<span>THRUST_INLINE_CONSTANT thrust::detail::functional::placeholder< 6 >::type <b><a href="{{ site.baseurl }}/api/namespaces/namespacethrust_1_1placeholders.html#variable--7">&#95;7</a></b>;</span>
<br>
<span>THRUST_INLINE_CONSTANT thrust::detail::functional::placeholder< 7 >::type <b><a href="{{ site.baseurl }}/api/namespaces/namespacethrust_1_1placeholders.html#variable--8">&#95;8</a></b>;</span>
<br>
<span>THRUST_INLINE_CONSTANT thrust::detail::functional::placeholder< 8 >::type <b><a href="{{ site.baseurl }}/api/namespaces/namespacethrust_1_1placeholders.html#variable--9">&#95;9</a></b>;</span>
<br>
<span>THRUST_INLINE_CONSTANT thrust::detail::functional::placeholder< 9 >::type <b><a href="{{ site.baseurl }}/api/namespaces/namespacethrust_1_1placeholders.html#variable--10">&#95;10</a></b>;</span>
<span>} /* namespace thrust::placeholders */</span>
</code>

## Variables

<h3 id="variable--1">
Variable <code>thrust::placeholders::&#95;1</code>
</h3>

<code class="doxybook">
<span>THRUST_INLINE_CONSTANT thrust::detail::functional::placeholder< 0 >::type <b>_1</b>;</span></code>
<code><a href="{{ site.baseurl }}/api/namespaces/namespacethrust_1_1placeholders.html#variable--1">thrust::placeholders::&#95;1</a></code> is the placeholder for the first function parameter. 

<h3 id="variable--2">
Variable <code>thrust::placeholders::&#95;2</code>
</h3>

<code class="doxybook">
<span>THRUST_INLINE_CONSTANT thrust::detail::functional::placeholder< 1 >::type <b>_2</b>;</span></code>
<code><a href="{{ site.baseurl }}/api/namespaces/namespacethrust_1_1placeholders.html#variable--2">thrust::placeholders::&#95;2</a></code> is the placeholder for the second function parameter. 

<h3 id="variable--3">
Variable <code>thrust::placeholders::&#95;3</code>
</h3>

<code class="doxybook">
<span>THRUST_INLINE_CONSTANT thrust::detail::functional::placeholder< 2 >::type <b>_3</b>;</span></code>
<code><a href="{{ site.baseurl }}/api/namespaces/namespacethrust_1_1placeholders.html#variable--3">thrust::placeholders::&#95;3</a></code> is the placeholder for the third function parameter. 

<h3 id="variable--4">
Variable <code>thrust::placeholders::&#95;4</code>
</h3>

<code class="doxybook">
<span>THRUST_INLINE_CONSTANT thrust::detail::functional::placeholder< 3 >::type <b>_4</b>;</span></code>
<code><a href="{{ site.baseurl }}/api/namespaces/namespacethrust_1_1placeholders.html#variable--4">thrust::placeholders::&#95;4</a></code> is the placeholder for the fourth function parameter. 

<h3 id="variable--5">
Variable <code>thrust::placeholders::&#95;5</code>
</h3>

<code class="doxybook">
<span>THRUST_INLINE_CONSTANT thrust::detail::functional::placeholder< 4 >::type <b>_5</b>;</span></code>
<code><a href="{{ site.baseurl }}/api/namespaces/namespacethrust_1_1placeholders.html#variable--5">thrust::placeholders::&#95;5</a></code> is the placeholder for the fifth function parameter. 

<h3 id="variable--6">
Variable <code>thrust::placeholders::&#95;6</code>
</h3>

<code class="doxybook">
<span>THRUST_INLINE_CONSTANT thrust::detail::functional::placeholder< 5 >::type <b>_6</b>;</span></code>
<code><a href="{{ site.baseurl }}/api/namespaces/namespacethrust_1_1placeholders.html#variable--6">thrust::placeholders::&#95;6</a></code> is the placeholder for the sixth function parameter. 

<h3 id="variable--7">
Variable <code>thrust::placeholders::&#95;7</code>
</h3>

<code class="doxybook">
<span>THRUST_INLINE_CONSTANT thrust::detail::functional::placeholder< 6 >::type <b>_7</b>;</span></code>
<code><a href="{{ site.baseurl }}/api/namespaces/namespacethrust_1_1placeholders.html#variable--7">thrust::placeholders::&#95;7</a></code> is the placeholder for the seventh function parameter. 

<h3 id="variable--8">
Variable <code>thrust::placeholders::&#95;8</code>
</h3>

<code class="doxybook">
<span>THRUST_INLINE_CONSTANT thrust::detail::functional::placeholder< 7 >::type <b>_8</b>;</span></code>
<code><a href="{{ site.baseurl }}/api/namespaces/namespacethrust_1_1placeholders.html#variable--8">thrust::placeholders::&#95;8</a></code> is the placeholder for the eighth function parameter. 

<h3 id="variable--9">
Variable <code>thrust::placeholders::&#95;9</code>
</h3>

<code class="doxybook">
<span>THRUST_INLINE_CONSTANT thrust::detail::functional::placeholder< 8 >::type <b>_9</b>;</span></code>
<code><a href="{{ site.baseurl }}/api/namespaces/namespacethrust_1_1placeholders.html#variable--9">thrust::placeholders::&#95;9</a></code> is the placeholder for the ninth function parameter. 

<h3 id="variable--10">
Variable <code>thrust::placeholders::&#95;10</code>
</h3>

<code class="doxybook">
<span>THRUST_INLINE_CONSTANT thrust::detail::functional::placeholder< 9 >::type <b>_10</b>;</span></code>
<code><a href="{{ site.baseurl }}/api/namespaces/namespacethrust_1_1placeholders.html#variable--10">thrust::placeholders::&#95;10</a></code> is the placeholder for the tenth function parameter. 


