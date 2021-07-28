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

Objects in the <code><a href="/api/namespaces/namespacethrust_1_1placeholders.html">thrust::placeholders</a></code> namespace may be used to create simple arithmetic functions inline in an algorithm invocation. Combining placeholders such as <code>&#95;1</code> and <code>&#95;2</code> with arithmetic operations such as <code>+</code> creates an unnamed function object which applies the operation to their arguments.

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
<span>} /* namespace thrust::placeholders */</span>
</code>

