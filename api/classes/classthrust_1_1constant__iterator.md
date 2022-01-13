---
title: thrust::constant_iterator
parent: Fancy Iterators
grand_parent: Iterators
nav_exclude: true
has_children: true
has_toc: false
---

# Class `thrust::constant_iterator`

<code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1constant__iterator.html">constant&#95;iterator</a></code> is an iterator which represents a pointer into a range of constant values. This iterator is useful for creating a range filled with the same value without explicitly storing it in memory. Using <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1constant__iterator.html">constant&#95;iterator</a></code> saves both memory capacity and bandwidth.

The following code snippet demonstrates how to create a <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1constant__iterator.html">constant&#95;iterator</a></code> whose <code>value&#95;type</code> is <code>int</code> and whose value is <code>10</code>.



```cpp
#include <thrust/iterator/constant_iterator.h>

thrust::constant_iterator<int> iter(10);

*iter;    // returns 10
iter[0];  // returns 10
iter[1];  // returns 10
iter[13]; // returns 10

// and so on...
```

This next example demonstrates how to use a <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1constant__iterator.html">constant&#95;iterator</a></code> with the <code>thrust::transform</code> function to increment all elements of a sequence by the same value. We will create a temporary <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1constant__iterator.html">constant&#95;iterator</a></code> with the function <code>make&#95;constant&#95;iterator</code> function in order to avoid explicitly specifying its type:



```cpp
#include <thrust/iterator/constant_iterator.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/device_vector.h>

int main()
{
  thrust::device_vector<int> data(4);
  data[0] = 3;
  data[1] = 7;
  data[2] = 2;
  data[3] = 5;
  
  // add 10 to all values in data
  thrust::transform(data.begin(), data.end(),
                    thrust::make_constant_iterator(10),
                    data.begin(),
                    thrust::plus<int>());
  
  // data is now [13, 17, 12, 15]
  
  return 0;
}
```

**Inherits From**:
`detail::constant_iterator_base::type`

**See**:
make_constant_iterator 

<code class="doxybook">
<span>#include <thrust/iterator/constant_iterator.h></span><br>
<span>template &lt;typename Value,</span>
<span>&nbsp;&nbsp;typename Incrementable = use&#95;default,</span>
<span>&nbsp;&nbsp;typename System = use&#95;default&gt;</span>
<span>class thrust::constant&#95;iterator {</span>
<span>};</span>
</code>

