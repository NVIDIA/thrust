---
title: thrust::discard_iterator
parent: Fancy Iterators
grand_parent: Iterators
nav_exclude: true
has_children: true
has_toc: false
---

# Class `thrust::discard_iterator`

<code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1discard__iterator.html">discard&#95;iterator</a></code> is an iterator which represents a special kind of pointer that ignores values written to it upon dereference. This iterator is useful for ignoring the output of certain algorithms without wasting memory capacity or bandwidth. <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1discard__iterator.html">discard&#95;iterator</a></code> may also be used to count the size of an algorithm's output which may not be known a priori.

The following code snippet demonstrates how to use <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1discard__iterator.html">discard&#95;iterator</a></code> to ignore ignore one of the output ranges of reduce_by_key



```cpp
#include <thrust/iterator/discard_iterator.h>
#include <thrust/reduce.h>
#include <thrust/device_vector.h>

int main()
{
  thrust::device_vector<int> keys(7), values(7);

  keys[0] = 1;
  keys[1] = 3;
  keys[2] = 3;
  keys[3] = 3;
  keys[4] = 2;
  keys[5] = 2;
  keys[6] = 1;

  values[0] = 9;
  values[1] = 8;
  values[2] = 7;
  values[3] = 6;
  values[4] = 5;
  values[5] = 4;
  values[6] = 3;

  thrust::device_vector<int> result(4);

  // we are only interested in the reduced values
  // use discard_iterator to ignore the output keys
  thrust::reduce_by_key(keys.begin(), keys.end(),
                        values.begin(),
                        thrust::make_discard_iterator(),
                        result.begin());

  // result is now [9, 21, 9, 3]

  return 0;
}
```

**Inherits From**:
`detail::discard_iterator_base::type`

**See**:
make_discard_iterator 

<code class="doxybook">
<span>#include <thrust/iterator/discard_iterator.h></span><br>
<span>template &lt;typename System = use&#95;default&gt;</span>
<span>class thrust::discard&#95;iterator {</span>
<span>};</span>
</code>

