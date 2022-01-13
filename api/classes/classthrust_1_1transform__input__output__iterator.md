---
title: thrust::transform_input_output_iterator
parent: Fancy Iterators
grand_parent: Iterators
nav_exclude: true
has_children: true
has_toc: false
---

# Class `thrust::transform_input_output_iterator`

<code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1transform__input__output__iterator.html">transform&#95;input&#95;output&#95;iterator</a></code> is a special kind of iterator which applies transform functions when reading from or writing to dereferenced values. This iterator is useful for algorithms that operate on a type that needs to be serialized/deserialized from values in another iterator, avoiding the need to materialize intermediate results in memory. This also enables the transform functions to be fused with the operations that read and write to the <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1transform__input__output__iterator.html">transform&#95;input&#95;output&#95;iterator</a></code>.

The following code snippet demonstrates how to create a <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1transform__input__output__iterator.html">transform&#95;input&#95;output&#95;iterator</a></code> which performs different transformations when reading from and writing to the iterator.



```cpp
#include <thrust/iterator/transform_input_output_iterator.h>
#include <thrust/device_vector.h>

 int main()
 {
   const size_t size = 4;
   thrust::device_vector<float> v(size);

   // Write 1.0f, 2.0f, 3.0f, 4.0f to vector
   thrust::sequence(v.begin(), v.end(), 1);

   // Iterator that returns negated values and writes squared values
   auto iter = thrust::make_transform_input_output_iterator(v.begin(),
       thrust::negate<float>{}, thrust::square<float>{});

   // Iterator negates values when reading
   std::cout << iter[0] << " ";  // -1.0f;
   std::cout << iter[1] << " ";  // -2.0f;
   std::cout << iter[2] << " ";  // -3.0f;
   std::cout << iter[3] << "\n"; // -4.0f;

   // Write 1.0f, 2.0f, 3.0f, 4.0f to iterator
   thrust::sequence(iter, iter + size, 1);

   // Values were squared before writing to vector
   std::cout << v[0] << " ";  // 1.0f;
   std::cout << v[1] << " ";  // 4.0f;
   std::cout << v[2] << " ";  // 9.0f;
   std::cout << v[3] << "\n"; // 16.0f;

 }
```

**Inherits From**:
`detail::transform_input_output_iterator_base::type`

**See**:
make_transform_input_output_iterator 

<code class="doxybook">
<span>#include <thrust/iterator/transform_input_output_iterator.h></span><br>
<span>template &lt;typename InputFunction,</span>
<span>&nbsp;&nbsp;typename OutputFunction,</span>
<span>&nbsp;&nbsp;typename Iterator&gt;</span>
<span>class thrust::transform&#95;input&#95;output&#95;iterator {</span>
<span>};</span>
</code>

