---
title: thrust::transform_output_iterator
parent: Fancy Iterators
grand_parent: Iterators
nav_exclude: true
has_children: true
has_toc: false
---

# Class `thrust::transform_output_iterator`

<code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1transform__output__iterator.html">transform&#95;output&#95;iterator</a></code> is a special kind of output iterator which transforms a value written upon dereference. This iterator is useful for transforming an output from algorithms without explicitly storing the intermediate result in the memory and applying subsequent transformation, thereby avoiding wasting memory capacity and bandwidth. Using <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1transform__iterator.html">transform&#95;iterator</a></code> facilitates kernel fusion by deferring execution of transformation until the value is written while saving both memory capacity and bandwidth.

The following code snippet demonstrated how to create a <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1transform__output__iterator.html">transform&#95;output&#95;iterator</a></code> which applies <code>sqrtf</code> to the assigning value.



```cpp
#include <thrust/iterator/transform_output_iterator.h>
#include <thrust/device_vector.h>

// note: functor inherits form unary function
 // note: functor inherits from unary_function
 struct square_root : public thrust::unary_function<float,float>
 {
   __host__ __device__
   float operator()(float x) const
   {
     return sqrtf(x);
   }
 };
 
 int main()
 {
   thrust::device_vector<float> v(4);

   typedef thrust::device_vector<float>::iterator FloatIterator;
   thrust::transform_output_iterator<square_root, FloatIterator> iter(v.begin(), square_root());

   iter[0] =  1.0f;    // stores sqrtf( 1.0f) 
   iter[1] =  4.0f;    // stores sqrtf( 4.0f)
   iter[2] =  9.0f;    // stores sqrtf( 9.0f)
   iter[3] = 16.0f;    // stores sqrtf(16.0f)
   // iter[4] is an out-of-bounds error
                                                                                          
   v[0]; // returns 1.0f;
   v[1]; // returns 2.0f;
   v[2]; // returns 3.0f;
   v[3]; // returns 4.0f;
                                                                                          
 }
```

**Inherits From**:
`detail::transform_output_iterator_base::type`

**See**:
make_transform_output_iterator 

<code class="doxybook">
<span>#include <thrust/iterator/transform_output_iterator.h></span><br>
<span>template &lt;typename UnaryFunction,</span>
<span>&nbsp;&nbsp;typename OutputIterator&gt;</span>
<span>class thrust::transform&#95;output&#95;iterator {</span>
<span>};</span>
</code>

