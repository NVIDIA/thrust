---
title: thrust::transform_iterator
parent: Fancy Iterators
grand_parent: Iterators
nav_exclude: true
has_children: true
has_toc: false
---

# Class `thrust::transform_iterator`

<code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1transform__iterator.html">transform&#95;iterator</a></code> is an iterator which represents a pointer into a range of values after transformation by a function. This iterator is useful for creating a range filled with the result of applying an operation to another range without either explicitly storing it in memory, or explicitly executing the transformation. Using <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1transform__iterator.html">transform&#95;iterator</a></code> facilitates kernel fusion by deferring the execution of a transformation until the value is needed while saving both memory capacity and bandwidth.

The following code snippet demonstrates how to create a <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1transform__iterator.html">transform&#95;iterator</a></code> which represents the result of <code>sqrtf</code> applied to the contents of a <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1device__vector.html">device&#95;vector</a></code>.



```cpp
#include <thrust/iterator/transform_iterator.h>
#include <thrust/device_vector.h>

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
  v[0] = 1.0f;
  v[1] = 4.0f;
  v[2] = 9.0f;
  v[3] = 16.0f;

  typedef thrust::device_vector<float>::iterator FloatIterator;

  thrust::transform_iterator<square_root, FloatIterator> iter(v.begin(), square_root());

  *iter;   // returns 1.0f
  iter[0]; // returns 1.0f;
  iter[1]; // returns 2.0f;
  iter[2]; // returns 3.0f;
  iter[3]; // returns 4.0f;

  // iter[4] is an out-of-bounds error
}
```

This next example demonstrates how to use a <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1transform__iterator.html">transform&#95;iterator</a></code> with the <code>thrust::reduce</code> function to compute the sum of squares of a sequence. We will create temporary <code>transform&#95;iterators</code> with the <code>make&#95;transform&#95;iterator</code> function in order to avoid explicitly specifying their type:



```cpp
#include <thrust/iterator/transform_iterator.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <iostream>

// note: functor inherits from unary_function
struct square : public thrust::unary_function<float,float>
{
  __host__ __device__
  float operator()(float x) const
  {
    return x * x;
  }
};

int main()
{
  // initialize a device array
  thrust::device_vector<float> v(4);
  v[0] = 1.0f;
  v[1] = 2.0f;
  v[2] = 3.0f;
  v[3] = 4.0f;

  float sum_of_squares =
   thrust::reduce(thrust::make_transform_iterator(v.begin(), square()),
                  thrust::make_transform_iterator(v.end(),   square()));

  std::cout << "sum of squares: " << sum_of_squares << std::endl;
  return 0;
}
```

Note that in the previous two examples the transform functor (namely <code>square&#95;root</code> and <code>square</code>) inherits from <code><a href="{{ site.baseurl }}/api/classes/structthrust_1_1unary__function.html">thrust::unary&#95;function</a></code>. Inheriting from <code><a href="{{ site.baseurl }}/api/classes/structthrust_1_1unary__function.html">thrust::unary&#95;function</a></code> ensures that a functor is a valid <code>AdaptableUnaryFunction</code> and provides all the necessary <code>typedef</code> declarations. The <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1transform__iterator.html">transform&#95;iterator</a></code> can also be applied to a <code>UnaryFunction</code> that does not inherit from <code><a href="{{ site.baseurl }}/api/classes/structthrust_1_1unary__function.html">thrust::unary&#95;function</a></code> using an optional template argument. The following example illustrates how to use the third template argument to specify the <code>result&#95;type</code> of the function.



```cpp
#include <thrust/iterator/transform_iterator.h>
#include <thrust/device_vector.h>

// note: functor *does not* inherit from unary_function
struct square_root
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
  v[0] = 1.0f;
  v[1] = 4.0f;
  v[2] = 9.0f;
  v[3] = 16.0f;

  typedef thrust::device_vector<float>::iterator FloatIterator;

  // note: float result_type is specified explicitly
  thrust::transform_iterator<square_root, FloatIterator, float> iter(v.begin(), square_root());

  *iter;   // returns 1.0f
  iter[0]; // returns 1.0f;
  iter[1]; // returns 2.0f;
  iter[2]; // returns 3.0f;
  iter[3]; // returns 4.0f;

  // iter[4] is an out-of-bounds error
}
```

**Inherits From**:
`detail::transform_iterator_base::type`

**See**:
make_transform_iterator 

<code class="doxybook">
<span>#include <thrust/iterator/transform_iterator.h></span><br>
<span>template &lt;class AdaptableUnaryFunction,</span>
<span>&nbsp;&nbsp;class Iterator,</span>
<span>&nbsp;&nbsp;class Reference = use&#95;default,</span>
<span>&nbsp;&nbsp;class Value = use&#95;default&gt;</span>
<span>class thrust::transform&#95;iterator {</span>
<span>public:</span><span>&nbsp;&nbsp;__host__ __device__ </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1transform__iterator.html#function-transform-iterator">transform&#95;iterator</a></b>();</span>
<br>
<span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1transform__iterator.html#function-transform-iterator">transform&#95;iterator</a></b>(<a href="{{ site.baseurl }}/api/classes/classthrust_1_1transform__iterator.html">transform_iterator</a> const &) = default;</span>
<br>
<span>&nbsp;&nbsp;__host__ __device__ </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1transform__iterator.html#function-transform-iterator">transform&#95;iterator</a></b>(Iterator const & x,</span>
<span>&nbsp;&nbsp;&nbsp;&nbsp;AdaptableUnaryFunction f);</span>
<br>
<span>&nbsp;&nbsp;explicit __host__ __device__ </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1transform__iterator.html#function-transform-iterator">transform&#95;iterator</a></b>(Iterator const & x);</span>
<br>
<span>&nbsp;&nbsp;template &lt;typename OtherAdaptableUnaryFunction,</span>
<span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;typename OtherIterator,</span>
<span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;typename OtherReference,</span>
<span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;typename OtherValue&gt;</span>
<span>&nbsp;&nbsp;__host__ __device__ </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1transform__iterator.html#function-transform-iterator">transform&#95;iterator</a></b>(const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1transform__iterator.html">transform_iterator</a>< OtherAdaptableUnaryFunction, OtherIterator, OtherReference, OtherValue > & other,</span>
<span>&nbsp;&nbsp;&nbsp;&nbsp;typename thrust::detail::enable_if_convertible< OtherIterator, Iterator >::type * = 0,</span>
<span>&nbsp;&nbsp;&nbsp;&nbsp;typename thrust::detail::enable_if_convertible< OtherAdaptableUnaryFunction, AdaptableUnaryFunction >::type * = 0);</span>
<br>
<span>&nbsp;&nbsp;__host__ __device__ <a href="{{ site.baseurl }}/api/classes/classthrust_1_1transform__iterator.html">transform_iterator</a> & </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1transform__iterator.html#function-operator=">operator=</a></b>(const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1transform__iterator.html">transform_iterator</a> & other);</span>
<br>
<span>&nbsp;&nbsp;__host__ __device__ AdaptableUnaryFunction </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1transform__iterator.html#function-functor">functor</a></b>() const;</span>
<span>};</span>
</code>

## Member Functions

<h3 id="function-transform-iterator">
Function <code>thrust::transform&#95;iterator::transform&#95;iterator</code>
</h3>

<code class="doxybook">
<span>__host__ __device__ </span><span><b>transform_iterator</b>();</span></code>
Null constructor does nothing. 

<h3 id="function-transform-iterator">
Function <code>thrust::transform&#95;iterator::transform&#95;iterator</code>
</h3>

<code class="doxybook">
<span><b>transform_iterator</b>(<a href="{{ site.baseurl }}/api/classes/classthrust_1_1transform__iterator.html">transform_iterator</a> const &) = default;</span></code>
<h3 id="function-transform-iterator">
Function <code>thrust::transform&#95;iterator::transform&#95;iterator</code>
</h3>

<code class="doxybook">
<span>__host__ __device__ </span><span><b>transform_iterator</b>(Iterator const & x,</span>
<span>&nbsp;&nbsp;AdaptableUnaryFunction f);</span></code>
This constructor takes as arguments an <code>Iterator</code> and an <code>AdaptableUnaryFunction</code> and copies them to a new <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1transform__iterator.html">transform&#95;iterator</a></code>.

**Function Parameters**:
* **`x`** An <code>Iterator</code> pointing to the input to this <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1transform__iterator.html">transform&#95;iterator</a>'s</code><code>AdaptableUnaryFunction</code>. 
* **`f`** An <code>AdaptableUnaryFunction</code> used to transform the objects pointed to by <code>x</code>. 

<h3 id="function-transform-iterator">
Function <code>thrust::transform&#95;iterator::transform&#95;iterator</code>
</h3>

<code class="doxybook">
<span>explicit __host__ __device__ </span><span><b>transform_iterator</b>(Iterator const & x);</span></code>
This explicit constructor copies the value of a given <code>Iterator</code> and creates this <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1transform__iterator.html">transform&#95;iterator</a>'s</code><code>AdaptableUnaryFunction</code> using its null constructor.

**Function Parameters**:
**`x`**: An <code>Iterator</code> to copy. 

<h3 id="function-transform-iterator">
Function <code>thrust::transform&#95;iterator::transform&#95;iterator</code>
</h3>

<code class="doxybook">
<span>template &lt;typename OtherAdaptableUnaryFunction,</span>
<span>&nbsp;&nbsp;typename OtherIterator,</span>
<span>&nbsp;&nbsp;typename OtherReference,</span>
<span>&nbsp;&nbsp;typename OtherValue&gt;</span>
<span>__host__ __device__ </span><span><b>transform_iterator</b>(const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1transform__iterator.html">transform_iterator</a>< OtherAdaptableUnaryFunction, OtherIterator, OtherReference, OtherValue > & other,</span>
<span>&nbsp;&nbsp;typename thrust::detail::enable_if_convertible< OtherIterator, Iterator >::type * = 0,</span>
<span>&nbsp;&nbsp;typename thrust::detail::enable_if_convertible< OtherAdaptableUnaryFunction, AdaptableUnaryFunction >::type * = 0);</span></code>
This copy constructor creates a new <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1transform__iterator.html">transform&#95;iterator</a></code> from another <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1transform__iterator.html">transform&#95;iterator</a></code>.

**Function Parameters**:
**`other`**: The <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1transform__iterator.html">transform&#95;iterator</a></code> to copy. 

<h3 id="function-operator=">
Function <code>thrust::transform&#95;iterator::operator=</code>
</h3>

<code class="doxybook">
<span>__host__ __device__ <a href="{{ site.baseurl }}/api/classes/classthrust_1_1transform__iterator.html">transform_iterator</a> & </span><span><b>operator=</b>(const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1transform__iterator.html">transform_iterator</a> & other);</span></code>
Copy assignment operator copies from another <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1transform__iterator.html">transform&#95;iterator</a></code>. <code>other</code> The other <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1transform__iterator.html">transform&#95;iterator</a></code> to copy 
In any case, this <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1transform__iterator.html">transform&#95;iterator</a>'s</code> underlying iterator will be copy assigned. 

**Note**:
If the type of this <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1transform__iterator.html">transform&#95;iterator</a>'s</code> functor is not copy assignable (for example, if it is a lambda) it is not an error to call this function. In this case, however, the functor will not be modified.

**Returns**:
<code>&#42;this</code>

<h3 id="function-functor">
Function <code>thrust::transform&#95;iterator::functor</code>
</h3>

<code class="doxybook">
<span>__host__ __device__ AdaptableUnaryFunction </span><span><b>functor</b>() const;</span></code>
This method returns a copy of this <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1transform__iterator.html">transform&#95;iterator</a>'s</code><code>AdaptableUnaryFunction</code>. 

**Returns**:
A copy of this <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1transform__iterator.html">transform&#95;iterator</a>'s</code><code>AdaptableUnaryFunction</code>. 


