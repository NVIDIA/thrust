---
title: thrust::mr::allocator
parent: Allocators
grand_parent: Memory Management
nav_exclude: true
has_children: true
has_toc: false
---

# Class `thrust::mr::allocator`

An <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1mr_1_1allocator.html">mr::allocator</a></code> is a template that fulfills the C++ requirements for Allocators, allowing to use the NPA-based memory resources where an Allocator is required. Unlike memory resources, but like other allocators, <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1mr_1_1allocator.html">mr::allocator</a></code> is typed and bound to allocate object of a specific type, however it can be freely rebound to other types.

**Template Parameters**:
* **`T`** the type that will be allocated by this allocator. 
* **`MR`** the upstream memory resource to use for memory allocation. Must derive from <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1mr_1_1memory__resource.html">thrust::mr::memory&#95;resource</a></code> and must be <code>final</code> (in C++11 and beyond). 

**Inherits From**:
`thrust::mr::validator< MR >`

<code class="doxybook">
<span>#include <thrust/mr/allocator.h></span><br>
<span>template &lt;typename T,</span>
<span>&nbsp;&nbsp;class MR&gt;</span>
<span>class thrust::mr::allocator {</span>
<span>public:</span><span>&nbsp;&nbsp;typedef <i>see below</i> <b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1mr_1_1allocator.html#typedef-void-pointer">void&#95;pointer</a></b>;</span>
<br>
<span>&nbsp;&nbsp;typedef <i>see below</i> <b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1mr_1_1allocator.html#typedef-value-type">value&#95;type</a></b>;</span>
<br>
<span>&nbsp;&nbsp;typedef <i>see below</i> <b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1mr_1_1allocator.html#typedef-pointer">pointer</a></b>;</span>
<br>
<span>&nbsp;&nbsp;typedef <i>see below</i> <b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1mr_1_1allocator.html#typedef-const-pointer">const&#95;pointer</a></b>;</span>
<br>
<span>&nbsp;&nbsp;typedef <i>see below</i> <b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1mr_1_1allocator.html#typedef-reference">reference</a></b>;</span>
<br>
<span>&nbsp;&nbsp;typedef <i>see below</i> <b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1mr_1_1allocator.html#typedef-const-reference">const&#95;reference</a></b>;</span>
<br>
<span>&nbsp;&nbsp;typedef <i>see below</i> <b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1mr_1_1allocator.html#typedef-size-type">size&#95;type</a></b>;</span>
<br>
<span>&nbsp;&nbsp;typedef <i>see below</i> <b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1mr_1_1allocator.html#typedef-difference-type">difference&#95;type</a></b>;</span>
<br>
<span>&nbsp;&nbsp;typedef <i>see below</i> <b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1mr_1_1allocator.html#typedef-propagate-on-container-copy-assignment">propagate&#95;on&#95;container&#95;copy&#95;assignment</a></b>;</span>
<br>
<span>&nbsp;&nbsp;typedef <i>see below</i> <b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1mr_1_1allocator.html#typedef-propagate-on-container-move-assignment">propagate&#95;on&#95;container&#95;move&#95;assignment</a></b>;</span>
<br>
<span>&nbsp;&nbsp;typedef <i>see below</i> <b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1mr_1_1allocator.html#typedef-propagate-on-container-swap">propagate&#95;on&#95;container&#95;swap</a></b>;</span>
<br>
<span>&nbsp;&nbsp;template &lt;typename U&gt;</span>
<span>&nbsp;&nbsp;struct <b><a href="{{ site.baseurl }}/api/classes/structthrust_1_1mr_1_1allocator_1_1rebind.html">rebind</a></b>;</span>
<br>
<span>&nbsp;&nbsp;__thrust_exec_check_disable__ __host__ __device__ <a href="{{ site.baseurl }}/api/classes/classthrust_1_1mr_1_1allocator.html#typedef-size-type">size_type</a> </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1mr_1_1allocator.html#function-max-size">max&#95;size</a></b>() const;</span>
<br>
<span>&nbsp;&nbsp;__host__ __device__ </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1mr_1_1allocator.html#function-allocator">allocator</a></b>(MR * resource);</span>
<br>
<span>&nbsp;&nbsp;template &lt;typename U&gt;</span>
<span>&nbsp;&nbsp;__host__ __device__ </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1mr_1_1allocator.html#function-allocator">allocator</a></b>(const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1mr_1_1allocator.html">allocator</a>< U, MR > & other);</span>
<br>
<span>&nbsp;&nbsp;__host__ <a href="{{ site.baseurl }}/api/classes/classthrust_1_1mr_1_1allocator.html#typedef-pointer">pointer</a> </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1mr_1_1allocator.html#function-allocate">allocate</a></b>(<a href="{{ site.baseurl }}/api/classes/classthrust_1_1mr_1_1allocator.html#typedef-size-type">size_type</a> n);</span>
<br>
<span>&nbsp;&nbsp;__host__ void </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1mr_1_1allocator.html#function-deallocate">deallocate</a></b>(<a href="{{ site.baseurl }}/api/classes/classthrust_1_1mr_1_1allocator.html#typedef-pointer">pointer</a> p,</span>
<span>&nbsp;&nbsp;&nbsp;&nbsp;<a href="{{ site.baseurl }}/api/classes/classthrust_1_1mr_1_1allocator.html#typedef-size-type">size_type</a> n);</span>
<br>
<span>&nbsp;&nbsp;__host__ __device__ MR * </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1mr_1_1allocator.html#function-resource">resource</a></b>() const;</span>
<span>};</span>
</code>

## Member Classes

<h3 id="struct-thrustmrallocatorrebind">
<a href="{{ site.baseurl }}/api/classes/structthrust_1_1mr_1_1allocator_1_1rebind.html">Struct <code>thrust::mr::allocator::rebind</code>
</a>
</h3>


## Member Types

<h3 id="typedef-void-pointer">
Typedef <code>thrust::mr::allocator::void&#95;pointer</code>
</h3>

<code class="doxybook">
<span>typedef MR::pointer<b>void_pointer</b>;</span></code>
The pointer to void type of this allocator. 

<h3 id="typedef-value-type">
Typedef <code>thrust::mr::allocator::value&#95;type</code>
</h3>

<code class="doxybook">
<span>typedef T<b>value_type</b>;</span></code>
The value type allocated by this allocator. Equivalent to <code>T</code>. 

<h3 id="typedef-pointer">
Typedef <code>thrust::mr::allocator::pointer</code>
</h3>

<code class="doxybook">
<span>typedef thrust::detail::pointer_traits< <a href="{{ site.baseurl }}/api/classes/classthrust_1_1mr_1_1allocator.html#typedef-void-pointer">void_pointer</a> >::template <a href="{{ site.baseurl }}/api/classes/structthrust_1_1mr_1_1allocator_1_1rebind.html">rebind</a>< T >::other<b>pointer</b>;</span></code>
The pointer type allocated by this allocator. Equivaled to the pointer type of <code>MR</code> rebound to <code>T</code>. 

<h3 id="typedef-const-pointer">
Typedef <code>thrust::mr::allocator::const&#95;pointer</code>
</h3>

<code class="doxybook">
<span>typedef thrust::detail::pointer_traits< <a href="{{ site.baseurl }}/api/classes/classthrust_1_1mr_1_1allocator.html#typedef-void-pointer">void_pointer</a> >::template <a href="{{ site.baseurl }}/api/classes/structthrust_1_1mr_1_1allocator_1_1rebind.html">rebind</a>< constT >::other<b>const_pointer</b>;</span></code>
The pointer to const type. Equivalent to a pointer type of <code>MR</code> rebound to <code>const T</code>. 

<h3 id="typedef-reference">
Typedef <code>thrust::mr::allocator::reference</code>
</h3>

<code class="doxybook">
<span>typedef thrust::detail::pointer_traits< <a href="{{ site.baseurl }}/api/classes/classthrust_1_1mr_1_1allocator.html#typedef-pointer">pointer</a> >::reference<b>reference</b>;</span></code>
The reference to the type allocated by this allocator. Supports smart references. 

<h3 id="typedef-const-reference">
Typedef <code>thrust::mr::allocator::const&#95;reference</code>
</h3>

<code class="doxybook">
<span>typedef thrust::detail::pointer_traits< <a href="{{ site.baseurl }}/api/classes/classthrust_1_1mr_1_1allocator.html#typedef-const-pointer">const_pointer</a> ><a href="{{ site.baseurl }}/api/classes/classthrust_1_1mr_1_1allocator.html#typedef-reference">::reference</a><b>const_reference</b>;</span></code>
The const reference to the type allocated by this allocator. Supports smart references. 

<h3 id="typedef-size-type">
Typedef <code>thrust::mr::allocator::size&#95;type</code>
</h3>

<code class="doxybook">
<span>typedef std::size_t<b>size_type</b>;</span></code>
The size type of this allocator. Always <code>std::size&#95;t</code>. 

<h3 id="typedef-difference-type">
Typedef <code>thrust::mr::allocator::difference&#95;type</code>
</h3>

<code class="doxybook">
<span>typedef thrust::detail::pointer_traits< <a href="{{ site.baseurl }}/api/classes/classthrust_1_1mr_1_1allocator.html#typedef-pointer">pointer</a> >::difference_type<b>difference_type</b>;</span></code>
The difference type between pointers allocated by this allocator. 

<h3 id="typedef-propagate-on-container-copy-assignment">
Typedef <code>thrust::mr::allocator::propagate&#95;on&#95;container&#95;copy&#95;assignment</code>
</h3>

<code class="doxybook">
<span>typedef detail::true_type<b>propagate_on_container_copy_assignment</b>;</span></code>
Specifies that the allocator shall be propagated on container copy assignment. 

<h3 id="typedef-propagate-on-container-move-assignment">
Typedef <code>thrust::mr::allocator::propagate&#95;on&#95;container&#95;move&#95;assignment</code>
</h3>

<code class="doxybook">
<span>typedef detail::true_type<b>propagate_on_container_move_assignment</b>;</span></code>
Specifies that the allocator shall be propagated on container move assignment. 

<h3 id="typedef-propagate-on-container-swap">
Typedef <code>thrust::mr::allocator::propagate&#95;on&#95;container&#95;swap</code>
</h3>

<code class="doxybook">
<span>typedef detail::true_type<b>propagate_on_container_swap</b>;</span></code>
Specifies that the allocator shall be propagated on container swap. 


## Member Functions

<h3 id="function-max-size">
Function <code>thrust::mr::allocator::max&#95;size</code>
</h3>

<code class="doxybook">
<span>__thrust_exec_check_disable__ __host__ __device__ <a href="{{ site.baseurl }}/api/classes/classthrust_1_1mr_1_1allocator.html#typedef-size-type">size_type</a> </span><span><b>max_size</b>() const;</span></code>
Calculates the maximum number of elements allocated by this allocator.

**Returns**:
the maximum value of <code>std::size&#95;t</code>, divided by the size of <code>T</code>. 

<h3 id="function-allocator">
Function <code>thrust::mr::allocator::allocator</code>
</h3>

<code class="doxybook">
<span>__host__ __device__ </span><span><b>allocator</b>(MR * resource);</span></code>
Constructor.

**Function Parameters**:
**`resource`**: the resource to be used to allocate raw memory. 

<h3 id="function-allocator">
Function <code>thrust::mr::allocator::allocator</code>
</h3>

<code class="doxybook">
<span>template &lt;typename U&gt;</span>
<span>__host__ __device__ </span><span><b>allocator</b>(const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1mr_1_1allocator.html">allocator</a>< U, MR > & other);</span></code>
Copy constructor. Copies the resource pointer. 

<h3 id="function-allocate">
Function <code>thrust::mr::allocator::allocate</code>
</h3>

<code class="doxybook">
<span>__host__ <a href="{{ site.baseurl }}/api/classes/classthrust_1_1mr_1_1allocator.html#typedef-pointer">pointer</a> </span><span><b>allocate</b>(<a href="{{ site.baseurl }}/api/classes/classthrust_1_1mr_1_1allocator.html#typedef-size-type">size_type</a> n);</span></code>
Allocates objects of type <code>T</code>.

**Function Parameters**:
**`n`**: number of elements to allocate 

**Returns**:
a pointer to the newly allocated storage. 

<h3 id="function-deallocate">
Function <code>thrust::mr::allocator::deallocate</code>
</h3>

<code class="doxybook">
<span>__host__ void </span><span><b>deallocate</b>(<a href="{{ site.baseurl }}/api/classes/classthrust_1_1mr_1_1allocator.html#typedef-pointer">pointer</a> p,</span>
<span>&nbsp;&nbsp;<a href="{{ site.baseurl }}/api/classes/classthrust_1_1mr_1_1allocator.html#typedef-size-type">size_type</a> n);</span></code>
Deallocates objects of type <code>T</code>.

**Function Parameters**:
* **`p`** pointer returned by a previous call to <code>allocate</code>
* **`n`** number of elements, passed as an argument to the <code>allocate</code> call that produced <code>p</code>

<h3 id="function-resource">
Function <code>thrust::mr::allocator::resource</code>
</h3>

<code class="doxybook">
<span>__host__ __device__ MR * </span><span><b>resource</b>() const;</span></code>
Extracts the memory resource used by this allocator.

**Returns**:
the memory resource used by this allocator. 


