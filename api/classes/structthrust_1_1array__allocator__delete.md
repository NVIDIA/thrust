---
title: thrust::array_allocator_delete
nav_exclude: true
has_children: true
has_toc: false
---

# Struct `thrust::array_allocator_delete`

<code class="doxybook">
<span>template &lt;typename T,</span>
<span>&nbsp;&nbsp;typename Allocator,</span>
<span>&nbsp;&nbsp;bool Uninitialized = false&gt;</span>
<span>struct thrust::array&#95;allocator&#95;delete {</span>
<span>public:</span><span>&nbsp;&nbsp;using <b><a href="{{ site.baseurl }}/api/classes/structthrust_1_1array__allocator__delete.html#using-allocator-type">allocator&#95;type</a></b> = <i>see below</i>;</span>
<br>
<span>&nbsp;&nbsp;using <b><a href="{{ site.baseurl }}/api/classes/structthrust_1_1array__allocator__delete.html#using-pointer">pointer</a></b> = <i>see below</i>;</span>
<br>
<span>&nbsp;&nbsp;template &lt;typename UAllocator&gt;</span>
<span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/structthrust_1_1array__allocator__delete.html#function-array-allocator-delete">array&#95;allocator&#95;delete</a></b>(UAllocator && other,</span>
<span>&nbsp;&nbsp;&nbsp;&nbsp;std::size_t n);</span>
<br>
<span>&nbsp;&nbsp;template &lt;typename U,</span>
<span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;typename UAllocator&gt;</span>
<span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/structthrust_1_1array__allocator__delete.html#function-array-allocator-delete">array&#95;allocator&#95;delete</a></b>(array_allocator_delete< U, UAllocator > const & other);</span>
<br>
<span>&nbsp;&nbsp;template &lt;typename U,</span>
<span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;typename UAllocator&gt;</span>
<span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/structthrust_1_1array__allocator__delete.html#function-array-allocator-delete">array&#95;allocator&#95;delete</a></b>(array_allocator_delete< U, UAllocator > && other);</span>
<br>
<span>&nbsp;&nbsp;template &lt;typename U,</span>
<span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;typename UAllocator&gt;</span>
<span>&nbsp;&nbsp;array_allocator_delete & </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/structthrust_1_1array__allocator__delete.html#function-operator=">operator=</a></b>(array_allocator_delete< U, UAllocator > const & other);</span>
<br>
<span>&nbsp;&nbsp;template &lt;typename U,</span>
<span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;typename UAllocator&gt;</span>
<span>&nbsp;&nbsp;array_allocator_delete & </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/structthrust_1_1array__allocator__delete.html#function-operator=">operator=</a></b>(array_allocator_delete< U, UAllocator > && other);</span>
<br>
<span>&nbsp;&nbsp;void </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/structthrust_1_1array__allocator__delete.html#function-operator()">operator()</a></b>(pointer p);</span>
<br>
<span>&nbsp;&nbsp;allocator_type & </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/structthrust_1_1array__allocator__delete.html#function-get-allocator">get&#95;allocator</a></b>();</span>
<br>
<span>&nbsp;&nbsp;allocator_type const & </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/structthrust_1_1array__allocator__delete.html#function-get-allocator">get&#95;allocator</a></b>() const;</span>
<br>
<span>&nbsp;&nbsp;void </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/structthrust_1_1array__allocator__delete.html#function-swap">swap</a></b>(array_allocator_delete & other);</span>
<span>};</span>
</code>

## Member Types

<h3 id="using-allocator-type">
Type Alias <code>thrust::array&#95;allocator&#95;delete::allocator&#95;type</code>
</h3>

<code class="doxybook">
<span>using <b>allocator_type</b> = typename std::remove&#95;cv&lt; typename std::remove&#95;reference&lt; Allocator &gt;::type &gt;::type::template rebind&lt; T &gt;::other;</span></code>
<h3 id="using-pointer">
Type Alias <code>thrust::array&#95;allocator&#95;delete::pointer</code>
</h3>

<code class="doxybook">
<span>using <b>pointer</b> = typename detail::allocator&#95;traits&lt; allocator&#95;type &gt;::pointer;</span></code>

## Member Functions

<h3 id="function-array-allocator-delete">
Function <code>thrust::array&#95;allocator&#95;delete::array&#95;allocator&#95;delete</code>
</h3>

<code class="doxybook">
<span>template &lt;typename UAllocator&gt;</span>
<span><b>array_allocator_delete</b>(UAllocator && other,</span>
<span>&nbsp;&nbsp;std::size_t n);</span></code>
<h3 id="function-array-allocator-delete">
Function <code>thrust::array&#95;allocator&#95;delete::array&#95;allocator&#95;delete</code>
</h3>

<code class="doxybook">
<span>template &lt;typename U,</span>
<span>&nbsp;&nbsp;typename UAllocator&gt;</span>
<span><b>array_allocator_delete</b>(array_allocator_delete< U, UAllocator > const & other);</span></code>
<h3 id="function-array-allocator-delete">
Function <code>thrust::array&#95;allocator&#95;delete::array&#95;allocator&#95;delete</code>
</h3>

<code class="doxybook">
<span>template &lt;typename U,</span>
<span>&nbsp;&nbsp;typename UAllocator&gt;</span>
<span><b>array_allocator_delete</b>(array_allocator_delete< U, UAllocator > && other);</span></code>
<h3 id="function-operator=">
Function <code>thrust::array&#95;allocator&#95;delete::operator=</code>
</h3>

<code class="doxybook">
<span>template &lt;typename U,</span>
<span>&nbsp;&nbsp;typename UAllocator&gt;</span>
<span>array_allocator_delete & </span><span><b>operator=</b>(array_allocator_delete< U, UAllocator > const & other);</span></code>
<h3 id="function-operator=">
Function <code>thrust::array&#95;allocator&#95;delete::operator=</code>
</h3>

<code class="doxybook">
<span>template &lt;typename U,</span>
<span>&nbsp;&nbsp;typename UAllocator&gt;</span>
<span>array_allocator_delete & </span><span><b>operator=</b>(array_allocator_delete< U, UAllocator > && other);</span></code>
<h3 id="function-operator()">
Function <code>thrust::array&#95;allocator&#95;delete::operator()</code>
</h3>

<code class="doxybook">
<span>void </span><span><b>operator()</b>(pointer p);</span></code>
<h3 id="function-get-allocator">
Function <code>thrust::array&#95;allocator&#95;delete::get&#95;allocator</code>
</h3>

<code class="doxybook">
<span>allocator_type & </span><span><b>get_allocator</b>();</span></code>
<h3 id="function-get-allocator">
Function <code>thrust::array&#95;allocator&#95;delete::get&#95;allocator</code>
</h3>

<code class="doxybook">
<span>allocator_type const & </span><span><b>get_allocator</b>() const;</span></code>
<h3 id="function-swap">
Function <code>thrust::array&#95;allocator&#95;delete::swap</code>
</h3>

<code class="doxybook">
<span>void </span><span><b>swap</b>(array_allocator_delete & other);</span></code>

