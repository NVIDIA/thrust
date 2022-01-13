---
title: thrust::mr::disjoint_unsynchronized_pool_resource::pool
nav_exclude: true
has_children: true
has_toc: false
---

# Struct `thrust::mr::disjoint_unsynchronized_pool_resource::pool`

<code class="doxybook">
<span>struct thrust::mr::disjoint&#95;unsynchronized&#95;pool&#95;resource::pool {</span>
<span>public:</span><span>&nbsp;&nbsp;<a href="{{ site.baseurl }}/api/classes/classthrust_1_1host__vector.html">pointer_vector</a> <b><a href="{{ site.baseurl }}/api/classes/structthrust_1_1mr_1_1disjoint__unsynchronized__pool__resource_1_1pool.html#variable-free-blocks">free&#95;blocks</a></b>;</span>
<br>
<span>&nbsp;&nbsp;std::size_t <b><a href="{{ site.baseurl }}/api/classes/structthrust_1_1mr_1_1disjoint__unsynchronized__pool__resource_1_1pool.html#variable-previous-allocated-count">previous&#95;allocated&#95;count</a></b>;</span>
<br>
<span>&nbsp;&nbsp;__host__ </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/structthrust_1_1mr_1_1disjoint__unsynchronized__pool__resource_1_1pool.html#function-pool">pool</a></b>(const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1host__vector.html">pointer_vector</a> & free);</span>
<br>
<span>&nbsp;&nbsp;__host__ </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/structthrust_1_1mr_1_1disjoint__unsynchronized__pool__resource_1_1pool.html#function-pool">pool</a></b>(const pool & other);</span>
<br>
<span>&nbsp;&nbsp;pool & </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/structthrust_1_1mr_1_1disjoint__unsynchronized__pool__resource_1_1pool.html#function-operator=">operator=</a></b>(const pool &) = default;</span>
<br>
<span>&nbsp;&nbsp;__host__ </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/structthrust_1_1mr_1_1disjoint__unsynchronized__pool__resource_1_1pool.html#function-~pool">~pool</a></b>();</span>
<span>};</span>
</code>

## Member Variables

<h3 id="variable-free-blocks">
Variable <code>thrust::mr::disjoint&#95;unsynchronized&#95;pool&#95;resource::pool::free&#95;blocks</code>
</h3>

<code class="doxybook">
<span><a href="{{ site.baseurl }}/api/classes/classthrust_1_1host__vector.html">pointer_vector</a> <b>free_blocks</b>;</span></code>
<h3 id="variable-previous-allocated-count">
Variable <code>thrust::mr::disjoint&#95;unsynchronized&#95;pool&#95;resource::pool::previous&#95;allocated&#95;count</code>
</h3>

<code class="doxybook">
<span>std::size_t <b>previous_allocated_count</b>;</span></code>

## Member Functions

<h3 id="function-pool">
Function <code>thrust::mr::disjoint&#95;unsynchronized&#95;pool&#95;resource::pool::pool</code>
</h3>

<code class="doxybook">
<span>__host__ </span><span><b>pool</b>(const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1host__vector.html">pointer_vector</a> & free);</span></code>
<h3 id="function-pool">
Function <code>thrust::mr::disjoint&#95;unsynchronized&#95;pool&#95;resource::pool::pool</code>
</h3>

<code class="doxybook">
<span>__host__ </span><span><b>pool</b>(const pool & other);</span></code>
<h3 id="function-operator=">
Function <code>thrust::mr::disjoint&#95;unsynchronized&#95;pool&#95;resource::pool::operator=</code>
</h3>

<code class="doxybook">
<span>pool & </span><span><b>operator=</b>(const pool &) = default;</span></code>
<h3 id="function-~pool">
Function <code>thrust::mr::disjoint&#95;unsynchronized&#95;pool&#95;resource::pool::~pool</code>
</h3>

<code class="doxybook">
<span>__host__ </span><span><b>~pool</b>();</span></code>

