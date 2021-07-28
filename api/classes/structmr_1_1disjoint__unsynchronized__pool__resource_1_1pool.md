---
title: mr::disjoint_unsynchronized_pool_resource::pool
nav_exclude: true
has_children: true
has_toc: false
---

# Struct `mr::disjoint_unsynchronized_pool_resource::pool`

<code class="doxybook">
<span>struct mr::disjoint&#95;unsynchronized&#95;pool&#95;resource::pool {</span>
<span>public:</span><span>&nbsp;&nbsp;pointer_vector <b><a href="/api/classes/structmr_1_1disjoint__unsynchronized__pool__resource_1_1pool.html#variable-free_blocks">free&#95;blocks</a></b>;</span>
<br>
<span>&nbsp;&nbsp;std::size_t <b><a href="/api/classes/structmr_1_1disjoint__unsynchronized__pool__resource_1_1pool.html#variable-previous_allocated_count">previous&#95;allocated&#95;count</a></b>;</span>
<br>
<span>&nbsp;&nbsp;__host__ </span><span>&nbsp;&nbsp;<b><a href="/api/classes/structmr_1_1disjoint__unsynchronized__pool__resource_1_1pool.html#function-pool">pool</a></b>(const pointer_vector & free);</span>
<br>
<span>&nbsp;&nbsp;__host__ </span><span>&nbsp;&nbsp;<b><a href="/api/classes/structmr_1_1disjoint__unsynchronized__pool__resource_1_1pool.html#function-pool">pool</a></b>(const pool & other);</span>
<br>
<span>&nbsp;&nbsp;pool & </span><span>&nbsp;&nbsp;<b><a href="/api/classes/structmr_1_1disjoint__unsynchronized__pool__resource_1_1pool.html#function-operator=">operator=</a></b>(const pool &) = default;</span>
<br>
<span>&nbsp;&nbsp;__host__ </span><span>&nbsp;&nbsp;<b><a href="/api/classes/structmr_1_1disjoint__unsynchronized__pool__resource_1_1pool.html#function-~pool">~pool</a></b>();</span>
<span>};</span>
</code>

## Member Variables

<h3 id="variable-free_blocks">
Variable <code>mr::disjoint&#95;unsynchronized&#95;pool&#95;resource::pool::mr::disjoint&#95;unsynchronized&#95;pool&#95;resource::pool::free&#95;blocks</code>
</h3>

<code class="doxybook">
<span>pointer_vector <b>free_blocks</b>;</span></code>
<h3 id="variable-previous_allocated_count">
Variable <code>mr::disjoint&#95;unsynchronized&#95;pool&#95;resource::pool::mr::disjoint&#95;unsynchronized&#95;pool&#95;resource::pool::previous&#95;allocated&#95;count</code>
</h3>

<code class="doxybook">
<span>std::size_t <b>previous_allocated_count</b>;</span></code>

## Member Functions

<h3 id="function-pool">
Function <code>mr::disjoint&#95;unsynchronized&#95;pool&#95;resource::pool::&gt;::pool::pool</code>
</h3>

<code class="doxybook">
<span>__host__ </span><span><b>pool</b>(const pointer_vector & free);</span></code>
<h3 id="function-pool">
Function <code>mr::disjoint&#95;unsynchronized&#95;pool&#95;resource::pool::&gt;::pool::pool</code>
</h3>

<code class="doxybook">
<span>__host__ </span><span><b>pool</b>(const pool & other);</span></code>
<h3 id="function-operator=">
Function <code>mr::disjoint&#95;unsynchronized&#95;pool&#95;resource::pool::&gt;::pool::operator=</code>
</h3>

<code class="doxybook">
<span>pool & </span><span><b>operator=</b>(const pool &) = default;</span></code>
<h3 id="function-~pool">
Function <code>mr::disjoint&#95;unsynchronized&#95;pool&#95;resource::pool::&gt;::pool::~pool</code>
</h3>

<code class="doxybook">
<span>__host__ </span><span><b>~pool</b>();</span></code>

