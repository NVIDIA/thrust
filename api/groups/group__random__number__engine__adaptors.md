---
title: Random Number Engine Adaptor Class Templates
parent: Random Number Generation
grand_parent: API
nav_exclude: false
has_children: true
has_toc: false
---

# Random Number Engine Adaptor Class Templates

<code class="doxybook">
<span class="doxybook-comment">/* A <code>discard&#95;block&#95;engine</code> adapts an existing base random number engine and produces random values by discarding some of the values returned by its base engine. Each cycle of the compound engine begins by returning <code>r</code> values successively produced by the base engine and ends by discarding <code>p-r</code> such values. The engine's state is the state of its base engine followed by the number of calls to <code>operator()</code> that have occurred since the beginning of the current cycle.  */</span><span>template &lt;typename Engine,</span>
<span>&nbsp;&nbsp;size_t p,</span>
<span>&nbsp;&nbsp;size_t r&gt;</span>
<span>class <b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1discard__block__engine.html">thrust::random::discard&#95;block&#95;engine</a></b>;</span>
<br>
<span class="doxybook-comment">/* An <code>xor&#95;combine&#95;engine</code> adapts two existing base random number engines and produces random values by combining the values produced by each.  */</span><span>template &lt;typename Engine1,</span>
<span>&nbsp;&nbsp;size_t s1,</span>
<span>&nbsp;&nbsp;typename Engine2,</span>
<span>&nbsp;&nbsp;size_t s2 = 0u&gt;</span>
<span>class <b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1xor__combine__engine.html">thrust::random::xor&#95;combine&#95;engine</a></b>;</span>
<br>
<span>template &lt;typename Engine,</span>
<span>&nbsp;&nbsp;size_t p,</span>
<span>&nbsp;&nbsp;size_t r&gt;</span>
<span>__host__ __device__ bool </span><span><b><a href="{{ site.baseurl }}/api/groups/group__random__number__engine__adaptors.html#function-operator==">thrust::random::operator==</a></b>(const discard_block_engine< Engine, p, r > & lhs,</span>
<span>&nbsp;&nbsp;const discard_block_engine< Engine, p, r > & rhs);</span>
<br>
<span>template &lt;typename Engine,</span>
<span>&nbsp;&nbsp;size_t p,</span>
<span>&nbsp;&nbsp;size_t r&gt;</span>
<span>__host__ __device__ bool </span><span><b><a href="{{ site.baseurl }}/api/groups/group__random__number__engine__adaptors.html#function-operator!=">thrust::random::operator!=</a></b>(const discard_block_engine< Engine, p, r > & lhs,</span>
<span>&nbsp;&nbsp;const discard_block_engine< Engine, p, r > & rhs);</span>
<br>
<span>template &lt;typename Engine,</span>
<span>&nbsp;&nbsp;size_t p,</span>
<span>&nbsp;&nbsp;size_t r,</span>
<span>&nbsp;&nbsp;typename CharT,</span>
<span>&nbsp;&nbsp;typename Traits&gt;</span>
<span>std::basic_ostream< CharT, Traits > & </span><span><b><a href="{{ site.baseurl }}/api/groups/group__random__number__engine__adaptors.html#function-operator<<">thrust::random::operator&lt;&lt;</a></b>(std::basic_ostream< CharT, Traits > & os,</span>
<span>&nbsp;&nbsp;const discard_block_engine< Engine, p, r > & e);</span>
<br>
<span>template &lt;typename Engine,</span>
<span>&nbsp;&nbsp;size_t p,</span>
<span>&nbsp;&nbsp;size_t r,</span>
<span>&nbsp;&nbsp;typename CharT,</span>
<span>&nbsp;&nbsp;typename Traits&gt;</span>
<span>std::basic_istream< CharT, Traits > & </span><span><b><a href="{{ site.baseurl }}/api/groups/group__random__number__engine__adaptors.html#function-operator>>">thrust::random::operator&gt;&gt;</a></b>(std::basic_istream< CharT, Traits > & is,</span>
<span>&nbsp;&nbsp;discard_block_engine< Engine, p, r > & e);</span>
<br>
<span>template &lt;typename Engine1_,</span>
<span>&nbsp;&nbsp;size_t s1_,</span>
<span>&nbsp;&nbsp;typename Engine2_,</span>
<span>&nbsp;&nbsp;size_t s2_&gt;</span>
<span>__host__ __device__ bool </span><span><b><a href="{{ site.baseurl }}/api/groups/group__random__number__engine__adaptors.html#function-operator==">thrust::random::operator==</a></b>(const xor_combine_engine< Engine1_, s1_, Engine2_, s2_ > & lhs,</span>
<span>&nbsp;&nbsp;const xor_combine_engine< Engine1_, s1_, Engine2_, s2_ > & rhs);</span>
<br>
<span>template &lt;typename Engine1_,</span>
<span>&nbsp;&nbsp;size_t s1_,</span>
<span>&nbsp;&nbsp;typename Engine2_,</span>
<span>&nbsp;&nbsp;size_t s2_&gt;</span>
<span>__host__ __device__ bool </span><span><b><a href="{{ site.baseurl }}/api/groups/group__random__number__engine__adaptors.html#function-operator!=">thrust::random::operator!=</a></b>(const xor_combine_engine< Engine1_, s1_, Engine2_, s2_ > & lhs,</span>
<span>&nbsp;&nbsp;const xor_combine_engine< Engine1_, s1_, Engine2_, s2_ > & rhs);</span>
<br>
<span>template &lt;typename Engine1_,</span>
<span>&nbsp;&nbsp;size_t s1_,</span>
<span>&nbsp;&nbsp;typename Engine2_,</span>
<span>&nbsp;&nbsp;size_t s2_,</span>
<span>&nbsp;&nbsp;typename CharT,</span>
<span>&nbsp;&nbsp;typename Traits&gt;</span>
<span>std::basic_ostream< CharT, Traits > & </span><span><b><a href="{{ site.baseurl }}/api/groups/group__random__number__engine__adaptors.html#function-operator<<">thrust::random::operator&lt;&lt;</a></b>(std::basic_ostream< CharT, Traits > & os,</span>
<span>&nbsp;&nbsp;const xor_combine_engine< Engine1_, s1_, Engine2_, s2_ > & e);</span>
<br>
<span>template &lt;typename Engine1_,</span>
<span>&nbsp;&nbsp;size_t s1_,</span>
<span>&nbsp;&nbsp;typename Engine2_,</span>
<span>&nbsp;&nbsp;size_t s2_,</span>
<span>&nbsp;&nbsp;typename CharT,</span>
<span>&nbsp;&nbsp;typename Traits&gt;</span>
<span>std::basic_istream< CharT, Traits > & </span><span><b><a href="{{ site.baseurl }}/api/groups/group__random__number__engine__adaptors.html#function-operator>>">thrust::random::operator&gt;&gt;</a></b>(std::basic_istream< CharT, Traits > & is,</span>
<span>&nbsp;&nbsp;xor_combine_engine< Engine1_, s1_, Engine2_, s2_ > & e);</span>
</code>

## Member Classes

<h3 id="class-thrustrandomdiscard-block-engine">
<a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1discard__block__engine.html">Class <code>thrust::random::discard&#95;block&#95;engine</code>
</a>
</h3>

A <code>discard&#95;block&#95;engine</code> adapts an existing base random number engine and produces random values by discarding some of the values returned by its base engine. Each cycle of the compound engine begins by returning <code>r</code> values successively produced by the base engine and ends by discarding <code>p-r</code> such values. The engine's state is the state of its base engine followed by the number of calls to <code>operator()</code> that have occurred since the beginning of the current cycle. 

<h3 id="class-thrustrandomxor-combine-engine">
<a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1xor__combine__engine.html">Class <code>thrust::random::xor&#95;combine&#95;engine</code>
</a>
</h3>

An <code>xor&#95;combine&#95;engine</code> adapts two existing base random number engines and produces random values by combining the values produced by each. 


## Functions

<h3 id="function-operator==">
Function <code>thrust::random::operator==</code>
</h3>

<code class="doxybook">
<span>template &lt;typename Engine,</span>
<span>&nbsp;&nbsp;size_t p,</span>
<span>&nbsp;&nbsp;size_t r&gt;</span>
<span>__host__ __device__ bool </span><span><b>operator==</b>(const discard_block_engine< Engine, p, r > & lhs,</span>
<span>&nbsp;&nbsp;const discard_block_engine< Engine, p, r > & rhs);</span></code>
This function checks two <code>discard&#95;block&#95;engines</code> for equality. 

**Function Parameters**:
* **`lhs`** The first <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1discard__block__engine.html">discard&#95;block&#95;engine</a></code> to test. 
* **`rhs`** The second <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1discard__block__engine.html">discard&#95;block&#95;engine</a></code> to test. 

**Returns**:
<code>true</code> if <code>lhs</code> is equal to <code>rhs</code>; <code>false</code>, otherwise. 

<h3 id="function-operator!=">
Function <code>thrust::random::operator!=</code>
</h3>

<code class="doxybook">
<span>template &lt;typename Engine,</span>
<span>&nbsp;&nbsp;size_t p,</span>
<span>&nbsp;&nbsp;size_t r&gt;</span>
<span>__host__ __device__ bool </span><span><b>operator!=</b>(const discard_block_engine< Engine, p, r > & lhs,</span>
<span>&nbsp;&nbsp;const discard_block_engine< Engine, p, r > & rhs);</span></code>
This function checks two <code>discard&#95;block&#95;engines</code> for inequality. 

**Function Parameters**:
* **`lhs`** The first <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1discard__block__engine.html">discard&#95;block&#95;engine</a></code> to test. 
* **`rhs`** The second <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1discard__block__engine.html">discard&#95;block&#95;engine</a></code> to test. 

**Returns**:
<code>true</code> if <code>lhs</code> is not equal to <code>rhs</code>; <code>false</code>, otherwise. 

<h3 id="function-operator<<">
Function <code>thrust::random::operator&lt;&lt;</code>
</h3>

<code class="doxybook">
<span>template &lt;typename Engine,</span>
<span>&nbsp;&nbsp;size_t p,</span>
<span>&nbsp;&nbsp;size_t r,</span>
<span>&nbsp;&nbsp;typename CharT,</span>
<span>&nbsp;&nbsp;typename Traits&gt;</span>
<span>std::basic_ostream< CharT, Traits > & </span><span><b>operator<<</b>(std::basic_ostream< CharT, Traits > & os,</span>
<span>&nbsp;&nbsp;const discard_block_engine< Engine, p, r > & e);</span></code>
This function streams a <a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1discard__block__engine.html">discard_block_engine</a> to a <code>std::basic&#95;ostream</code>. 

**Function Parameters**:
* **`os`** The <code>basic&#95;ostream</code> to stream out to. 
* **`e`** The <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1discard__block__engine.html">discard&#95;block&#95;engine</a></code> to stream out. 

**Returns**:
<code>os</code>

<h3 id="function-operator>>">
Function <code>thrust::random::operator&gt;&gt;</code>
</h3>

<code class="doxybook">
<span>template &lt;typename Engine,</span>
<span>&nbsp;&nbsp;size_t p,</span>
<span>&nbsp;&nbsp;size_t r,</span>
<span>&nbsp;&nbsp;typename CharT,</span>
<span>&nbsp;&nbsp;typename Traits&gt;</span>
<span>std::basic_istream< CharT, Traits > & </span><span><b>operator>></b>(std::basic_istream< CharT, Traits > & is,</span>
<span>&nbsp;&nbsp;discard_block_engine< Engine, p, r > & e);</span></code>
This function streams a <a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1discard__block__engine.html">discard_block_engine</a> in from a std::basic_istream. 

**Function Parameters**:
* **`is`** The <code>basic&#95;istream</code> to stream from. 
* **`e`** The <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1discard__block__engine.html">discard&#95;block&#95;engine</a></code> to stream in. 

**Returns**:
<code>is</code>

<h3 id="function-operator==">
Function <code>thrust::random::operator==</code>
</h3>

<code class="doxybook">
<span>template &lt;typename Engine1_,</span>
<span>&nbsp;&nbsp;size_t s1_,</span>
<span>&nbsp;&nbsp;typename Engine2_,</span>
<span>&nbsp;&nbsp;size_t s2_&gt;</span>
<span>__host__ __device__ bool </span><span><b>operator==</b>(const xor_combine_engine< Engine1_, s1_, Engine2_, s2_ > & lhs,</span>
<span>&nbsp;&nbsp;const xor_combine_engine< Engine1_, s1_, Engine2_, s2_ > & rhs);</span></code>
This function checks two <code>xor&#95;combine&#95;engines</code> for equality. 

**Function Parameters**:
* **`lhs`** The first <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1xor__combine__engine.html">xor&#95;combine&#95;engine</a></code> to test. 
* **`rhs`** The second <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1xor__combine__engine.html">xor&#95;combine&#95;engine</a></code> to test. 

**Returns**:
<code>true</code> if <code>lhs</code> is equal to <code>rhs</code>; <code>false</code>, otherwise. 

<h3 id="function-operator!=">
Function <code>thrust::random::operator!=</code>
</h3>

<code class="doxybook">
<span>template &lt;typename Engine1_,</span>
<span>&nbsp;&nbsp;size_t s1_,</span>
<span>&nbsp;&nbsp;typename Engine2_,</span>
<span>&nbsp;&nbsp;size_t s2_&gt;</span>
<span>__host__ __device__ bool </span><span><b>operator!=</b>(const xor_combine_engine< Engine1_, s1_, Engine2_, s2_ > & lhs,</span>
<span>&nbsp;&nbsp;const xor_combine_engine< Engine1_, s1_, Engine2_, s2_ > & rhs);</span></code>
This function checks two <code>xor&#95;combine&#95;engines</code> for inequality. 

**Function Parameters**:
* **`lhs`** The first <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1xor__combine__engine.html">xor&#95;combine&#95;engine</a></code> to test. 
* **`rhs`** The second <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1xor__combine__engine.html">xor&#95;combine&#95;engine</a></code> to test. 

**Returns**:
<code>true</code> if <code>lhs</code> is not equal to <code>rhs</code>; <code>false</code>, otherwise. 

<h3 id="function-operator<<">
Function <code>thrust::random::operator&lt;&lt;</code>
</h3>

<code class="doxybook">
<span>template &lt;typename Engine1_,</span>
<span>&nbsp;&nbsp;size_t s1_,</span>
<span>&nbsp;&nbsp;typename Engine2_,</span>
<span>&nbsp;&nbsp;size_t s2_,</span>
<span>&nbsp;&nbsp;typename CharT,</span>
<span>&nbsp;&nbsp;typename Traits&gt;</span>
<span>std::basic_ostream< CharT, Traits > & </span><span><b>operator<<</b>(std::basic_ostream< CharT, Traits > & os,</span>
<span>&nbsp;&nbsp;const xor_combine_engine< Engine1_, s1_, Engine2_, s2_ > & e);</span></code>
This function streams a <a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1xor__combine__engine.html">xor_combine_engine</a> to a <code>std::basic&#95;ostream</code>. 

**Function Parameters**:
* **`os`** The <code>basic&#95;ostream</code> to stream out to. 
* **`e`** The <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1xor__combine__engine.html">xor&#95;combine&#95;engine</a></code> to stream out. 

**Returns**:
<code>os</code>

<h3 id="function-operator>>">
Function <code>thrust::random::operator&gt;&gt;</code>
</h3>

<code class="doxybook">
<span>template &lt;typename Engine1_,</span>
<span>&nbsp;&nbsp;size_t s1_,</span>
<span>&nbsp;&nbsp;typename Engine2_,</span>
<span>&nbsp;&nbsp;size_t s2_,</span>
<span>&nbsp;&nbsp;typename CharT,</span>
<span>&nbsp;&nbsp;typename Traits&gt;</span>
<span>std::basic_istream< CharT, Traits > & </span><span><b>operator>></b>(std::basic_istream< CharT, Traits > & is,</span>
<span>&nbsp;&nbsp;xor_combine_engine< Engine1_, s1_, Engine2_, s2_ > & e);</span></code>
This function streams a <a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1xor__combine__engine.html">xor_combine_engine</a> in from a std::basic_istream. 

**Function Parameters**:
* **`is`** The <code>basic&#95;istream</code> to stream from. 
* **`e`** The <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1xor__combine__engine.html">xor&#95;combine&#95;engine</a></code> to stream in. 

**Returns**:
<code>is</code>


