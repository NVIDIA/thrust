---
title: Random Number Engines with Predefined Parameters
parent: Random Number Generation
grand_parent: API
nav_exclude: false
has_children: true
has_toc: false
---

# Random Number Engines with Predefined Parameters

<code class="doxybook">
<span class="doxybook-comment">/* A random number engine with predefined parameters which implements the RANLUX level-3 random number generation algorithm.  */</span><span>typedef <i>see below</i> <b><a href="{{ site.baseurl }}/api/groups/group__predefined__random.html#typedef-ranlux24">thrust::random::ranlux24</a></b>;</span>
<br>
<span class="doxybook-comment">/* A random number engine with predefined parameters which implements the RANLUX level-4 random number generation algorithm.  */</span><span>typedef <i>see below</i> <b><a href="{{ site.baseurl }}/api/groups/group__predefined__random.html#typedef-ranlux48">thrust::random::ranlux48</a></b>;</span>
<br>
<span class="doxybook-comment">/* A random number engine with predefined parameters which implements L'Ecuyer's 1996 three-component Tausworthe random number generator.  */</span><span>typedef <i>see below</i> <b><a href="{{ site.baseurl }}/api/groups/group__predefined__random.html#typedef-taus88">thrust::random::taus88</a></b>;</span>
<br>
<span class="doxybook-comment">/* An implementation-defined "default" random number engine.  */</span><span>typedef <i>see below</i> <b><a href="{{ site.baseurl }}/api/groups/group__predefined__random.html#typedef-default-random-engine">thrust::random::default&#95;random&#95;engine</a></b>;</span>
<br>
<span class="doxybook-comment">/* A random number engine with predefined parameters which implements a version of the Minimal Standard random number generation algorithm.  */</span><span>typedef <i>see below</i> <b><a href="{{ site.baseurl }}/api/groups/group__predefined__random.html#typedef-minstd-rand0">thrust::random::minstd&#95;rand0</a></b>;</span>
<br>
<span class="doxybook-comment">/* A random number engine with predefined parameters which implements a version of the Minimal Standard random number generation algorithm.  */</span><span>typedef <i>see below</i> <b><a href="{{ site.baseurl }}/api/groups/group__predefined__random.html#typedef-minstd-rand">thrust::random::minstd&#95;rand</a></b>;</span>
<br>
<span class="doxybook-comment">/* A random number engine with predefined parameters which implements the base engine of the <code>ranlux24</code> random number engine.  */</span><span>typedef <i>see below</i> <b><a href="{{ site.baseurl }}/api/groups/group__predefined__random.html#typedef-ranlux24-base">thrust::random::ranlux24&#95;base</a></b>;</span>
<br>
<span class="doxybook-comment">/* A random number engine with predefined parameters which implements the base engine of the <code>ranlux48</code> random number engine.  */</span><span>typedef <i>see below</i> <b><a href="{{ site.baseurl }}/api/groups/group__predefined__random.html#typedef-ranlux48-base">thrust::random::ranlux48&#95;base</a></b>;</span>
</code>

## Types

<h3 id="typedef-ranlux24">
Typedef <code>thrust::random::ranlux24</code>
</h3>

<code class="doxybook">
<span>typedef discard_block_engine< ranlux24_base, 223, 23 ><b>ranlux24</b>;</span></code>
A random number engine with predefined parameters which implements the RANLUX level-3 random number generation algorithm. 

**Note**:
The 10000th consecutive invocation of a default-constructed object of type <code>ranlux24</code> shall produce the value <code>9901578</code> . 

<h3 id="typedef-ranlux48">
Typedef <code>thrust::random::ranlux48</code>
</h3>

<code class="doxybook">
<span>typedef discard_block_engine< ranlux48_base, 389, 11 ><b>ranlux48</b>;</span></code>
A random number engine with predefined parameters which implements the RANLUX level-4 random number generation algorithm. 

**Note**:
The 10000th consecutive invocation of a default-constructed object of type <code>ranlux48</code> shall produce the value <code>88229545517833</code> . 

<h3 id="typedef-taus88">
Typedef <code>thrust::random::taus88</code>
</h3>

<code class="doxybook">
<span>typedef xor_combine_engine< linear_feedback_shift_engine< thrust::detail::uint32_t, 32u, 31u, 13u, 12u >, 0, xor_combine_engine< linear_feedback_shift_engine< thrust::detail::uint32_t, 32u, 29u, 2u, 4u >, 0, linear_feedback_shift_engine< thrust::detail::uint32_t, 32u, 28u, 3u, 17u >, 0 >, 0 ><b>taus88</b>;</span></code>
A random number engine with predefined parameters which implements L'Ecuyer's 1996 three-component Tausworthe random number generator. 

**Note**:
The 10000th consecutive invocation of a default-constructed object of type <code>taus88</code> shall produce the value <code>3535848941</code> . 

<h3 id="typedef-default-random-engine">
Typedef <code>thrust::random::default&#95;random&#95;engine</code>
</h3>

<code class="doxybook">
<span>typedef minstd_rand<b>default_random_engine</b>;</span></code>
An implementation-defined "default" random number engine. 

**Note**:
<code>default&#95;random&#95;engine</code> is currently an alias for <code>minstd&#95;rand</code>, and may change in a future version. 

<h3 id="typedef-minstd-rand0">
Typedef <code>thrust::random::minstd&#95;rand0</code>
</h3>

<code class="doxybook">
<span>typedef linear_congruential_engine< thrust::detail::uint32_t, 16807, 0, 2147483647 ><b>minstd_rand0</b>;</span></code>
A random number engine with predefined parameters which implements a version of the Minimal Standard random number generation algorithm. 

**Note**:
The 10000th consecutive invocation of a default-constructed object of type <code>minstd&#95;rand0</code> shall produce the value <code>1043618065</code> . 

<h3 id="typedef-minstd-rand">
Typedef <code>thrust::random::minstd&#95;rand</code>
</h3>

<code class="doxybook">
<span>typedef linear_congruential_engine< thrust::detail::uint32_t, 48271, 0, 2147483647 ><b>minstd_rand</b>;</span></code>
A random number engine with predefined parameters which implements a version of the Minimal Standard random number generation algorithm. 

**Note**:
The 10000th consecutive invocation of a default-constructed object of type <code>minstd&#95;rand</code> shall produce the value <code>399268537</code> . 

<h3 id="typedef-ranlux24-base">
Typedef <code>thrust::random::ranlux24&#95;base</code>
</h3>

<code class="doxybook">
<span>typedef subtract_with_carry_engine< thrust::detail::uint32_t, 24, 10, 24 ><b>ranlux24_base</b>;</span></code>
A random number engine with predefined parameters which implements the base engine of the <code>ranlux24</code> random number engine. 

**Note**:
The 10000th consecutive invocation of a default-constructed object of type <code>ranlux24&#95;base</code> shall produce the value <code>7937952</code> . 

<h3 id="typedef-ranlux48-base">
Typedef <code>thrust::random::ranlux48&#95;base</code>
</h3>

<code class="doxybook">
<span>typedef subtract_with_carry_engine< thrust::detail::uint64_t, 48, 5, 12 ><b>ranlux48_base</b>;</span></code>
A random number engine with predefined parameters which implements the base engine of the <code>ranlux48</code> random number engine. 

**Note**:
The 10000th consecutive invocation of a default-constructed object of type <code>ranlux48&#95;base</code> shall produce the value <code>192113843633948</code> . 


