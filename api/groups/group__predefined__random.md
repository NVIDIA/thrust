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
<span class="doxybook-comment">/* A random number engine with predefined parameters which implements a version of the Minimal Standard random number generation algorithm.  */</span><span>typedef <i>see below</i> <b><a href="/thrust/api/groups/group__predefined__random.html#typedef-minstd_rand0">minstd&#95;rand0</a></b>;</span>
<br>
<span class="doxybook-comment">/* A random number engine with predefined parameters which implements a version of the Minimal Standard random number generation algorithm.  */</span><span>typedef <i>see below</i> <b><a href="/thrust/api/groups/group__predefined__random.html#typedef-minstd_rand">minstd&#95;rand</a></b>;</span>
<br>
<span class="doxybook-comment">/* A random number engine with predefined parameters which implements the base engine of the <code>ranlux24</code> random number engine.  */</span><span>typedef <i>see below</i> <b><a href="/thrust/api/groups/group__predefined__random.html#typedef-ranlux24_base">ranlux24&#95;base</a></b>;</span>
<br>
<span class="doxybook-comment">/* A random number engine with predefined parameters which implements the base engine of the <code>ranlux48</code> random number engine.  */</span><span>typedef <i>see below</i> <b><a href="/thrust/api/groups/group__predefined__random.html#typedef-ranlux48_base">ranlux48&#95;base</a></b>;</span>
<br>
<span class="doxybook-comment">/* A random number engine with predefined parameters which implements the RANLUX level-3 random number generation algorithm.  */</span><span>typedef <i>see below</i> <b><a href="/thrust/api/groups/group__predefined__random.html#typedef-ranlux24">ranlux24</a></b>;</span>
<br>
<span class="doxybook-comment">/* A random number engine with predefined parameters which implements the RANLUX level-4 random number generation algorithm.  */</span><span>typedef <i>see below</i> <b><a href="/thrust/api/groups/group__predefined__random.html#typedef-ranlux48">ranlux48</a></b>;</span>
<br>
<span class="doxybook-comment">/* A random number engine with predefined parameters which implements L'Ecuyer's 1996 three-component Tausworthe random number generator.  */</span><span>typedef <i>see below</i> <b><a href="/thrust/api/groups/group__predefined__random.html#typedef-taus88">taus88</a></b>;</span>
<br>
<span class="doxybook-comment">/* An implementation-defined "default" random number engine.  */</span><span>typedef <i>see below</i> <b><a href="/thrust/api/groups/group__predefined__random.html#typedef-default_random_engine">default&#95;random&#95;engine</a></b>;</span>
</code>

## Types

<h3 id="typedef-minstd_rand0">
Typedef <code>minstd&#95;rand0</code>
</h3>

<code class="doxybook">
<span>typedef <a href="/thrust/api/classes/classrandom_1_1linear__congruential__engine.html">linear_congruential_engine</a>< thrust::detail::uint32_t, 16807, 0, 2147483647 ><b>minstd_rand0</b>;</span></code>
A random number engine with predefined parameters which implements a version of the Minimal Standard random number generation algorithm. 

**Note**:
The 10000th consecutive invocation of a default-constructed object of type <code>minstd&#95;rand0</code> shall produce the value <code>1043618065</code> . 

<h3 id="typedef-minstd_rand">
Typedef <code>minstd&#95;rand</code>
</h3>

<code class="doxybook">
<span>typedef <a href="/thrust/api/classes/classrandom_1_1linear__congruential__engine.html">linear_congruential_engine</a>< thrust::detail::uint32_t, 48271, 0, 2147483647 ><b>minstd_rand</b>;</span></code>
A random number engine with predefined parameters which implements a version of the Minimal Standard random number generation algorithm. 

**Note**:
The 10000th consecutive invocation of a default-constructed object of type <code>minstd&#95;rand</code> shall produce the value <code>399268537</code> . 

<h3 id="typedef-ranlux24_base">
Typedef <code>ranlux24&#95;base</code>
</h3>

<code class="doxybook">
<span>typedef <a href="/thrust/api/classes/classrandom_1_1subtract__with__carry__engine.html">subtract_with_carry_engine</a>< thrust::detail::uint32_t, 24, 10, 24 ><b>ranlux24_base</b>;</span></code>
A random number engine with predefined parameters which implements the base engine of the <code>ranlux24</code> random number engine. 

**Note**:
The 10000th consecutive invocation of a default-constructed object of type <code>ranlux24&#95;base</code> shall produce the value <code>7937952</code> . 

<h3 id="typedef-ranlux48_base">
Typedef <code>ranlux48&#95;base</code>
</h3>

<code class="doxybook">
<span>typedef <a href="/thrust/api/classes/classrandom_1_1subtract__with__carry__engine.html">subtract_with_carry_engine</a>< thrust::detail::uint64_t, 48, 5, 12 ><b>ranlux48_base</b>;</span></code>
A random number engine with predefined parameters which implements the base engine of the <code>ranlux48</code> random number engine. 

**Note**:
The 10000th consecutive invocation of a default-constructed object of type <code>ranlux48&#95;base</code> shall produce the value <code>192113843633948</code> . 

<h3 id="typedef-ranlux24">
Typedef <code>ranlux24</code>
</h3>

<code class="doxybook">
<span>typedef <a href="/thrust/api/classes/classrandom_1_1discard__block__engine.html">discard_block_engine</a>< ranlux24_base, 223, 23 ><b>ranlux24</b>;</span></code>
A random number engine with predefined parameters which implements the RANLUX level-3 random number generation algorithm. 

**Note**:
The 10000th consecutive invocation of a default-constructed object of type <code>ranlux24</code> shall produce the value <code>9901578</code> . 

<h3 id="typedef-ranlux48">
Typedef <code>ranlux48</code>
</h3>

<code class="doxybook">
<span>typedef <a href="/thrust/api/classes/classrandom_1_1discard__block__engine.html">discard_block_engine</a>< ranlux48_base, 389, 11 ><b>ranlux48</b>;</span></code>
A random number engine with predefined parameters which implements the RANLUX level-4 random number generation algorithm. 

**Note**:
The 10000th consecutive invocation of a default-constructed object of type <code>ranlux48</code> shall produce the value <code>88229545517833</code> . 

<h3 id="typedef-taus88">
Typedef <code>taus88</code>
</h3>

<code class="doxybook">
<span>typedef <a href="/thrust/api/classes/classrandom_1_1xor__combine__engine.html">xor_combine_engine</a>< <a href="/thrust/api/classes/classrandom_1_1linear__feedback__shift__engine.html">linear_feedback_shift_engine</a>< thrust::detail::uint32_t, 32u, 31u, 13u, 12u >, 0, <a href="/thrust/api/classes/classrandom_1_1xor__combine__engine.html">xor_combine_engine</a>< <a href="/thrust/api/classes/classrandom_1_1linear__feedback__shift__engine.html">linear_feedback_shift_engine</a>< thrust::detail::uint32_t, 32u, 29u, 2u, 4u >, 0, <a href="/thrust/api/classes/classrandom_1_1linear__feedback__shift__engine.html">linear_feedback_shift_engine</a>< thrust::detail::uint32_t, 32u, 28u, 3u, 17u >, 0 >, 0 ><b>taus88</b>;</span></code>
A random number engine with predefined parameters which implements L'Ecuyer's 1996 three-component Tausworthe random number generator. 

**Note**:
The 10000th consecutive invocation of a default-constructed object of type <code>taus88</code> shall produce the value <code>3535848941</code> . 

<h3 id="typedef-default_random_engine">
Typedef <code>default&#95;random&#95;engine</code>
</h3>

<code class="doxybook">
<span>typedef minstd_rand<b>default_random_engine</b>;</span></code>
An implementation-defined "default" random number engine. 

**Note**:
<code>default&#95;random&#95;engine</code> is currently an alias for <code>minstd&#95;rand</code>, and may change in a future version. 


