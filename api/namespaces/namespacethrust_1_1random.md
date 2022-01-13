---
title: thrust::random
summary: thrust::random is the namespace which contains random number engine class templates, random number engine adaptor class templates, engines with predefined parameters, and random number distribution class templates. They are provided in a separate namespace for import convenience but are also aliased in the top-level thrust namespace for easy access. 
parent: Random Number Generation
nav_exclude: true
has_children: true
has_toc: false
---

# Namespace `thrust::random`

<code class="doxybook">
<span>namespace thrust::random {</span>
<br>
<span class="doxybook-comment">/* A <code>discard&#95;block&#95;engine</code> adapts an existing base random number engine and produces random values by discarding some of the values returned by its base engine. Each cycle of the compound engine begins by returning <code>r</code> values successively produced by the base engine and ends by discarding <code>p-r</code> such values. The engine's state is the state of its base engine followed by the number of calls to <code>operator()</code> that have occurred since the beginning of the current cycle.  */</span><span>template &lt;typename Engine,</span>
<span>&nbsp;&nbsp;size_t p,</span>
<span>&nbsp;&nbsp;size_t r&gt;</span>
<span>class <b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1discard__block__engine.html">discard&#95;block&#95;engine</a></b>;</span>
<br>
<span class="doxybook-comment">/* A <code>linear&#95;congruential&#95;engine</code> random number engine produces unsigned integer random numbers using a linear congruential random number generation algorithm.  */</span><span>template &lt;typename UIntType,</span>
<span>&nbsp;&nbsp;UIntType a,</span>
<span>&nbsp;&nbsp;UIntType c,</span>
<span>&nbsp;&nbsp;UIntType m&gt;</span>
<span>class <b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1linear__congruential__engine.html">linear&#95;congruential&#95;engine</a></b>;</span>
<br>
<span class="doxybook-comment">/* A <code>linear&#95;feedback&#95;shift&#95;engine</code> random number engine produces unsigned integer random values using a linear feedback shift random number generation algorithm.  */</span><span>template &lt;typename UIntType,</span>
<span>&nbsp;&nbsp;size_t w,</span>
<span>&nbsp;&nbsp;size_t k,</span>
<span>&nbsp;&nbsp;size_t q,</span>
<span>&nbsp;&nbsp;size_t s&gt;</span>
<span>class <b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1linear__feedback__shift__engine.html">linear&#95;feedback&#95;shift&#95;engine</a></b>;</span>
<br>
<span class="doxybook-comment">/* A <code>normal&#95;distribution</code> random number distribution produces floating point Normally distributed random numbers.  */</span><span>template &lt;typename RealType = double&gt;</span>
<span>class <b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1normal__distribution.html">normal&#95;distribution</a></b>;</span>
<br>
<span class="doxybook-comment">/* A <code>subtract&#95;with&#95;carry&#95;engine</code> random number engine produces unsigned integer random numbers using the subtract with carry algorithm of Marsaglia & Zaman.  */</span><span>template &lt;typename UIntType,</span>
<span>&nbsp;&nbsp;size_t w,</span>
<span>&nbsp;&nbsp;size_t s,</span>
<span>&nbsp;&nbsp;size_t r&gt;</span>
<span>class <b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1subtract__with__carry__engine.html">subtract&#95;with&#95;carry&#95;engine</a></b>;</span>
<br>
<span class="doxybook-comment">/* A <code>uniform&#95;int&#95;distribution</code> random number distribution produces signed or unsigned integer uniform random numbers from a given range.  */</span><span>template &lt;typename IntType = int&gt;</span>
<span>class <b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1uniform__int__distribution.html">uniform&#95;int&#95;distribution</a></b>;</span>
<br>
<span class="doxybook-comment">/* A <code>uniform&#95;real&#95;distribution</code> random number distribution produces floating point uniform random numbers from a half-open interval.  */</span><span>template &lt;typename RealType = double&gt;</span>
<span>class <b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1uniform__real__distribution.html">uniform&#95;real&#95;distribution</a></b>;</span>
<br>
<span class="doxybook-comment">/* An <code>xor&#95;combine&#95;engine</code> adapts two existing base random number engines and produces random values by combining the values produced by each.  */</span><span>template &lt;typename Engine1,</span>
<span>&nbsp;&nbsp;size_t s1,</span>
<span>&nbsp;&nbsp;typename Engine2,</span>
<span>&nbsp;&nbsp;size_t s2 = 0u&gt;</span>
<span>class <b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1xor__combine__engine.html">xor&#95;combine&#95;engine</a></b>;</span>
<br>
<span class="doxybook-comment">/* A random number engine with predefined parameters which implements the RANLUX level-3 random number generation algorithm.  */</span><span>typedef <i>see below</i> <b><a href="{{ site.baseurl }}/api/groups/group__predefined__random.html#typedef-ranlux24">ranlux24</a></b>;</span>
<br>
<span class="doxybook-comment">/* A random number engine with predefined parameters which implements the RANLUX level-4 random number generation algorithm.  */</span><span>typedef <i>see below</i> <b><a href="{{ site.baseurl }}/api/groups/group__predefined__random.html#typedef-ranlux48">ranlux48</a></b>;</span>
<br>
<span class="doxybook-comment">/* A random number engine with predefined parameters which implements L'Ecuyer's 1996 three-component Tausworthe random number generator.  */</span><span>typedef <i>see below</i> <b><a href="{{ site.baseurl }}/api/groups/group__predefined__random.html#typedef-taus88">taus88</a></b>;</span>
<br>
<span class="doxybook-comment">/* An implementation-defined "default" random number engine.  */</span><span>typedef <i>see below</i> <b><a href="{{ site.baseurl }}/api/groups/group__predefined__random.html#typedef-default-random-engine">default&#95;random&#95;engine</a></b>;</span>
<br>
<span class="doxybook-comment">/* A random number engine with predefined parameters which implements a version of the Minimal Standard random number generation algorithm.  */</span><span>typedef <i>see below</i> <b><a href="{{ site.baseurl }}/api/groups/group__predefined__random.html#typedef-minstd-rand0">minstd&#95;rand0</a></b>;</span>
<br>
<span class="doxybook-comment">/* A random number engine with predefined parameters which implements a version of the Minimal Standard random number generation algorithm.  */</span><span>typedef <i>see below</i> <b><a href="{{ site.baseurl }}/api/groups/group__predefined__random.html#typedef-minstd-rand">minstd&#95;rand</a></b>;</span>
<br>
<span class="doxybook-comment">/* A random number engine with predefined parameters which implements the base engine of the <code>ranlux24</code> random number engine.  */</span><span>typedef <i>see below</i> <b><a href="{{ site.baseurl }}/api/groups/group__predefined__random.html#typedef-ranlux24-base">ranlux24&#95;base</a></b>;</span>
<br>
<span class="doxybook-comment">/* A random number engine with predefined parameters which implements the base engine of the <code>ranlux48</code> random number engine.  */</span><span>typedef <i>see below</i> <b><a href="{{ site.baseurl }}/api/groups/group__predefined__random.html#typedef-ranlux48-base">ranlux48&#95;base</a></b>;</span>
<br>
<span>template &lt;typename Engine,</span>
<span>&nbsp;&nbsp;size_t p,</span>
<span>&nbsp;&nbsp;size_t r&gt;</span>
<span>__host__ __device__ bool </span><span><b><a href="{{ site.baseurl }}/api/groups/group__random__number__engine__adaptors.html#function-operator==">operator==</a></b>(const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1discard__block__engine.html">discard_block_engine</a>< Engine, p, r > & lhs,</span>
<span>&nbsp;&nbsp;const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1discard__block__engine.html">discard_block_engine</a>< Engine, p, r > & rhs);</span>
<br>
<span>template &lt;typename Engine,</span>
<span>&nbsp;&nbsp;size_t p,</span>
<span>&nbsp;&nbsp;size_t r&gt;</span>
<span>__host__ __device__ bool </span><span><b><a href="{{ site.baseurl }}/api/groups/group__random__number__engine__adaptors.html#function-operator!=">operator!=</a></b>(const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1discard__block__engine.html">discard_block_engine</a>< Engine, p, r > & lhs,</span>
<span>&nbsp;&nbsp;const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1discard__block__engine.html">discard_block_engine</a>< Engine, p, r > & rhs);</span>
<br>
<span>template &lt;typename Engine,</span>
<span>&nbsp;&nbsp;size_t p,</span>
<span>&nbsp;&nbsp;size_t r,</span>
<span>&nbsp;&nbsp;typename CharT,</span>
<span>&nbsp;&nbsp;typename Traits&gt;</span>
<span>std::basic_ostream< CharT, Traits > & </span><span><b><a href="{{ site.baseurl }}/api/groups/group__random__number__engine__adaptors.html#function-operator<<">operator&lt;&lt;</a></b>(std::basic_ostream< CharT, Traits > & os,</span>
<span>&nbsp;&nbsp;const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1discard__block__engine.html">discard_block_engine</a>< Engine, p, r > & e);</span>
<br>
<span>template &lt;typename Engine,</span>
<span>&nbsp;&nbsp;size_t p,</span>
<span>&nbsp;&nbsp;size_t r,</span>
<span>&nbsp;&nbsp;typename CharT,</span>
<span>&nbsp;&nbsp;typename Traits&gt;</span>
<span>std::basic_istream< CharT, Traits > & </span><span><b><a href="{{ site.baseurl }}/api/groups/group__random__number__engine__adaptors.html#function-operator>>">operator&gt;&gt;</a></b>(std::basic_istream< CharT, Traits > & is,</span>
<span>&nbsp;&nbsp;<a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1discard__block__engine.html">discard_block_engine</a>< Engine, p, r > & e);</span>
<br>
<span>template &lt;typename UIntType_,</span>
<span>&nbsp;&nbsp;UIntType_ a_,</span>
<span>&nbsp;&nbsp;UIntType_ c_,</span>
<span>&nbsp;&nbsp;UIntType_ m_&gt;</span>
<span>__host__ __device__ bool </span><span><b><a href="{{ site.baseurl }}/api/groups/group__random__number__engine__templates.html#function-operator==">operator==</a></b>(const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1linear__congruential__engine.html">linear_congruential_engine</a>< UIntType_, a_, c_, m_ > & lhs,</span>
<span>&nbsp;&nbsp;const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1linear__congruential__engine.html">linear_congruential_engine</a>< UIntType_, a_, c_, m_ > & rhs);</span>
<br>
<span>template &lt;typename UIntType_,</span>
<span>&nbsp;&nbsp;UIntType_ a_,</span>
<span>&nbsp;&nbsp;UIntType_ c_,</span>
<span>&nbsp;&nbsp;UIntType_ m_&gt;</span>
<span>__host__ __device__ bool </span><span><b><a href="{{ site.baseurl }}/api/groups/group__random__number__engine__templates.html#function-operator!=">operator!=</a></b>(const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1linear__congruential__engine.html">linear_congruential_engine</a>< UIntType_, a_, c_, m_ > & lhs,</span>
<span>&nbsp;&nbsp;const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1linear__congruential__engine.html">linear_congruential_engine</a>< UIntType_, a_, c_, m_ > & rhs);</span>
<br>
<span>template &lt;typename UIntType_,</span>
<span>&nbsp;&nbsp;UIntType_ a_,</span>
<span>&nbsp;&nbsp;UIntType_ c_,</span>
<span>&nbsp;&nbsp;UIntType_ m_,</span>
<span>&nbsp;&nbsp;typename CharT,</span>
<span>&nbsp;&nbsp;typename Traits&gt;</span>
<span>std::basic_ostream< CharT, Traits > & </span><span><b><a href="{{ site.baseurl }}/api/groups/group__random__number__engine__templates.html#function-operator<<">operator&lt;&lt;</a></b>(std::basic_ostream< CharT, Traits > & os,</span>
<span>&nbsp;&nbsp;const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1linear__congruential__engine.html">linear_congruential_engine</a>< UIntType_, a_, c_, m_ > & e);</span>
<br>
<span>template &lt;typename UIntType_,</span>
<span>&nbsp;&nbsp;UIntType_ a_,</span>
<span>&nbsp;&nbsp;UIntType_ c_,</span>
<span>&nbsp;&nbsp;UIntType_ m_,</span>
<span>&nbsp;&nbsp;typename CharT,</span>
<span>&nbsp;&nbsp;typename Traits&gt;</span>
<span>std::basic_istream< CharT, Traits > & </span><span><b><a href="{{ site.baseurl }}/api/groups/group__random__number__engine__templates.html#function-operator>>">operator&gt;&gt;</a></b>(std::basic_istream< CharT, Traits > & is,</span>
<span>&nbsp;&nbsp;<a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1linear__congruential__engine.html">linear_congruential_engine</a>< UIntType_, a_, c_, m_ > & e);</span>
<br>
<span>template &lt;typename UIntType_,</span>
<span>&nbsp;&nbsp;size_t w_,</span>
<span>&nbsp;&nbsp;size_t k_,</span>
<span>&nbsp;&nbsp;size_t q_,</span>
<span>&nbsp;&nbsp;size_t s_&gt;</span>
<span>__host__ __device__ bool </span><span><b><a href="{{ site.baseurl }}/api/groups/group__random__number__engine__templates.html#function-operator==">operator==</a></b>(const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1linear__feedback__shift__engine.html">linear_feedback_shift_engine</a>< UIntType_, w_, k_, q_, s_ > & lhs,</span>
<span>&nbsp;&nbsp;const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1linear__feedback__shift__engine.html">linear_feedback_shift_engine</a>< UIntType_, w_, k_, q_, s_ > & rhs);</span>
<br>
<span>template &lt;typename UIntType_,</span>
<span>&nbsp;&nbsp;size_t w_,</span>
<span>&nbsp;&nbsp;size_t k_,</span>
<span>&nbsp;&nbsp;size_t q_,</span>
<span>&nbsp;&nbsp;size_t s_&gt;</span>
<span>__host__ __device__ bool </span><span><b><a href="{{ site.baseurl }}/api/groups/group__random__number__engine__templates.html#function-operator!=">operator!=</a></b>(const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1linear__feedback__shift__engine.html">linear_feedback_shift_engine</a>< UIntType_, w_, k_, q_, s_ > & lhs,</span>
<span>&nbsp;&nbsp;const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1linear__feedback__shift__engine.html">linear_feedback_shift_engine</a>< UIntType_, w_, k_, q_, s_ > & rhs);</span>
<br>
<span>template &lt;typename UIntType_,</span>
<span>&nbsp;&nbsp;size_t w_,</span>
<span>&nbsp;&nbsp;size_t k_,</span>
<span>&nbsp;&nbsp;size_t q_,</span>
<span>&nbsp;&nbsp;size_t s_,</span>
<span>&nbsp;&nbsp;typename CharT,</span>
<span>&nbsp;&nbsp;typename Traits&gt;</span>
<span>std::basic_ostream< CharT, Traits > & </span><span><b><a href="{{ site.baseurl }}/api/groups/group__random__number__engine__templates.html#function-operator<<">operator&lt;&lt;</a></b>(std::basic_ostream< CharT, Traits > & os,</span>
<span>&nbsp;&nbsp;const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1linear__feedback__shift__engine.html">linear_feedback_shift_engine</a>< UIntType_, w_, k_, q_, s_ > & e);</span>
<br>
<span>template &lt;typename UIntType_,</span>
<span>&nbsp;&nbsp;size_t w_,</span>
<span>&nbsp;&nbsp;size_t k_,</span>
<span>&nbsp;&nbsp;size_t q_,</span>
<span>&nbsp;&nbsp;size_t s_,</span>
<span>&nbsp;&nbsp;typename CharT,</span>
<span>&nbsp;&nbsp;typename Traits&gt;</span>
<span>std::basic_istream< CharT, Traits > & </span><span><b><a href="{{ site.baseurl }}/api/groups/group__random__number__engine__templates.html#function-operator>>">operator&gt;&gt;</a></b>(std::basic_istream< CharT, Traits > & is,</span>
<span>&nbsp;&nbsp;<a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1linear__feedback__shift__engine.html">linear_feedback_shift_engine</a>< UIntType_, w_, k_, q_, s_ > & e);</span>
<br>
<span>template &lt;typename RealType&gt;</span>
<span>__host__ __device__ bool </span><span><b><a href="{{ site.baseurl }}/api/groups/group__random__number__distributions.html#function-operator==">operator==</a></b>(const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1normal__distribution.html">normal_distribution</a>< RealType > & lhs,</span>
<span>&nbsp;&nbsp;const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1normal__distribution.html">normal_distribution</a>< RealType > & rhs);</span>
<br>
<span>template &lt;typename RealType&gt;</span>
<span>__host__ __device__ bool </span><span><b><a href="{{ site.baseurl }}/api/groups/group__random__number__distributions.html#function-operator!=">operator!=</a></b>(const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1normal__distribution.html">normal_distribution</a>< RealType > & lhs,</span>
<span>&nbsp;&nbsp;const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1normal__distribution.html">normal_distribution</a>< RealType > & rhs);</span>
<br>
<span>template &lt;typename RealType,</span>
<span>&nbsp;&nbsp;typename CharT,</span>
<span>&nbsp;&nbsp;typename Traits&gt;</span>
<span>std::basic_ostream< CharT, Traits > & </span><span><b><a href="{{ site.baseurl }}/api/groups/group__random__number__distributions.html#function-operator<<">operator&lt;&lt;</a></b>(std::basic_ostream< CharT, Traits > & os,</span>
<span>&nbsp;&nbsp;const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1normal__distribution.html">normal_distribution</a>< RealType > & d);</span>
<br>
<span>template &lt;typename RealType,</span>
<span>&nbsp;&nbsp;typename CharT,</span>
<span>&nbsp;&nbsp;typename Traits&gt;</span>
<span>std::basic_istream< CharT, Traits > & </span><span><b><a href="{{ site.baseurl }}/api/groups/group__random__number__distributions.html#function-operator>>">operator&gt;&gt;</a></b>(std::basic_istream< CharT, Traits > & is,</span>
<span>&nbsp;&nbsp;<a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1normal__distribution.html">normal_distribution</a>< RealType > & d);</span>
<br>
<span>template &lt;typename UIntType_,</span>
<span>&nbsp;&nbsp;size_t w_,</span>
<span>&nbsp;&nbsp;size_t s_,</span>
<span>&nbsp;&nbsp;size_t r_&gt;</span>
<span>__host__ __device__ bool </span><span><b><a href="{{ site.baseurl }}/api/groups/group__random__number__engine__templates.html#function-operator==">operator==</a></b>(const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1subtract__with__carry__engine.html">subtract_with_carry_engine</a>< UIntType_, w_, s_, r_ > & lhs,</span>
<span>&nbsp;&nbsp;const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1subtract__with__carry__engine.html">subtract_with_carry_engine</a>< UIntType_, w_, s_, r_ > & rhs);</span>
<br>
<span>template &lt;typename UIntType_,</span>
<span>&nbsp;&nbsp;size_t w_,</span>
<span>&nbsp;&nbsp;size_t s_,</span>
<span>&nbsp;&nbsp;size_t r_&gt;</span>
<span>__host__ __device__ bool </span><span><b><a href="{{ site.baseurl }}/api/groups/group__random__number__engine__templates.html#function-operator!=">operator!=</a></b>(const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1subtract__with__carry__engine.html">subtract_with_carry_engine</a>< UIntType_, w_, s_, r_ > & lhs,</span>
<span>&nbsp;&nbsp;const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1subtract__with__carry__engine.html">subtract_with_carry_engine</a>< UIntType_, w_, s_, r_ > & rhs);</span>
<br>
<span>template &lt;typename UIntType_,</span>
<span>&nbsp;&nbsp;size_t w_,</span>
<span>&nbsp;&nbsp;size_t s_,</span>
<span>&nbsp;&nbsp;size_t r_,</span>
<span>&nbsp;&nbsp;typename CharT,</span>
<span>&nbsp;&nbsp;typename Traits&gt;</span>
<span>std::basic_ostream< CharT, Traits > & </span><span><b><a href="{{ site.baseurl }}/api/groups/group__random__number__engine__templates.html#function-operator<<">operator&lt;&lt;</a></b>(std::basic_ostream< CharT, Traits > & os,</span>
<span>&nbsp;&nbsp;const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1subtract__with__carry__engine.html">subtract_with_carry_engine</a>< UIntType_, w_, s_, r_ > & e);</span>
<br>
<span>template &lt;typename UIntType_,</span>
<span>&nbsp;&nbsp;size_t w_,</span>
<span>&nbsp;&nbsp;size_t s_,</span>
<span>&nbsp;&nbsp;size_t r_,</span>
<span>&nbsp;&nbsp;typename CharT,</span>
<span>&nbsp;&nbsp;typename Traits&gt;</span>
<span>std::basic_istream< CharT, Traits > & </span><span><b><a href="{{ site.baseurl }}/api/groups/group__random__number__engine__templates.html#function-operator>>">operator&gt;&gt;</a></b>(std::basic_istream< CharT, Traits > & is,</span>
<span>&nbsp;&nbsp;<a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1subtract__with__carry__engine.html">subtract_with_carry_engine</a>< UIntType_, w_, s_, r_ > & e);</span>
<br>
<span>template &lt;typename IntType&gt;</span>
<span>__host__ __device__ bool </span><span><b><a href="{{ site.baseurl }}/api/groups/group__random__number__distributions.html#function-operator==">operator==</a></b>(const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1uniform__int__distribution.html">uniform_int_distribution</a>< IntType > & lhs,</span>
<span>&nbsp;&nbsp;const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1uniform__int__distribution.html">uniform_int_distribution</a>< IntType > & rhs);</span>
<br>
<span>template &lt;typename IntType&gt;</span>
<span>__host__ __device__ bool </span><span><b><a href="{{ site.baseurl }}/api/groups/group__random__number__distributions.html#function-operator!=">operator!=</a></b>(const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1uniform__int__distribution.html">uniform_int_distribution</a>< IntType > & lhs,</span>
<span>&nbsp;&nbsp;const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1uniform__int__distribution.html">uniform_int_distribution</a>< IntType > & rhs);</span>
<br>
<span>template &lt;typename IntType,</span>
<span>&nbsp;&nbsp;typename CharT,</span>
<span>&nbsp;&nbsp;typename Traits&gt;</span>
<span>std::basic_ostream< CharT, Traits > & </span><span><b><a href="{{ site.baseurl }}/api/groups/group__random__number__distributions.html#function-operator<<">operator&lt;&lt;</a></b>(std::basic_ostream< CharT, Traits > & os,</span>
<span>&nbsp;&nbsp;const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1uniform__int__distribution.html">uniform_int_distribution</a>< IntType > & d);</span>
<br>
<span>template &lt;typename IntType,</span>
<span>&nbsp;&nbsp;typename CharT,</span>
<span>&nbsp;&nbsp;typename Traits&gt;</span>
<span>std::basic_istream< CharT, Traits > & </span><span><b><a href="{{ site.baseurl }}/api/groups/group__random__number__distributions.html#function-operator>>">operator&gt;&gt;</a></b>(std::basic_istream< CharT, Traits > & is,</span>
<span>&nbsp;&nbsp;<a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1uniform__int__distribution.html">uniform_int_distribution</a>< IntType > & d);</span>
<br>
<span>template &lt;typename RealType&gt;</span>
<span>__host__ __device__ bool </span><span><b><a href="{{ site.baseurl }}/api/groups/group__random__number__distributions.html#function-operator==">operator==</a></b>(const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1uniform__real__distribution.html">uniform_real_distribution</a>< RealType > & lhs,</span>
<span>&nbsp;&nbsp;const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1uniform__real__distribution.html">uniform_real_distribution</a>< RealType > & rhs);</span>
<br>
<span>template &lt;typename RealType&gt;</span>
<span>__host__ __device__ bool </span><span><b><a href="{{ site.baseurl }}/api/groups/group__random__number__distributions.html#function-operator!=">operator!=</a></b>(const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1uniform__real__distribution.html">uniform_real_distribution</a>< RealType > & lhs,</span>
<span>&nbsp;&nbsp;const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1uniform__real__distribution.html">uniform_real_distribution</a>< RealType > & rhs);</span>
<br>
<span>template &lt;typename RealType,</span>
<span>&nbsp;&nbsp;typename CharT,</span>
<span>&nbsp;&nbsp;typename Traits&gt;</span>
<span>std::basic_ostream< CharT, Traits > & </span><span><b><a href="{{ site.baseurl }}/api/groups/group__random__number__distributions.html#function-operator<<">operator&lt;&lt;</a></b>(std::basic_ostream< CharT, Traits > & os,</span>
<span>&nbsp;&nbsp;const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1uniform__real__distribution.html">uniform_real_distribution</a>< RealType > & d);</span>
<br>
<span>template &lt;typename RealType,</span>
<span>&nbsp;&nbsp;typename CharT,</span>
<span>&nbsp;&nbsp;typename Traits&gt;</span>
<span>std::basic_istream< CharT, Traits > & </span><span><b><a href="{{ site.baseurl }}/api/groups/group__random__number__distributions.html#function-operator>>">operator&gt;&gt;</a></b>(std::basic_istream< CharT, Traits > & is,</span>
<span>&nbsp;&nbsp;<a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1uniform__real__distribution.html">uniform_real_distribution</a>< RealType > & d);</span>
<br>
<span>template &lt;typename Engine1_,</span>
<span>&nbsp;&nbsp;size_t s1_,</span>
<span>&nbsp;&nbsp;typename Engine2_,</span>
<span>&nbsp;&nbsp;size_t s2_&gt;</span>
<span>__host__ __device__ bool </span><span><b><a href="{{ site.baseurl }}/api/groups/group__random__number__engine__adaptors.html#function-operator==">operator==</a></b>(const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1xor__combine__engine.html">xor_combine_engine</a>< Engine1_, s1_, Engine2_, s2_ > & lhs,</span>
<span>&nbsp;&nbsp;const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1xor__combine__engine.html">xor_combine_engine</a>< Engine1_, s1_, Engine2_, s2_ > & rhs);</span>
<br>
<span>template &lt;typename Engine1_,</span>
<span>&nbsp;&nbsp;size_t s1_,</span>
<span>&nbsp;&nbsp;typename Engine2_,</span>
<span>&nbsp;&nbsp;size_t s2_&gt;</span>
<span>__host__ __device__ bool </span><span><b><a href="{{ site.baseurl }}/api/groups/group__random__number__engine__adaptors.html#function-operator!=">operator!=</a></b>(const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1xor__combine__engine.html">xor_combine_engine</a>< Engine1_, s1_, Engine2_, s2_ > & lhs,</span>
<span>&nbsp;&nbsp;const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1xor__combine__engine.html">xor_combine_engine</a>< Engine1_, s1_, Engine2_, s2_ > & rhs);</span>
<br>
<span>template &lt;typename Engine1_,</span>
<span>&nbsp;&nbsp;size_t s1_,</span>
<span>&nbsp;&nbsp;typename Engine2_,</span>
<span>&nbsp;&nbsp;size_t s2_,</span>
<span>&nbsp;&nbsp;typename CharT,</span>
<span>&nbsp;&nbsp;typename Traits&gt;</span>
<span>std::basic_ostream< CharT, Traits > & </span><span><b><a href="{{ site.baseurl }}/api/groups/group__random__number__engine__adaptors.html#function-operator<<">operator&lt;&lt;</a></b>(std::basic_ostream< CharT, Traits > & os,</span>
<span>&nbsp;&nbsp;const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1xor__combine__engine.html">xor_combine_engine</a>< Engine1_, s1_, Engine2_, s2_ > & e);</span>
<br>
<span>template &lt;typename Engine1_,</span>
<span>&nbsp;&nbsp;size_t s1_,</span>
<span>&nbsp;&nbsp;typename Engine2_,</span>
<span>&nbsp;&nbsp;size_t s2_,</span>
<span>&nbsp;&nbsp;typename CharT,</span>
<span>&nbsp;&nbsp;typename Traits&gt;</span>
<span>std::basic_istream< CharT, Traits > & </span><span><b><a href="{{ site.baseurl }}/api/groups/group__random__number__engine__adaptors.html#function-operator>>">operator&gt;&gt;</a></b>(std::basic_istream< CharT, Traits > & is,</span>
<span>&nbsp;&nbsp;<a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1xor__combine__engine.html">xor_combine_engine</a>< Engine1_, s1_, Engine2_, s2_ > & e);</span>
<span>} /* namespace thrust::random */</span>
</code>

## Member Classes

<h3 id="class-thrustrandomdiscard-block-engine">
<a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1discard__block__engine.html">Class <code>thrust::random::discard&#95;block&#95;engine</code>
</a>
</h3>

A <code>discard&#95;block&#95;engine</code> adapts an existing base random number engine and produces random values by discarding some of the values returned by its base engine. Each cycle of the compound engine begins by returning <code>r</code> values successively produced by the base engine and ends by discarding <code>p-r</code> such values. The engine's state is the state of its base engine followed by the number of calls to <code>operator()</code> that have occurred since the beginning of the current cycle. 

<h3 id="class-thrustrandomlinear-congruential-engine">
<a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1linear__congruential__engine.html">Class <code>thrust::random::linear&#95;congruential&#95;engine</code>
</a>
</h3>

A <code>linear&#95;congruential&#95;engine</code> random number engine produces unsigned integer random numbers using a linear congruential random number generation algorithm. 

<h3 id="class-thrustrandomlinear-feedback-shift-engine">
<a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1linear__feedback__shift__engine.html">Class <code>thrust::random::linear&#95;feedback&#95;shift&#95;engine</code>
</a>
</h3>

A <code>linear&#95;feedback&#95;shift&#95;engine</code> random number engine produces unsigned integer random values using a linear feedback shift random number generation algorithm. 

<h3 id="class-thrustrandomnormal-distribution">
<a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1normal__distribution.html">Class <code>thrust::random::normal&#95;distribution</code>
</a>
</h3>

A <code>normal&#95;distribution</code> random number distribution produces floating point Normally distributed random numbers. 

**Inherits From**:
`detail::normal_distribution_base::type`

<h3 id="class-thrustrandomsubtract-with-carry-engine">
<a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1subtract__with__carry__engine.html">Class <code>thrust::random::subtract&#95;with&#95;carry&#95;engine</code>
</a>
</h3>

A <code>subtract&#95;with&#95;carry&#95;engine</code> random number engine produces unsigned integer random numbers using the subtract with carry algorithm of Marsaglia & Zaman. 

<h3 id="class-thrustrandomuniform-int-distribution">
<a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1uniform__int__distribution.html">Class <code>thrust::random::uniform&#95;int&#95;distribution</code>
</a>
</h3>

A <code>uniform&#95;int&#95;distribution</code> random number distribution produces signed or unsigned integer uniform random numbers from a given range. 

<h3 id="class-thrustrandomuniform-real-distribution">
<a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1uniform__real__distribution.html">Class <code>thrust::random::uniform&#95;real&#95;distribution</code>
</a>
</h3>

A <code>uniform&#95;real&#95;distribution</code> random number distribution produces floating point uniform random numbers from a half-open interval. 

<h3 id="class-thrustrandomxor-combine-engine">
<a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1xor__combine__engine.html">Class <code>thrust::random::xor&#95;combine&#95;engine</code>
</a>
</h3>

An <code>xor&#95;combine&#95;engine</code> adapts two existing base random number engines and produces random values by combining the values produced by each. 


## Types

<h3 id="typedef-ranlux24">
Typedef <code>thrust::random::ranlux24</code>
</h3>

<code class="doxybook">
<span>typedef <a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1discard__block__engine.html">discard_block_engine</a>< <a href="{{ site.baseurl }}/api/groups/group__predefined__random.html#typedef-ranlux24-base">ranlux24_base</a>, 223, 23 ><b>ranlux24</b>;</span></code>
A random number engine with predefined parameters which implements the RANLUX level-3 random number generation algorithm. 

**Note**:
The 10000th consecutive invocation of a default-constructed object of type <code>ranlux24</code> shall produce the value <code>9901578</code> . 

<h3 id="typedef-ranlux48">
Typedef <code>thrust::random::ranlux48</code>
</h3>

<code class="doxybook">
<span>typedef <a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1discard__block__engine.html">discard_block_engine</a>< <a href="{{ site.baseurl }}/api/groups/group__predefined__random.html#typedef-ranlux48-base">ranlux48_base</a>, 389, 11 ><b>ranlux48</b>;</span></code>
A random number engine with predefined parameters which implements the RANLUX level-4 random number generation algorithm. 

**Note**:
The 10000th consecutive invocation of a default-constructed object of type <code>ranlux48</code> shall produce the value <code>88229545517833</code> . 

<h3 id="typedef-taus88">
Typedef <code>thrust::random::taus88</code>
</h3>

<code class="doxybook">
<span>typedef <a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1xor__combine__engine.html">xor_combine_engine</a>< <a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1linear__feedback__shift__engine.html">linear_feedback_shift_engine</a>< thrust::detail::uint32_t, 32u, 31u, 13u, 12u >, 0, <a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1xor__combine__engine.html">xor_combine_engine</a>< <a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1linear__feedback__shift__engine.html">linear_feedback_shift_engine</a>< thrust::detail::uint32_t, 32u, 29u, 2u, 4u >, 0, <a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1linear__feedback__shift__engine.html">linear_feedback_shift_engine</a>< thrust::detail::uint32_t, 32u, 28u, 3u, 17u >, 0 >, 0 ><b>taus88</b>;</span></code>
A random number engine with predefined parameters which implements L'Ecuyer's 1996 three-component Tausworthe random number generator. 

**Note**:
The 10000th consecutive invocation of a default-constructed object of type <code>taus88</code> shall produce the value <code>3535848941</code> . 

<h3 id="typedef-default-random-engine">
Typedef <code>thrust::random::default&#95;random&#95;engine</code>
</h3>

<code class="doxybook">
<span>typedef <a href="{{ site.baseurl }}/api/groups/group__predefined__random.html#typedef-minstd-rand">minstd_rand</a><b>default_random_engine</b>;</span></code>
An implementation-defined "default" random number engine. 

**Note**:
<code>default&#95;random&#95;engine</code> is currently an alias for <code>minstd&#95;rand</code>, and may change in a future version. 

<h3 id="typedef-minstd-rand0">
Typedef <code>thrust::random::minstd&#95;rand0</code>
</h3>

<code class="doxybook">
<span>typedef <a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1linear__congruential__engine.html">linear_congruential_engine</a>< thrust::detail::uint32_t, 16807, 0, 2147483647 ><b>minstd_rand0</b>;</span></code>
A random number engine with predefined parameters which implements a version of the Minimal Standard random number generation algorithm. 

**Note**:
The 10000th consecutive invocation of a default-constructed object of type <code>minstd&#95;rand0</code> shall produce the value <code>1043618065</code> . 

<h3 id="typedef-minstd-rand">
Typedef <code>thrust::random::minstd&#95;rand</code>
</h3>

<code class="doxybook">
<span>typedef <a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1linear__congruential__engine.html">linear_congruential_engine</a>< thrust::detail::uint32_t, 48271, 0, 2147483647 ><b>minstd_rand</b>;</span></code>
A random number engine with predefined parameters which implements a version of the Minimal Standard random number generation algorithm. 

**Note**:
The 10000th consecutive invocation of a default-constructed object of type <code>minstd&#95;rand</code> shall produce the value <code>399268537</code> . 

<h3 id="typedef-ranlux24-base">
Typedef <code>thrust::random::ranlux24&#95;base</code>
</h3>

<code class="doxybook">
<span>typedef <a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1subtract__with__carry__engine.html">subtract_with_carry_engine</a>< thrust::detail::uint32_t, 24, 10, 24 ><b>ranlux24_base</b>;</span></code>
A random number engine with predefined parameters which implements the base engine of the <code>ranlux24</code> random number engine. 

**Note**:
The 10000th consecutive invocation of a default-constructed object of type <code>ranlux24&#95;base</code> shall produce the value <code>7937952</code> . 

<h3 id="typedef-ranlux48-base">
Typedef <code>thrust::random::ranlux48&#95;base</code>
</h3>

<code class="doxybook">
<span>typedef <a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1subtract__with__carry__engine.html">subtract_with_carry_engine</a>< thrust::detail::uint64_t, 48, 5, 12 ><b>ranlux48_base</b>;</span></code>
A random number engine with predefined parameters which implements the base engine of the <code>ranlux48</code> random number engine. 

**Note**:
The 10000th consecutive invocation of a default-constructed object of type <code>ranlux48&#95;base</code> shall produce the value <code>192113843633948</code> . 


## Functions

<h3 id="function-operator==">
Function <code>thrust::random::operator==</code>
</h3>

<code class="doxybook">
<span>template &lt;typename Engine,</span>
<span>&nbsp;&nbsp;size_t p,</span>
<span>&nbsp;&nbsp;size_t r&gt;</span>
<span>__host__ __device__ bool </span><span><b>operator==</b>(const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1discard__block__engine.html">discard_block_engine</a>< Engine, p, r > & lhs,</span>
<span>&nbsp;&nbsp;const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1discard__block__engine.html">discard_block_engine</a>< Engine, p, r > & rhs);</span></code>
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
<span>__host__ __device__ bool </span><span><b>operator!=</b>(const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1discard__block__engine.html">discard_block_engine</a>< Engine, p, r > & lhs,</span>
<span>&nbsp;&nbsp;const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1discard__block__engine.html">discard_block_engine</a>< Engine, p, r > & rhs);</span></code>
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
<span>&nbsp;&nbsp;const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1discard__block__engine.html">discard_block_engine</a>< Engine, p, r > & e);</span></code>
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
<span>&nbsp;&nbsp;<a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1discard__block__engine.html">discard_block_engine</a>< Engine, p, r > & e);</span></code>
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
<span>template &lt;typename UIntType_,</span>
<span>&nbsp;&nbsp;UIntType_ a_,</span>
<span>&nbsp;&nbsp;UIntType_ c_,</span>
<span>&nbsp;&nbsp;UIntType_ m_&gt;</span>
<span>__host__ __device__ bool </span><span><b>operator==</b>(const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1linear__congruential__engine.html">linear_congruential_engine</a>< UIntType_, a_, c_, m_ > & lhs,</span>
<span>&nbsp;&nbsp;const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1linear__congruential__engine.html">linear_congruential_engine</a>< UIntType_, a_, c_, m_ > & rhs);</span></code>
This function checks two <code>linear&#95;congruential&#95;engines</code> for equality. 

**Function Parameters**:
* **`lhs`** The first <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1linear__congruential__engine.html">linear&#95;congruential&#95;engine</a></code> to test. 
* **`rhs`** The second <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1linear__congruential__engine.html">linear&#95;congruential&#95;engine</a></code> to test. 

**Returns**:
<code>true</code> if <code>lhs</code> is equal to <code>rhs</code>; <code>false</code>, otherwise. 

<h3 id="function-operator!=">
Function <code>thrust::random::operator!=</code>
</h3>

<code class="doxybook">
<span>template &lt;typename UIntType_,</span>
<span>&nbsp;&nbsp;UIntType_ a_,</span>
<span>&nbsp;&nbsp;UIntType_ c_,</span>
<span>&nbsp;&nbsp;UIntType_ m_&gt;</span>
<span>__host__ __device__ bool </span><span><b>operator!=</b>(const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1linear__congruential__engine.html">linear_congruential_engine</a>< UIntType_, a_, c_, m_ > & lhs,</span>
<span>&nbsp;&nbsp;const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1linear__congruential__engine.html">linear_congruential_engine</a>< UIntType_, a_, c_, m_ > & rhs);</span></code>
This function checks two <code>linear&#95;congruential&#95;engines</code> for inequality. 

**Function Parameters**:
* **`lhs`** The first <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1linear__congruential__engine.html">linear&#95;congruential&#95;engine</a></code> to test. 
* **`rhs`** The second <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1linear__congruential__engine.html">linear&#95;congruential&#95;engine</a></code> to test. 

**Returns**:
<code>true</code> if <code>lhs</code> is not equal to <code>rhs</code>; <code>false</code>, otherwise. 

<h3 id="function-operator<<">
Function <code>thrust::random::operator&lt;&lt;</code>
</h3>

<code class="doxybook">
<span>template &lt;typename UIntType_,</span>
<span>&nbsp;&nbsp;UIntType_ a_,</span>
<span>&nbsp;&nbsp;UIntType_ c_,</span>
<span>&nbsp;&nbsp;UIntType_ m_,</span>
<span>&nbsp;&nbsp;typename CharT,</span>
<span>&nbsp;&nbsp;typename Traits&gt;</span>
<span>std::basic_ostream< CharT, Traits > & </span><span><b>operator<<</b>(std::basic_ostream< CharT, Traits > & os,</span>
<span>&nbsp;&nbsp;const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1linear__congruential__engine.html">linear_congruential_engine</a>< UIntType_, a_, c_, m_ > & e);</span></code>
This function streams a <a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1linear__congruential__engine.html">linear_congruential_engine</a> to a <code>std::basic&#95;ostream</code>. 

**Function Parameters**:
* **`os`** The <code>basic&#95;ostream</code> to stream out to. 
* **`e`** The <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1linear__congruential__engine.html">linear&#95;congruential&#95;engine</a></code> to stream out. 

**Returns**:
<code>os</code>

<h3 id="function-operator>>">
Function <code>thrust::random::operator&gt;&gt;</code>
</h3>

<code class="doxybook">
<span>template &lt;typename UIntType_,</span>
<span>&nbsp;&nbsp;UIntType_ a_,</span>
<span>&nbsp;&nbsp;UIntType_ c_,</span>
<span>&nbsp;&nbsp;UIntType_ m_,</span>
<span>&nbsp;&nbsp;typename CharT,</span>
<span>&nbsp;&nbsp;typename Traits&gt;</span>
<span>std::basic_istream< CharT, Traits > & </span><span><b>operator>></b>(std::basic_istream< CharT, Traits > & is,</span>
<span>&nbsp;&nbsp;<a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1linear__congruential__engine.html">linear_congruential_engine</a>< UIntType_, a_, c_, m_ > & e);</span></code>
This function streams a <a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1linear__congruential__engine.html">linear_congruential_engine</a> in from a std::basic_istream. 

**Function Parameters**:
* **`is`** The <code>basic&#95;istream</code> to stream from. 
* **`e`** The <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1linear__congruential__engine.html">linear&#95;congruential&#95;engine</a></code> to stream in. 

**Returns**:
<code>is</code>

<h3 id="function-operator==">
Function <code>thrust::random::operator==</code>
</h3>

<code class="doxybook">
<span>template &lt;typename UIntType_,</span>
<span>&nbsp;&nbsp;size_t w_,</span>
<span>&nbsp;&nbsp;size_t k_,</span>
<span>&nbsp;&nbsp;size_t q_,</span>
<span>&nbsp;&nbsp;size_t s_&gt;</span>
<span>__host__ __device__ bool </span><span><b>operator==</b>(const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1linear__feedback__shift__engine.html">linear_feedback_shift_engine</a>< UIntType_, w_, k_, q_, s_ > & lhs,</span>
<span>&nbsp;&nbsp;const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1linear__feedback__shift__engine.html">linear_feedback_shift_engine</a>< UIntType_, w_, k_, q_, s_ > & rhs);</span></code>
This function checks two <code>linear&#95;feedback&#95;shift&#95;engines</code> for equality. 

**Function Parameters**:
* **`lhs`** The first <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1linear__feedback__shift__engine.html">linear&#95;feedback&#95;shift&#95;engine</a></code> to test. 
* **`rhs`** The second <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1linear__feedback__shift__engine.html">linear&#95;feedback&#95;shift&#95;engine</a></code> to test. 

**Returns**:
<code>true</code> if <code>lhs</code> is equal to <code>rhs</code>; <code>false</code>, otherwise. 

<h3 id="function-operator!=">
Function <code>thrust::random::operator!=</code>
</h3>

<code class="doxybook">
<span>template &lt;typename UIntType_,</span>
<span>&nbsp;&nbsp;size_t w_,</span>
<span>&nbsp;&nbsp;size_t k_,</span>
<span>&nbsp;&nbsp;size_t q_,</span>
<span>&nbsp;&nbsp;size_t s_&gt;</span>
<span>__host__ __device__ bool </span><span><b>operator!=</b>(const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1linear__feedback__shift__engine.html">linear_feedback_shift_engine</a>< UIntType_, w_, k_, q_, s_ > & lhs,</span>
<span>&nbsp;&nbsp;const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1linear__feedback__shift__engine.html">linear_feedback_shift_engine</a>< UIntType_, w_, k_, q_, s_ > & rhs);</span></code>
This function checks two <code>linear&#95;feedback&#95;shift&#95;engines</code> for inequality. 

**Function Parameters**:
* **`lhs`** The first <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1linear__feedback__shift__engine.html">linear&#95;feedback&#95;shift&#95;engine</a></code> to test. 
* **`rhs`** The second <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1linear__feedback__shift__engine.html">linear&#95;feedback&#95;shift&#95;engine</a></code> to test. 

**Returns**:
<code>true</code> if <code>lhs</code> is not equal to <code>rhs</code>; <code>false</code>, otherwise. 

<h3 id="function-operator<<">
Function <code>thrust::random::operator&lt;&lt;</code>
</h3>

<code class="doxybook">
<span>template &lt;typename UIntType_,</span>
<span>&nbsp;&nbsp;size_t w_,</span>
<span>&nbsp;&nbsp;size_t k_,</span>
<span>&nbsp;&nbsp;size_t q_,</span>
<span>&nbsp;&nbsp;size_t s_,</span>
<span>&nbsp;&nbsp;typename CharT,</span>
<span>&nbsp;&nbsp;typename Traits&gt;</span>
<span>std::basic_ostream< CharT, Traits > & </span><span><b>operator<<</b>(std::basic_ostream< CharT, Traits > & os,</span>
<span>&nbsp;&nbsp;const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1linear__feedback__shift__engine.html">linear_feedback_shift_engine</a>< UIntType_, w_, k_, q_, s_ > & e);</span></code>
This function streams a <a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1linear__feedback__shift__engine.html">linear_feedback_shift_engine</a> to a <code>std::basic&#95;ostream</code>. 

**Function Parameters**:
* **`os`** The <code>basic&#95;ostream</code> to stream out to. 
* **`e`** The <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1linear__feedback__shift__engine.html">linear&#95;feedback&#95;shift&#95;engine</a></code> to stream out. 

**Returns**:
<code>os</code>

<h3 id="function-operator>>">
Function <code>thrust::random::operator&gt;&gt;</code>
</h3>

<code class="doxybook">
<span>template &lt;typename UIntType_,</span>
<span>&nbsp;&nbsp;size_t w_,</span>
<span>&nbsp;&nbsp;size_t k_,</span>
<span>&nbsp;&nbsp;size_t q_,</span>
<span>&nbsp;&nbsp;size_t s_,</span>
<span>&nbsp;&nbsp;typename CharT,</span>
<span>&nbsp;&nbsp;typename Traits&gt;</span>
<span>std::basic_istream< CharT, Traits > & </span><span><b>operator>></b>(std::basic_istream< CharT, Traits > & is,</span>
<span>&nbsp;&nbsp;<a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1linear__feedback__shift__engine.html">linear_feedback_shift_engine</a>< UIntType_, w_, k_, q_, s_ > & e);</span></code>
This function streams a <a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1linear__feedback__shift__engine.html">linear_feedback_shift_engine</a> in from a std::basic_istream. 

**Function Parameters**:
* **`is`** The <code>basic&#95;istream</code> to stream from. 
* **`e`** The <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1linear__feedback__shift__engine.html">linear&#95;feedback&#95;shift&#95;engine</a></code> to stream in. 

**Returns**:
<code>is</code>

<h3 id="function-operator==">
Function <code>thrust::random::operator==</code>
</h3>

<code class="doxybook">
<span>template &lt;typename RealType&gt;</span>
<span>__host__ __device__ bool </span><span><b>operator==</b>(const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1normal__distribution.html">normal_distribution</a>< RealType > & lhs,</span>
<span>&nbsp;&nbsp;const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1normal__distribution.html">normal_distribution</a>< RealType > & rhs);</span></code>
This function checks two <code>normal&#95;distributions</code> for equality. 

**Function Parameters**:
* **`lhs`** The first <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1normal__distribution.html">normal&#95;distribution</a></code> to test. 
* **`rhs`** The second <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1normal__distribution.html">normal&#95;distribution</a></code> to test. 

**Returns**:
<code>true</code> if <code>lhs</code> is equal to <code>rhs</code>; <code>false</code>, otherwise. 

<h3 id="function-operator!=">
Function <code>thrust::random::operator!=</code>
</h3>

<code class="doxybook">
<span>template &lt;typename RealType&gt;</span>
<span>__host__ __device__ bool </span><span><b>operator!=</b>(const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1normal__distribution.html">normal_distribution</a>< RealType > & lhs,</span>
<span>&nbsp;&nbsp;const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1normal__distribution.html">normal_distribution</a>< RealType > & rhs);</span></code>
This function checks two <code>normal&#95;distributions</code> for inequality. 

**Function Parameters**:
* **`lhs`** The first <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1normal__distribution.html">normal&#95;distribution</a></code> to test. 
* **`rhs`** The second <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1normal__distribution.html">normal&#95;distribution</a></code> to test. 

**Returns**:
<code>true</code> if <code>lhs</code> is not equal to <code>rhs</code>; <code>false</code>, otherwise. 

<h3 id="function-operator<<">
Function <code>thrust::random::operator&lt;&lt;</code>
</h3>

<code class="doxybook">
<span>template &lt;typename RealType,</span>
<span>&nbsp;&nbsp;typename CharT,</span>
<span>&nbsp;&nbsp;typename Traits&gt;</span>
<span>std::basic_ostream< CharT, Traits > & </span><span><b>operator<<</b>(std::basic_ostream< CharT, Traits > & os,</span>
<span>&nbsp;&nbsp;const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1normal__distribution.html">normal_distribution</a>< RealType > & d);</span></code>
This function streams a <a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1normal__distribution.html">normal_distribution</a> to a <code>std::basic&#95;ostream</code>. 

**Function Parameters**:
* **`os`** The <code>basic&#95;ostream</code> to stream out to. 
* **`d`** The <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1normal__distribution.html">normal&#95;distribution</a></code> to stream out. 

**Returns**:
<code>os</code>

<h3 id="function-operator>>">
Function <code>thrust::random::operator&gt;&gt;</code>
</h3>

<code class="doxybook">
<span>template &lt;typename RealType,</span>
<span>&nbsp;&nbsp;typename CharT,</span>
<span>&nbsp;&nbsp;typename Traits&gt;</span>
<span>std::basic_istream< CharT, Traits > & </span><span><b>operator>></b>(std::basic_istream< CharT, Traits > & is,</span>
<span>&nbsp;&nbsp;<a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1normal__distribution.html">normal_distribution</a>< RealType > & d);</span></code>
This function streams a <a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1normal__distribution.html">normal_distribution</a> in from a std::basic_istream. 

**Function Parameters**:
* **`is`** The <code>basic&#95;istream</code> to stream from. 
* **`d`** The <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1normal__distribution.html">normal&#95;distribution</a></code> to stream in. 

**Returns**:
<code>is</code>

<h3 id="function-operator==">
Function <code>thrust::random::operator==</code>
</h3>

<code class="doxybook">
<span>template &lt;typename UIntType_,</span>
<span>&nbsp;&nbsp;size_t w_,</span>
<span>&nbsp;&nbsp;size_t s_,</span>
<span>&nbsp;&nbsp;size_t r_&gt;</span>
<span>__host__ __device__ bool </span><span><b>operator==</b>(const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1subtract__with__carry__engine.html">subtract_with_carry_engine</a>< UIntType_, w_, s_, r_ > & lhs,</span>
<span>&nbsp;&nbsp;const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1subtract__with__carry__engine.html">subtract_with_carry_engine</a>< UIntType_, w_, s_, r_ > & rhs);</span></code>
This function checks two <code>subtract&#95;with&#95;carry&#95;engines</code> for equality. 

**Function Parameters**:
* **`lhs`** The first <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1subtract__with__carry__engine.html">subtract&#95;with&#95;carry&#95;engine</a></code> to test. 
* **`rhs`** The second <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1subtract__with__carry__engine.html">subtract&#95;with&#95;carry&#95;engine</a></code> to test. 

**Returns**:
<code>true</code> if <code>lhs</code> is equal to <code>rhs</code>; <code>false</code>, otherwise. 

<h3 id="function-operator!=">
Function <code>thrust::random::operator!=</code>
</h3>

<code class="doxybook">
<span>template &lt;typename UIntType_,</span>
<span>&nbsp;&nbsp;size_t w_,</span>
<span>&nbsp;&nbsp;size_t s_,</span>
<span>&nbsp;&nbsp;size_t r_&gt;</span>
<span>__host__ __device__ bool </span><span><b>operator!=</b>(const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1subtract__with__carry__engine.html">subtract_with_carry_engine</a>< UIntType_, w_, s_, r_ > & lhs,</span>
<span>&nbsp;&nbsp;const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1subtract__with__carry__engine.html">subtract_with_carry_engine</a>< UIntType_, w_, s_, r_ > & rhs);</span></code>
This function checks two <code>subtract&#95;with&#95;carry&#95;engines</code> for inequality. 

**Function Parameters**:
* **`lhs`** The first <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1subtract__with__carry__engine.html">subtract&#95;with&#95;carry&#95;engine</a></code> to test. 
* **`rhs`** The second <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1subtract__with__carry__engine.html">subtract&#95;with&#95;carry&#95;engine</a></code> to test. 

**Returns**:
<code>true</code> if <code>lhs</code> is not equal to <code>rhs</code>; <code>false</code>, otherwise. 

<h3 id="function-operator<<">
Function <code>thrust::random::operator&lt;&lt;</code>
</h3>

<code class="doxybook">
<span>template &lt;typename UIntType_,</span>
<span>&nbsp;&nbsp;size_t w_,</span>
<span>&nbsp;&nbsp;size_t s_,</span>
<span>&nbsp;&nbsp;size_t r_,</span>
<span>&nbsp;&nbsp;typename CharT,</span>
<span>&nbsp;&nbsp;typename Traits&gt;</span>
<span>std::basic_ostream< CharT, Traits > & </span><span><b>operator<<</b>(std::basic_ostream< CharT, Traits > & os,</span>
<span>&nbsp;&nbsp;const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1subtract__with__carry__engine.html">subtract_with_carry_engine</a>< UIntType_, w_, s_, r_ > & e);</span></code>
This function streams a <a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1subtract__with__carry__engine.html">subtract_with_carry_engine</a> to a <code>std::basic&#95;ostream</code>. 

**Function Parameters**:
* **`os`** The <code>basic&#95;ostream</code> to stream out to. 
* **`e`** The <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1subtract__with__carry__engine.html">subtract&#95;with&#95;carry&#95;engine</a></code> to stream out. 

**Returns**:
<code>os</code>

<h3 id="function-operator>>">
Function <code>thrust::random::operator&gt;&gt;</code>
</h3>

<code class="doxybook">
<span>template &lt;typename UIntType_,</span>
<span>&nbsp;&nbsp;size_t w_,</span>
<span>&nbsp;&nbsp;size_t s_,</span>
<span>&nbsp;&nbsp;size_t r_,</span>
<span>&nbsp;&nbsp;typename CharT,</span>
<span>&nbsp;&nbsp;typename Traits&gt;</span>
<span>std::basic_istream< CharT, Traits > & </span><span><b>operator>></b>(std::basic_istream< CharT, Traits > & is,</span>
<span>&nbsp;&nbsp;<a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1subtract__with__carry__engine.html">subtract_with_carry_engine</a>< UIntType_, w_, s_, r_ > & e);</span></code>
This function streams a <a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1subtract__with__carry__engine.html">subtract_with_carry_engine</a> in from a std::basic_istream. 

**Function Parameters**:
* **`is`** The <code>basic&#95;istream</code> to stream from. 
* **`e`** The <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1subtract__with__carry__engine.html">subtract&#95;with&#95;carry&#95;engine</a></code> to stream in. 

**Returns**:
<code>is</code>

<h3 id="function-operator==">
Function <code>thrust::random::operator==</code>
</h3>

<code class="doxybook">
<span>template &lt;typename IntType&gt;</span>
<span>__host__ __device__ bool </span><span><b>operator==</b>(const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1uniform__int__distribution.html">uniform_int_distribution</a>< IntType > & lhs,</span>
<span>&nbsp;&nbsp;const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1uniform__int__distribution.html">uniform_int_distribution</a>< IntType > & rhs);</span></code>
This function checks two <code>uniform&#95;int&#95;distributions</code> for equality. 

**Function Parameters**:
* **`lhs`** The first <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1uniform__int__distribution.html">uniform&#95;int&#95;distribution</a></code> to test. 
* **`rhs`** The second <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1uniform__int__distribution.html">uniform&#95;int&#95;distribution</a></code> to test. 

**Returns**:
<code>true</code> if <code>lhs</code> is equal to <code>rhs</code>; <code>false</code>, otherwise. 

<h3 id="function-operator!=">
Function <code>thrust::random::operator!=</code>
</h3>

<code class="doxybook">
<span>template &lt;typename IntType&gt;</span>
<span>__host__ __device__ bool </span><span><b>operator!=</b>(const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1uniform__int__distribution.html">uniform_int_distribution</a>< IntType > & lhs,</span>
<span>&nbsp;&nbsp;const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1uniform__int__distribution.html">uniform_int_distribution</a>< IntType > & rhs);</span></code>
This function checks two <code>uniform&#95;int&#95;distributions</code> for inequality. 

**Function Parameters**:
* **`lhs`** The first <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1uniform__int__distribution.html">uniform&#95;int&#95;distribution</a></code> to test. 
* **`rhs`** The second <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1uniform__int__distribution.html">uniform&#95;int&#95;distribution</a></code> to test. 

**Returns**:
<code>true</code> if <code>lhs</code> is not equal to <code>rhs</code>; <code>false</code>, otherwise. 

<h3 id="function-operator<<">
Function <code>thrust::random::operator&lt;&lt;</code>
</h3>

<code class="doxybook">
<span>template &lt;typename IntType,</span>
<span>&nbsp;&nbsp;typename CharT,</span>
<span>&nbsp;&nbsp;typename Traits&gt;</span>
<span>std::basic_ostream< CharT, Traits > & </span><span><b>operator<<</b>(std::basic_ostream< CharT, Traits > & os,</span>
<span>&nbsp;&nbsp;const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1uniform__int__distribution.html">uniform_int_distribution</a>< IntType > & d);</span></code>
This function streams a <a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1uniform__int__distribution.html">uniform_int_distribution</a> to a <code>std::basic&#95;ostream</code>. 

**Function Parameters**:
* **`os`** The <code>basic&#95;ostream</code> to stream out to. 
* **`d`** The <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1uniform__int__distribution.html">uniform&#95;int&#95;distribution</a></code> to stream out. 

**Returns**:
<code>os</code>

<h3 id="function-operator>>">
Function <code>thrust::random::operator&gt;&gt;</code>
</h3>

<code class="doxybook">
<span>template &lt;typename IntType,</span>
<span>&nbsp;&nbsp;typename CharT,</span>
<span>&nbsp;&nbsp;typename Traits&gt;</span>
<span>std::basic_istream< CharT, Traits > & </span><span><b>operator>></b>(std::basic_istream< CharT, Traits > & is,</span>
<span>&nbsp;&nbsp;<a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1uniform__int__distribution.html">uniform_int_distribution</a>< IntType > & d);</span></code>
This function streams a <a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1uniform__int__distribution.html">uniform_int_distribution</a> in from a std::basic_istream. 

**Function Parameters**:
* **`is`** The <code>basic&#95;istream</code> to stream from. 
* **`d`** The <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1uniform__int__distribution.html">uniform&#95;int&#95;distribution</a></code> to stream in. 

**Returns**:
<code>is</code>

<h3 id="function-operator==">
Function <code>thrust::random::operator==</code>
</h3>

<code class="doxybook">
<span>template &lt;typename RealType&gt;</span>
<span>__host__ __device__ bool </span><span><b>operator==</b>(const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1uniform__real__distribution.html">uniform_real_distribution</a>< RealType > & lhs,</span>
<span>&nbsp;&nbsp;const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1uniform__real__distribution.html">uniform_real_distribution</a>< RealType > & rhs);</span></code>
This function checks two <code>uniform&#95;real&#95;distributions</code> for equality. 

**Function Parameters**:
* **`lhs`** The first <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1uniform__real__distribution.html">uniform&#95;real&#95;distribution</a></code> to test. 
* **`rhs`** The second <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1uniform__real__distribution.html">uniform&#95;real&#95;distribution</a></code> to test. 

**Returns**:
<code>true</code> if <code>lhs</code> is equal to <code>rhs</code>; <code>false</code>, otherwise. 

<h3 id="function-operator!=">
Function <code>thrust::random::operator!=</code>
</h3>

<code class="doxybook">
<span>template &lt;typename RealType&gt;</span>
<span>__host__ __device__ bool </span><span><b>operator!=</b>(const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1uniform__real__distribution.html">uniform_real_distribution</a>< RealType > & lhs,</span>
<span>&nbsp;&nbsp;const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1uniform__real__distribution.html">uniform_real_distribution</a>< RealType > & rhs);</span></code>
This function checks two <code>uniform&#95;real&#95;distributions</code> for inequality. 

**Function Parameters**:
* **`lhs`** The first <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1uniform__real__distribution.html">uniform&#95;real&#95;distribution</a></code> to test. 
* **`rhs`** The second <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1uniform__real__distribution.html">uniform&#95;real&#95;distribution</a></code> to test. 

**Returns**:
<code>true</code> if <code>lhs</code> is not equal to <code>rhs</code>; <code>false</code>, otherwise. 

<h3 id="function-operator<<">
Function <code>thrust::random::operator&lt;&lt;</code>
</h3>

<code class="doxybook">
<span>template &lt;typename RealType,</span>
<span>&nbsp;&nbsp;typename CharT,</span>
<span>&nbsp;&nbsp;typename Traits&gt;</span>
<span>std::basic_ostream< CharT, Traits > & </span><span><b>operator<<</b>(std::basic_ostream< CharT, Traits > & os,</span>
<span>&nbsp;&nbsp;const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1uniform__real__distribution.html">uniform_real_distribution</a>< RealType > & d);</span></code>
This function streams a <a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1uniform__real__distribution.html">uniform_real_distribution</a> to a <code>std::basic&#95;ostream</code>. 

**Function Parameters**:
* **`os`** The <code>basic&#95;ostream</code> to stream out to. 
* **`d`** The <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1uniform__real__distribution.html">uniform&#95;real&#95;distribution</a></code> to stream out. 

**Returns**:
<code>os</code>

<h3 id="function-operator>>">
Function <code>thrust::random::operator&gt;&gt;</code>
</h3>

<code class="doxybook">
<span>template &lt;typename RealType,</span>
<span>&nbsp;&nbsp;typename CharT,</span>
<span>&nbsp;&nbsp;typename Traits&gt;</span>
<span>std::basic_istream< CharT, Traits > & </span><span><b>operator>></b>(std::basic_istream< CharT, Traits > & is,</span>
<span>&nbsp;&nbsp;<a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1uniform__real__distribution.html">uniform_real_distribution</a>< RealType > & d);</span></code>
This function streams a <a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1uniform__real__distribution.html">uniform_real_distribution</a> in from a std::basic_istream. 

**Function Parameters**:
* **`is`** The <code>basic&#95;istream</code> to stream from. 
* **`d`** The <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1uniform__real__distribution.html">uniform&#95;real&#95;distribution</a></code> to stream in. 

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
<span>__host__ __device__ bool </span><span><b>operator==</b>(const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1xor__combine__engine.html">xor_combine_engine</a>< Engine1_, s1_, Engine2_, s2_ > & lhs,</span>
<span>&nbsp;&nbsp;const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1xor__combine__engine.html">xor_combine_engine</a>< Engine1_, s1_, Engine2_, s2_ > & rhs);</span></code>
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
<span>__host__ __device__ bool </span><span><b>operator!=</b>(const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1xor__combine__engine.html">xor_combine_engine</a>< Engine1_, s1_, Engine2_, s2_ > & lhs,</span>
<span>&nbsp;&nbsp;const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1xor__combine__engine.html">xor_combine_engine</a>< Engine1_, s1_, Engine2_, s2_ > & rhs);</span></code>
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
<span>&nbsp;&nbsp;const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1xor__combine__engine.html">xor_combine_engine</a>< Engine1_, s1_, Engine2_, s2_ > & e);</span></code>
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
<span>&nbsp;&nbsp;<a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1xor__combine__engine.html">xor_combine_engine</a>< Engine1_, s1_, Engine2_, s2_ > & e);</span></code>
This function streams a <a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1xor__combine__engine.html">xor_combine_engine</a> in from a std::basic_istream. 

**Function Parameters**:
* **`is`** The <code>basic&#95;istream</code> to stream from. 
* **`e`** The <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1xor__combine__engine.html">xor&#95;combine&#95;engine</a></code> to stream in. 

**Returns**:
<code>is</code>


