---
title: Random Number Engine Class Templates
parent: Random Number Generation
grand_parent: API
nav_exclude: false
has_children: true
has_toc: false
---

# Random Number Engine Class Templates

<code class="doxybook">
<span class="doxybook-comment">/* A <code>linear&#95;congruential&#95;engine</code> random number engine produces unsigned integer random numbers using a linear congruential random number generation algorithm.  */</span><span>template &lt;typename UIntType,</span>
<span>&nbsp;&nbsp;UIntType a,</span>
<span>&nbsp;&nbsp;UIntType c,</span>
<span>&nbsp;&nbsp;UIntType m&gt;</span>
<span>class <b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1linear__congruential__engine.html">thrust::random::linear&#95;congruential&#95;engine</a></b>;</span>
<br>
<span class="doxybook-comment">/* A <code>linear&#95;feedback&#95;shift&#95;engine</code> random number engine produces unsigned integer random values using a linear feedback shift random number generation algorithm.  */</span><span>template &lt;typename UIntType,</span>
<span>&nbsp;&nbsp;size_t w,</span>
<span>&nbsp;&nbsp;size_t k,</span>
<span>&nbsp;&nbsp;size_t q,</span>
<span>&nbsp;&nbsp;size_t s&gt;</span>
<span>class <b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1linear__feedback__shift__engine.html">thrust::random::linear&#95;feedback&#95;shift&#95;engine</a></b>;</span>
<br>
<span class="doxybook-comment">/* A <code>subtract&#95;with&#95;carry&#95;engine</code> random number engine produces unsigned integer random numbers using the subtract with carry algorithm of Marsaglia & Zaman.  */</span><span>template &lt;typename UIntType,</span>
<span>&nbsp;&nbsp;size_t w,</span>
<span>&nbsp;&nbsp;size_t s,</span>
<span>&nbsp;&nbsp;size_t r&gt;</span>
<span>class <b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1subtract__with__carry__engine.html">thrust::random::subtract&#95;with&#95;carry&#95;engine</a></b>;</span>
<br>
<span>template &lt;typename UIntType_,</span>
<span>&nbsp;&nbsp;UIntType_ a_,</span>
<span>&nbsp;&nbsp;UIntType_ c_,</span>
<span>&nbsp;&nbsp;UIntType_ m_&gt;</span>
<span>__host__ __device__ bool </span><span><b><a href="{{ site.baseurl }}/api/groups/group__random__number__engine__templates.html#function-operator==">thrust::random::operator==</a></b>(const linear_congruential_engine< UIntType_, a_, c_, m_ > & lhs,</span>
<span>&nbsp;&nbsp;const linear_congruential_engine< UIntType_, a_, c_, m_ > & rhs);</span>
<br>
<span>template &lt;typename UIntType_,</span>
<span>&nbsp;&nbsp;UIntType_ a_,</span>
<span>&nbsp;&nbsp;UIntType_ c_,</span>
<span>&nbsp;&nbsp;UIntType_ m_&gt;</span>
<span>__host__ __device__ bool </span><span><b><a href="{{ site.baseurl }}/api/groups/group__random__number__engine__templates.html#function-operator!=">thrust::random::operator!=</a></b>(const linear_congruential_engine< UIntType_, a_, c_, m_ > & lhs,</span>
<span>&nbsp;&nbsp;const linear_congruential_engine< UIntType_, a_, c_, m_ > & rhs);</span>
<br>
<span>template &lt;typename UIntType_,</span>
<span>&nbsp;&nbsp;UIntType_ a_,</span>
<span>&nbsp;&nbsp;UIntType_ c_,</span>
<span>&nbsp;&nbsp;UIntType_ m_,</span>
<span>&nbsp;&nbsp;typename CharT,</span>
<span>&nbsp;&nbsp;typename Traits&gt;</span>
<span>std::basic_ostream< CharT, Traits > & </span><span><b><a href="{{ site.baseurl }}/api/groups/group__random__number__engine__templates.html#function-operator<<">thrust::random::operator&lt;&lt;</a></b>(std::basic_ostream< CharT, Traits > & os,</span>
<span>&nbsp;&nbsp;const linear_congruential_engine< UIntType_, a_, c_, m_ > & e);</span>
<br>
<span>template &lt;typename UIntType_,</span>
<span>&nbsp;&nbsp;UIntType_ a_,</span>
<span>&nbsp;&nbsp;UIntType_ c_,</span>
<span>&nbsp;&nbsp;UIntType_ m_,</span>
<span>&nbsp;&nbsp;typename CharT,</span>
<span>&nbsp;&nbsp;typename Traits&gt;</span>
<span>std::basic_istream< CharT, Traits > & </span><span><b><a href="{{ site.baseurl }}/api/groups/group__random__number__engine__templates.html#function-operator>>">thrust::random::operator&gt;&gt;</a></b>(std::basic_istream< CharT, Traits > & is,</span>
<span>&nbsp;&nbsp;linear_congruential_engine< UIntType_, a_, c_, m_ > & e);</span>
<br>
<span>template &lt;typename UIntType_,</span>
<span>&nbsp;&nbsp;size_t w_,</span>
<span>&nbsp;&nbsp;size_t k_,</span>
<span>&nbsp;&nbsp;size_t q_,</span>
<span>&nbsp;&nbsp;size_t s_&gt;</span>
<span>__host__ __device__ bool </span><span><b><a href="{{ site.baseurl }}/api/groups/group__random__number__engine__templates.html#function-operator==">thrust::random::operator==</a></b>(const linear_feedback_shift_engine< UIntType_, w_, k_, q_, s_ > & lhs,</span>
<span>&nbsp;&nbsp;const linear_feedback_shift_engine< UIntType_, w_, k_, q_, s_ > & rhs);</span>
<br>
<span>template &lt;typename UIntType_,</span>
<span>&nbsp;&nbsp;size_t w_,</span>
<span>&nbsp;&nbsp;size_t k_,</span>
<span>&nbsp;&nbsp;size_t q_,</span>
<span>&nbsp;&nbsp;size_t s_&gt;</span>
<span>__host__ __device__ bool </span><span><b><a href="{{ site.baseurl }}/api/groups/group__random__number__engine__templates.html#function-operator!=">thrust::random::operator!=</a></b>(const linear_feedback_shift_engine< UIntType_, w_, k_, q_, s_ > & lhs,</span>
<span>&nbsp;&nbsp;const linear_feedback_shift_engine< UIntType_, w_, k_, q_, s_ > & rhs);</span>
<br>
<span>template &lt;typename UIntType_,</span>
<span>&nbsp;&nbsp;size_t w_,</span>
<span>&nbsp;&nbsp;size_t k_,</span>
<span>&nbsp;&nbsp;size_t q_,</span>
<span>&nbsp;&nbsp;size_t s_,</span>
<span>&nbsp;&nbsp;typename CharT,</span>
<span>&nbsp;&nbsp;typename Traits&gt;</span>
<span>std::basic_ostream< CharT, Traits > & </span><span><b><a href="{{ site.baseurl }}/api/groups/group__random__number__engine__templates.html#function-operator<<">thrust::random::operator&lt;&lt;</a></b>(std::basic_ostream< CharT, Traits > & os,</span>
<span>&nbsp;&nbsp;const linear_feedback_shift_engine< UIntType_, w_, k_, q_, s_ > & e);</span>
<br>
<span>template &lt;typename UIntType_,</span>
<span>&nbsp;&nbsp;size_t w_,</span>
<span>&nbsp;&nbsp;size_t k_,</span>
<span>&nbsp;&nbsp;size_t q_,</span>
<span>&nbsp;&nbsp;size_t s_,</span>
<span>&nbsp;&nbsp;typename CharT,</span>
<span>&nbsp;&nbsp;typename Traits&gt;</span>
<span>std::basic_istream< CharT, Traits > & </span><span><b><a href="{{ site.baseurl }}/api/groups/group__random__number__engine__templates.html#function-operator>>">thrust::random::operator&gt;&gt;</a></b>(std::basic_istream< CharT, Traits > & is,</span>
<span>&nbsp;&nbsp;linear_feedback_shift_engine< UIntType_, w_, k_, q_, s_ > & e);</span>
<br>
<span>template &lt;typename UIntType_,</span>
<span>&nbsp;&nbsp;size_t w_,</span>
<span>&nbsp;&nbsp;size_t s_,</span>
<span>&nbsp;&nbsp;size_t r_&gt;</span>
<span>__host__ __device__ bool </span><span><b><a href="{{ site.baseurl }}/api/groups/group__random__number__engine__templates.html#function-operator==">thrust::random::operator==</a></b>(const subtract_with_carry_engine< UIntType_, w_, s_, r_ > & lhs,</span>
<span>&nbsp;&nbsp;const subtract_with_carry_engine< UIntType_, w_, s_, r_ > & rhs);</span>
<br>
<span>template &lt;typename UIntType_,</span>
<span>&nbsp;&nbsp;size_t w_,</span>
<span>&nbsp;&nbsp;size_t s_,</span>
<span>&nbsp;&nbsp;size_t r_&gt;</span>
<span>__host__ __device__ bool </span><span><b><a href="{{ site.baseurl }}/api/groups/group__random__number__engine__templates.html#function-operator!=">thrust::random::operator!=</a></b>(const subtract_with_carry_engine< UIntType_, w_, s_, r_ > & lhs,</span>
<span>&nbsp;&nbsp;const subtract_with_carry_engine< UIntType_, w_, s_, r_ > & rhs);</span>
<br>
<span>template &lt;typename UIntType_,</span>
<span>&nbsp;&nbsp;size_t w_,</span>
<span>&nbsp;&nbsp;size_t s_,</span>
<span>&nbsp;&nbsp;size_t r_,</span>
<span>&nbsp;&nbsp;typename CharT,</span>
<span>&nbsp;&nbsp;typename Traits&gt;</span>
<span>std::basic_ostream< CharT, Traits > & </span><span><b><a href="{{ site.baseurl }}/api/groups/group__random__number__engine__templates.html#function-operator<<">thrust::random::operator&lt;&lt;</a></b>(std::basic_ostream< CharT, Traits > & os,</span>
<span>&nbsp;&nbsp;const subtract_with_carry_engine< UIntType_, w_, s_, r_ > & e);</span>
<br>
<span>template &lt;typename UIntType_,</span>
<span>&nbsp;&nbsp;size_t w_,</span>
<span>&nbsp;&nbsp;size_t s_,</span>
<span>&nbsp;&nbsp;size_t r_,</span>
<span>&nbsp;&nbsp;typename CharT,</span>
<span>&nbsp;&nbsp;typename Traits&gt;</span>
<span>std::basic_istream< CharT, Traits > & </span><span><b><a href="{{ site.baseurl }}/api/groups/group__random__number__engine__templates.html#function-operator>>">thrust::random::operator&gt;&gt;</a></b>(std::basic_istream< CharT, Traits > & is,</span>
<span>&nbsp;&nbsp;subtract_with_carry_engine< UIntType_, w_, s_, r_ > & e);</span>
</code>

## Member Classes

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

<h3 id="class-thrustrandomsubtract-with-carry-engine">
<a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1subtract__with__carry__engine.html">Class <code>thrust::random::subtract&#95;with&#95;carry&#95;engine</code>
</a>
</h3>

A <code>subtract&#95;with&#95;carry&#95;engine</code> random number engine produces unsigned integer random numbers using the subtract with carry algorithm of Marsaglia & Zaman. 


## Functions

<h3 id="function-operator==">
Function <code>thrust::random::operator==</code>
</h3>

<code class="doxybook">
<span>template &lt;typename UIntType_,</span>
<span>&nbsp;&nbsp;UIntType_ a_,</span>
<span>&nbsp;&nbsp;UIntType_ c_,</span>
<span>&nbsp;&nbsp;UIntType_ m_&gt;</span>
<span>__host__ __device__ bool </span><span><b>operator==</b>(const linear_congruential_engine< UIntType_, a_, c_, m_ > & lhs,</span>
<span>&nbsp;&nbsp;const linear_congruential_engine< UIntType_, a_, c_, m_ > & rhs);</span></code>
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
<span>__host__ __device__ bool </span><span><b>operator!=</b>(const linear_congruential_engine< UIntType_, a_, c_, m_ > & lhs,</span>
<span>&nbsp;&nbsp;const linear_congruential_engine< UIntType_, a_, c_, m_ > & rhs);</span></code>
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
<span>&nbsp;&nbsp;const linear_congruential_engine< UIntType_, a_, c_, m_ > & e);</span></code>
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
<span>&nbsp;&nbsp;linear_congruential_engine< UIntType_, a_, c_, m_ > & e);</span></code>
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
<span>__host__ __device__ bool </span><span><b>operator==</b>(const linear_feedback_shift_engine< UIntType_, w_, k_, q_, s_ > & lhs,</span>
<span>&nbsp;&nbsp;const linear_feedback_shift_engine< UIntType_, w_, k_, q_, s_ > & rhs);</span></code>
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
<span>__host__ __device__ bool </span><span><b>operator!=</b>(const linear_feedback_shift_engine< UIntType_, w_, k_, q_, s_ > & lhs,</span>
<span>&nbsp;&nbsp;const linear_feedback_shift_engine< UIntType_, w_, k_, q_, s_ > & rhs);</span></code>
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
<span>&nbsp;&nbsp;const linear_feedback_shift_engine< UIntType_, w_, k_, q_, s_ > & e);</span></code>
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
<span>&nbsp;&nbsp;linear_feedback_shift_engine< UIntType_, w_, k_, q_, s_ > & e);</span></code>
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
<span>template &lt;typename UIntType_,</span>
<span>&nbsp;&nbsp;size_t w_,</span>
<span>&nbsp;&nbsp;size_t s_,</span>
<span>&nbsp;&nbsp;size_t r_&gt;</span>
<span>__host__ __device__ bool </span><span><b>operator==</b>(const subtract_with_carry_engine< UIntType_, w_, s_, r_ > & lhs,</span>
<span>&nbsp;&nbsp;const subtract_with_carry_engine< UIntType_, w_, s_, r_ > & rhs);</span></code>
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
<span>__host__ __device__ bool </span><span><b>operator!=</b>(const subtract_with_carry_engine< UIntType_, w_, s_, r_ > & lhs,</span>
<span>&nbsp;&nbsp;const subtract_with_carry_engine< UIntType_, w_, s_, r_ > & rhs);</span></code>
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
<span>&nbsp;&nbsp;const subtract_with_carry_engine< UIntType_, w_, s_, r_ > & e);</span></code>
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
<span>&nbsp;&nbsp;subtract_with_carry_engine< UIntType_, w_, s_, r_ > & e);</span></code>
This function streams a <a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1subtract__with__carry__engine.html">subtract_with_carry_engine</a> in from a std::basic_istream. 

**Function Parameters**:
* **`is`** The <code>basic&#95;istream</code> to stream from. 
* **`e`** The <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1random_1_1subtract__with__carry__engine.html">subtract&#95;with&#95;carry&#95;engine</a></code> to stream in. 

**Returns**:
<code>is</code>


