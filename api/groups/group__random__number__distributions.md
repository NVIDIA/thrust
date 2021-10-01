---
title: Random Number Distributions Class Templates
parent: Random Number Generation
grand_parent: API
nav_exclude: false
has_children: true
has_toc: false
---

# Random Number Distributions Class Templates

<code class="doxybook">
<span class="doxybook-comment">/* A <code>normal&#95;distribution</code> random number distribution produces floating point Normally distributed random numbers.  */</span><span>template &lt;typename RealType = double&gt;</span>
<span>class <b><a href="/thrust/api/classes/classrandom_1_1normal__distribution.html">random::normal&#95;distribution</a></b>;</span>
<br>
<span class="doxybook-comment">/* A <code>uniform&#95;int&#95;distribution</code> random number distribution produces signed or unsigned integer uniform random numbers from a given range.  */</span><span>template &lt;typename IntType = int&gt;</span>
<span>class <b><a href="/thrust/api/classes/classrandom_1_1uniform__int__distribution.html">random::uniform&#95;int&#95;distribution</a></b>;</span>
<br>
<span class="doxybook-comment">/* A <code>uniform&#95;real&#95;distribution</code> random number distribution produces floating point uniform random numbers from a half-open interval.  */</span><span>template &lt;typename RealType = double&gt;</span>
<span>class <b><a href="/thrust/api/classes/classrandom_1_1uniform__real__distribution.html">random::uniform&#95;real&#95;distribution</a></b>;</span>
<br>
<span>template &lt;typename RealType&gt;</span>
<span>__host__ __device__ bool </span><span><b><a href="/thrust/api/groups/group__random__number__distributions.html#function-operator==">random::operator==</a></b>(const <a href="/thrust/api/classes/classrandom_1_1normal__distribution.html">normal_distribution</a>< RealType > & lhs,</span>
<span>&nbsp;&nbsp;const <a href="/thrust/api/classes/classrandom_1_1normal__distribution.html">normal_distribution</a>< RealType > & rhs);</span>
<br>
<span>template &lt;typename RealType&gt;</span>
<span>__host__ __device__ bool </span><span><b><a href="/thrust/api/groups/group__random__number__distributions.html#function-operator!=">random::operator!=</a></b>(const <a href="/thrust/api/classes/classrandom_1_1normal__distribution.html">normal_distribution</a>< RealType > & lhs,</span>
<span>&nbsp;&nbsp;const <a href="/thrust/api/classes/classrandom_1_1normal__distribution.html">normal_distribution</a>< RealType > & rhs);</span>
<br>
<span>template &lt;typename RealType,</span>
<span>&nbsp;&nbsp;typename CharT,</span>
<span>&nbsp;&nbsp;typename Traits&gt;</span>
<span>std::basic_ostream< CharT, Traits > & </span><span><b><a href="/thrust/api/groups/group__random__number__distributions.html#function-operator<<">random::operator&lt;&lt;</a></b>(std::basic_ostream< CharT, Traits > & os,</span>
<span>&nbsp;&nbsp;const <a href="/thrust/api/classes/classrandom_1_1normal__distribution.html">normal_distribution</a>< RealType > & d);</span>
<br>
<span>template &lt;typename RealType,</span>
<span>&nbsp;&nbsp;typename CharT,</span>
<span>&nbsp;&nbsp;typename Traits&gt;</span>
<span>std::basic_istream< CharT, Traits > & </span><span><b><a href="/thrust/api/groups/group__random__number__distributions.html#function-operator>>">random::operator&gt;&gt;</a></b>(std::basic_istream< CharT, Traits > & is,</span>
<span>&nbsp;&nbsp;<a href="/thrust/api/classes/classrandom_1_1normal__distribution.html">normal_distribution</a>< RealType > & d);</span>
<br>
<span>template &lt;typename IntType&gt;</span>
<span>__host__ __device__ bool </span><span><b><a href="/thrust/api/groups/group__random__number__distributions.html#function-operator==">random::operator==</a></b>(const <a href="/thrust/api/classes/classrandom_1_1uniform__int__distribution.html">uniform_int_distribution</a>< IntType > & lhs,</span>
<span>&nbsp;&nbsp;const <a href="/thrust/api/classes/classrandom_1_1uniform__int__distribution.html">uniform_int_distribution</a>< IntType > & rhs);</span>
<br>
<span>template &lt;typename IntType&gt;</span>
<span>__host__ __device__ bool </span><span><b><a href="/thrust/api/groups/group__random__number__distributions.html#function-operator!=">random::operator!=</a></b>(const <a href="/thrust/api/classes/classrandom_1_1uniform__int__distribution.html">uniform_int_distribution</a>< IntType > & lhs,</span>
<span>&nbsp;&nbsp;const <a href="/thrust/api/classes/classrandom_1_1uniform__int__distribution.html">uniform_int_distribution</a>< IntType > & rhs);</span>
<br>
<span>template &lt;typename IntType,</span>
<span>&nbsp;&nbsp;typename CharT,</span>
<span>&nbsp;&nbsp;typename Traits&gt;</span>
<span>std::basic_ostream< CharT, Traits > & </span><span><b><a href="/thrust/api/groups/group__random__number__distributions.html#function-operator<<">random::operator&lt;&lt;</a></b>(std::basic_ostream< CharT, Traits > & os,</span>
<span>&nbsp;&nbsp;const <a href="/thrust/api/classes/classrandom_1_1uniform__int__distribution.html">uniform_int_distribution</a>< IntType > & d);</span>
<br>
<span>template &lt;typename IntType,</span>
<span>&nbsp;&nbsp;typename CharT,</span>
<span>&nbsp;&nbsp;typename Traits&gt;</span>
<span>std::basic_istream< CharT, Traits > & </span><span><b><a href="/thrust/api/groups/group__random__number__distributions.html#function-operator>>">random::operator&gt;&gt;</a></b>(std::basic_istream< CharT, Traits > & is,</span>
<span>&nbsp;&nbsp;<a href="/thrust/api/classes/classrandom_1_1uniform__int__distribution.html">uniform_int_distribution</a>< IntType > & d);</span>
<br>
<span>template &lt;typename RealType&gt;</span>
<span>__host__ __device__ bool </span><span><b><a href="/thrust/api/groups/group__random__number__distributions.html#function-operator==">random::operator==</a></b>(const <a href="/thrust/api/classes/classrandom_1_1uniform__real__distribution.html">uniform_real_distribution</a>< RealType > & lhs,</span>
<span>&nbsp;&nbsp;const <a href="/thrust/api/classes/classrandom_1_1uniform__real__distribution.html">uniform_real_distribution</a>< RealType > & rhs);</span>
<br>
<span>template &lt;typename RealType&gt;</span>
<span>__host__ __device__ bool </span><span><b><a href="/thrust/api/groups/group__random__number__distributions.html#function-operator!=">random::operator!=</a></b>(const <a href="/thrust/api/classes/classrandom_1_1uniform__real__distribution.html">uniform_real_distribution</a>< RealType > & lhs,</span>
<span>&nbsp;&nbsp;const <a href="/thrust/api/classes/classrandom_1_1uniform__real__distribution.html">uniform_real_distribution</a>< RealType > & rhs);</span>
<br>
<span>template &lt;typename RealType,</span>
<span>&nbsp;&nbsp;typename CharT,</span>
<span>&nbsp;&nbsp;typename Traits&gt;</span>
<span>std::basic_ostream< CharT, Traits > & </span><span><b><a href="/thrust/api/groups/group__random__number__distributions.html#function-operator<<">random::operator&lt;&lt;</a></b>(std::basic_ostream< CharT, Traits > & os,</span>
<span>&nbsp;&nbsp;const <a href="/thrust/api/classes/classrandom_1_1uniform__real__distribution.html">uniform_real_distribution</a>< RealType > & d);</span>
<br>
<span>template &lt;typename RealType,</span>
<span>&nbsp;&nbsp;typename CharT,</span>
<span>&nbsp;&nbsp;typename Traits&gt;</span>
<span>std::basic_istream< CharT, Traits > & </span><span><b><a href="/thrust/api/groups/group__random__number__distributions.html#function-operator>>">random::operator&gt;&gt;</a></b>(std::basic_istream< CharT, Traits > & is,</span>
<span>&nbsp;&nbsp;<a href="/thrust/api/classes/classrandom_1_1uniform__real__distribution.html">uniform_real_distribution</a>< RealType > & d);</span>
</code>

## Member Classes

<h3 id="class-random::normal_distribution">
<a href="/thrust/api/classes/classrandom_1_1normal__distribution.html">Class <code>random::normal&#95;distribution</code>
</a>
</h3>

A <code>normal&#95;distribution</code> random number distribution produces floating point Normally distributed random numbers. 

**Inherits From**:
`detail::normal_distribution_base::type< double >`

<h3 id="class-random::uniform_int_distribution">
<a href="/thrust/api/classes/classrandom_1_1uniform__int__distribution.html">Class <code>random::uniform&#95;int&#95;distribution</code>
</a>
</h3>

A <code>uniform&#95;int&#95;distribution</code> random number distribution produces signed or unsigned integer uniform random numbers from a given range. 

<h3 id="class-random::uniform_real_distribution">
<a href="/thrust/api/classes/classrandom_1_1uniform__real__distribution.html">Class <code>random::uniform&#95;real&#95;distribution</code>
</a>
</h3>

A <code>uniform&#95;real&#95;distribution</code> random number distribution produces floating point uniform random numbers from a half-open interval. 


## Functions

<h3 id="function-operator==">
Function <code>random::operator==</code>
</h3>

<code class="doxybook">
<span>template &lt;typename RealType&gt;</span>
<span>__host__ __device__ bool </span><span><b>operator==</b>(const <a href="/thrust/api/classes/classrandom_1_1normal__distribution.html">normal_distribution</a>< RealType > & lhs,</span>
<span>&nbsp;&nbsp;const <a href="/thrust/api/classes/classrandom_1_1normal__distribution.html">normal_distribution</a>< RealType > & rhs);</span></code>
This function checks two <code>normal&#95;distributions</code> for equality. 

**Function Parameters**:
* **`lhs`** The first <code><a href="/thrust/api/classes/classrandom_1_1normal__distribution.html">normal&#95;distribution</a></code> to test. 
* **`rhs`** The second <code><a href="/thrust/api/classes/classrandom_1_1normal__distribution.html">normal&#95;distribution</a></code> to test. 

**Returns**:
<code>true</code> if <code>lhs</code> is equal to <code>rhs</code>; <code>false</code>, otherwise. 

<h3 id="function-operator!=">
Function <code>random::operator!=</code>
</h3>

<code class="doxybook">
<span>template &lt;typename RealType&gt;</span>
<span>__host__ __device__ bool </span><span><b>operator!=</b>(const <a href="/thrust/api/classes/classrandom_1_1normal__distribution.html">normal_distribution</a>< RealType > & lhs,</span>
<span>&nbsp;&nbsp;const <a href="/thrust/api/classes/classrandom_1_1normal__distribution.html">normal_distribution</a>< RealType > & rhs);</span></code>
This function checks two <code>normal&#95;distributions</code> for inequality. 

**Function Parameters**:
* **`lhs`** The first <code><a href="/thrust/api/classes/classrandom_1_1normal__distribution.html">normal&#95;distribution</a></code> to test. 
* **`rhs`** The second <code><a href="/thrust/api/classes/classrandom_1_1normal__distribution.html">normal&#95;distribution</a></code> to test. 

**Returns**:
<code>true</code> if <code>lhs</code> is not equal to <code>rhs</code>; <code>false</code>, otherwise. 

<h3 id="function-operator<<">
Function <code>random::operator&lt;&lt;</code>
</h3>

<code class="doxybook">
<span>template &lt;typename RealType,</span>
<span>&nbsp;&nbsp;typename CharT,</span>
<span>&nbsp;&nbsp;typename Traits&gt;</span>
<span>std::basic_ostream< CharT, Traits > & </span><span><b>operator<<</b>(std::basic_ostream< CharT, Traits > & os,</span>
<span>&nbsp;&nbsp;const <a href="/thrust/api/classes/classrandom_1_1normal__distribution.html">normal_distribution</a>< RealType > & d);</span></code>
This function streams a <a href="/thrust/api/classes/classrandom_1_1normal__distribution.html">normal_distribution</a> to a <code>std::basic&#95;ostream</code>. 

**Function Parameters**:
* **`os`** The <code>basic&#95;ostream</code> to stream out to. 
* **`d`** The <code><a href="/thrust/api/classes/classrandom_1_1normal__distribution.html">normal&#95;distribution</a></code> to stream out. 

**Returns**:
<code>os</code>

<h3 id="function-operator>>">
Function <code>random::operator&gt;&gt;</code>
</h3>

<code class="doxybook">
<span>template &lt;typename RealType,</span>
<span>&nbsp;&nbsp;typename CharT,</span>
<span>&nbsp;&nbsp;typename Traits&gt;</span>
<span>std::basic_istream< CharT, Traits > & </span><span><b>operator>></b>(std::basic_istream< CharT, Traits > & is,</span>
<span>&nbsp;&nbsp;<a href="/thrust/api/classes/classrandom_1_1normal__distribution.html">normal_distribution</a>< RealType > & d);</span></code>
This function streams a <a href="/thrust/api/classes/classrandom_1_1normal__distribution.html">normal_distribution</a> in from a std::basic_istream. 

**Function Parameters**:
* **`is`** The <code>basic&#95;istream</code> to stream from. 
* **`d`** The <code><a href="/thrust/api/classes/classrandom_1_1normal__distribution.html">normal&#95;distribution</a></code> to stream in. 

**Returns**:
<code>is</code>

<h3 id="function-operator==">
Function <code>random::operator==</code>
</h3>

<code class="doxybook">
<span>template &lt;typename IntType&gt;</span>
<span>__host__ __device__ bool </span><span><b>operator==</b>(const <a href="/thrust/api/classes/classrandom_1_1uniform__int__distribution.html">uniform_int_distribution</a>< IntType > & lhs,</span>
<span>&nbsp;&nbsp;const <a href="/thrust/api/classes/classrandom_1_1uniform__int__distribution.html">uniform_int_distribution</a>< IntType > & rhs);</span></code>
This function checks two <code>uniform&#95;int&#95;distributions</code> for equality. 

**Function Parameters**:
* **`lhs`** The first <code><a href="/thrust/api/classes/classrandom_1_1uniform__int__distribution.html">uniform&#95;int&#95;distribution</a></code> to test. 
* **`rhs`** The second <code><a href="/thrust/api/classes/classrandom_1_1uniform__int__distribution.html">uniform&#95;int&#95;distribution</a></code> to test. 

**Returns**:
<code>true</code> if <code>lhs</code> is equal to <code>rhs</code>; <code>false</code>, otherwise. 

<h3 id="function-operator!=">
Function <code>random::operator!=</code>
</h3>

<code class="doxybook">
<span>template &lt;typename IntType&gt;</span>
<span>__host__ __device__ bool </span><span><b>operator!=</b>(const <a href="/thrust/api/classes/classrandom_1_1uniform__int__distribution.html">uniform_int_distribution</a>< IntType > & lhs,</span>
<span>&nbsp;&nbsp;const <a href="/thrust/api/classes/classrandom_1_1uniform__int__distribution.html">uniform_int_distribution</a>< IntType > & rhs);</span></code>
This function checks two <code>uniform&#95;int&#95;distributions</code> for inequality. 

**Function Parameters**:
* **`lhs`** The first <code><a href="/thrust/api/classes/classrandom_1_1uniform__int__distribution.html">uniform&#95;int&#95;distribution</a></code> to test. 
* **`rhs`** The second <code><a href="/thrust/api/classes/classrandom_1_1uniform__int__distribution.html">uniform&#95;int&#95;distribution</a></code> to test. 

**Returns**:
<code>true</code> if <code>lhs</code> is not equal to <code>rhs</code>; <code>false</code>, otherwise. 

<h3 id="function-operator<<">
Function <code>random::operator&lt;&lt;</code>
</h3>

<code class="doxybook">
<span>template &lt;typename IntType,</span>
<span>&nbsp;&nbsp;typename CharT,</span>
<span>&nbsp;&nbsp;typename Traits&gt;</span>
<span>std::basic_ostream< CharT, Traits > & </span><span><b>operator<<</b>(std::basic_ostream< CharT, Traits > & os,</span>
<span>&nbsp;&nbsp;const <a href="/thrust/api/classes/classrandom_1_1uniform__int__distribution.html">uniform_int_distribution</a>< IntType > & d);</span></code>
This function streams a <a href="/thrust/api/classes/classrandom_1_1uniform__int__distribution.html">uniform_int_distribution</a> to a <code>std::basic&#95;ostream</code>. 

**Function Parameters**:
* **`os`** The <code>basic&#95;ostream</code> to stream out to. 
* **`d`** The <code><a href="/thrust/api/classes/classrandom_1_1uniform__int__distribution.html">uniform&#95;int&#95;distribution</a></code> to stream out. 

**Returns**:
<code>os</code>

<h3 id="function-operator>>">
Function <code>random::operator&gt;&gt;</code>
</h3>

<code class="doxybook">
<span>template &lt;typename IntType,</span>
<span>&nbsp;&nbsp;typename CharT,</span>
<span>&nbsp;&nbsp;typename Traits&gt;</span>
<span>std::basic_istream< CharT, Traits > & </span><span><b>operator>></b>(std::basic_istream< CharT, Traits > & is,</span>
<span>&nbsp;&nbsp;<a href="/thrust/api/classes/classrandom_1_1uniform__int__distribution.html">uniform_int_distribution</a>< IntType > & d);</span></code>
This function streams a <a href="/thrust/api/classes/classrandom_1_1uniform__int__distribution.html">uniform_int_distribution</a> in from a std::basic_istream. 

**Function Parameters**:
* **`is`** The <code>basic&#95;istream</code> to stream from. 
* **`d`** The <code><a href="/thrust/api/classes/classrandom_1_1uniform__int__distribution.html">uniform&#95;int&#95;distribution</a></code> to stream in. 

**Returns**:
<code>is</code>

<h3 id="function-operator==">
Function <code>random::operator==</code>
</h3>

<code class="doxybook">
<span>template &lt;typename RealType&gt;</span>
<span>__host__ __device__ bool </span><span><b>operator==</b>(const <a href="/thrust/api/classes/classrandom_1_1uniform__real__distribution.html">uniform_real_distribution</a>< RealType > & lhs,</span>
<span>&nbsp;&nbsp;const <a href="/thrust/api/classes/classrandom_1_1uniform__real__distribution.html">uniform_real_distribution</a>< RealType > & rhs);</span></code>
This function checks two <code>uniform&#95;real&#95;distributions</code> for equality. 

**Function Parameters**:
* **`lhs`** The first <code><a href="/thrust/api/classes/classrandom_1_1uniform__real__distribution.html">uniform&#95;real&#95;distribution</a></code> to test. 
* **`rhs`** The second <code><a href="/thrust/api/classes/classrandom_1_1uniform__real__distribution.html">uniform&#95;real&#95;distribution</a></code> to test. 

**Returns**:
<code>true</code> if <code>lhs</code> is equal to <code>rhs</code>; <code>false</code>, otherwise. 

<h3 id="function-operator!=">
Function <code>random::operator!=</code>
</h3>

<code class="doxybook">
<span>template &lt;typename RealType&gt;</span>
<span>__host__ __device__ bool </span><span><b>operator!=</b>(const <a href="/thrust/api/classes/classrandom_1_1uniform__real__distribution.html">uniform_real_distribution</a>< RealType > & lhs,</span>
<span>&nbsp;&nbsp;const <a href="/thrust/api/classes/classrandom_1_1uniform__real__distribution.html">uniform_real_distribution</a>< RealType > & rhs);</span></code>
This function checks two <code>uniform&#95;real&#95;distributions</code> for inequality. 

**Function Parameters**:
* **`lhs`** The first <code><a href="/thrust/api/classes/classrandom_1_1uniform__real__distribution.html">uniform&#95;real&#95;distribution</a></code> to test. 
* **`rhs`** The second <code><a href="/thrust/api/classes/classrandom_1_1uniform__real__distribution.html">uniform&#95;real&#95;distribution</a></code> to test. 

**Returns**:
<code>true</code> if <code>lhs</code> is not equal to <code>rhs</code>; <code>false</code>, otherwise. 

<h3 id="function-operator<<">
Function <code>random::operator&lt;&lt;</code>
</h3>

<code class="doxybook">
<span>template &lt;typename RealType,</span>
<span>&nbsp;&nbsp;typename CharT,</span>
<span>&nbsp;&nbsp;typename Traits&gt;</span>
<span>std::basic_ostream< CharT, Traits > & </span><span><b>operator<<</b>(std::basic_ostream< CharT, Traits > & os,</span>
<span>&nbsp;&nbsp;const <a href="/thrust/api/classes/classrandom_1_1uniform__real__distribution.html">uniform_real_distribution</a>< RealType > & d);</span></code>
This function streams a <a href="/thrust/api/classes/classrandom_1_1uniform__real__distribution.html">uniform_real_distribution</a> to a <code>std::basic&#95;ostream</code>. 

**Function Parameters**:
* **`os`** The <code>basic&#95;ostream</code> to stream out to. 
* **`d`** The <code><a href="/thrust/api/classes/classrandom_1_1uniform__real__distribution.html">uniform&#95;real&#95;distribution</a></code> to stream out. 

**Returns**:
<code>os</code>

<h3 id="function-operator>>">
Function <code>random::operator&gt;&gt;</code>
</h3>

<code class="doxybook">
<span>template &lt;typename RealType,</span>
<span>&nbsp;&nbsp;typename CharT,</span>
<span>&nbsp;&nbsp;typename Traits&gt;</span>
<span>std::basic_istream< CharT, Traits > & </span><span><b>operator>></b>(std::basic_istream< CharT, Traits > & is,</span>
<span>&nbsp;&nbsp;<a href="/thrust/api/classes/classrandom_1_1uniform__real__distribution.html">uniform_real_distribution</a>< RealType > & d);</span></code>
This function streams a <a href="/thrust/api/classes/classrandom_1_1uniform__real__distribution.html">uniform_real_distribution</a> in from a std::basic_istream. 

**Function Parameters**:
* **`is`** The <code>basic&#95;istream</code> to stream from. 
* **`d`** The <code><a href="/thrust/api/classes/classrandom_1_1uniform__real__distribution.html">uniform&#95;real&#95;distribution</a></code> to stream in. 

**Returns**:
<code>is</code>


