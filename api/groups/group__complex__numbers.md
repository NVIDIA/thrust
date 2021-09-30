---
title: Complex Numbers
parent: Numerics
grand_parent: API
nav_exclude: false
has_children: true
has_toc: false
---

# Complex Numbers

<code class="doxybook">
<span>template &lt;typename T&gt;</span>
<span>struct <b><a href="/thrust/api/classes/structcomplex.html">complex</a></b>;</span>
<br>
<span>template &lt;typename T&gt;</span>
<span>__host__ __device__ T </span><span><b><a href="/thrust/api/groups/group__complex__numbers.html#function-abs">abs</a></b>(const <a href="/thrust/api/classes/structcomplex.html">complex</a>< T > & z);</span>
<br>
<span>template &lt;typename T&gt;</span>
<span>__host__ __device__ T </span><span><b><a href="/thrust/api/groups/group__complex__numbers.html#function-arg">arg</a></b>(const <a href="/thrust/api/classes/structcomplex.html">complex</a>< T > & z);</span>
<br>
<span>template &lt;typename T&gt;</span>
<span>__host__ __device__ T </span><span><b><a href="/thrust/api/groups/group__complex__numbers.html#function-norm">norm</a></b>(const <a href="/thrust/api/classes/structcomplex.html">complex</a>< T > & z);</span>
<br>
<span>template &lt;typename T&gt;</span>
<span>__host__ __device__ <a href="/thrust/api/classes/structcomplex.html">complex</a>< T > </span><span><b><a href="/thrust/api/groups/group__complex__numbers.html#function-conj">conj</a></b>(const <a href="/thrust/api/classes/structcomplex.html">complex</a>< T > & z);</span>
<br>
<span>template &lt;typename T0,</span>
<span>&nbsp;&nbsp;typename T1&gt;</span>
<span>__host__ __device__ <a href="/thrust/api/classes/structcomplex.html">complex</a>< typename detail::promoted_numerical_type< T0, T1 >::type > </span><span><b><a href="/thrust/api/groups/group__complex__numbers.html#function-polar">polar</a></b>(const T0 & m,</span>
<span>&nbsp;&nbsp;const T1 & theta = T1());</span>
<br>
<span>template &lt;typename T&gt;</span>
<span>__host__ __device__ <a href="/thrust/api/classes/structcomplex.html">complex</a>< T > </span><span><b><a href="/thrust/api/groups/group__complex__numbers.html#function-proj">proj</a></b>(const T & z);</span>
<br>
<span>template &lt;typename T0,</span>
<span>&nbsp;&nbsp;typename T1&gt;</span>
<span>__host__ __device__ <a href="/thrust/api/classes/structcomplex.html">complex</a>< typename detail::promoted_numerical_type< T0, T1 >::type > </span><span><b><a href="/thrust/api/groups/group__complex__numbers.html#function-operator+">operator+</a></b>(const <a href="/thrust/api/classes/structcomplex.html">complex</a>< T0 > & x,</span>
<span>&nbsp;&nbsp;const <a href="/thrust/api/classes/structcomplex.html">complex</a>< T1 > & y);</span>
<br>
<span>template &lt;typename T0,</span>
<span>&nbsp;&nbsp;typename T1&gt;</span>
<span>__host__ __device__ <a href="/thrust/api/classes/structcomplex.html">complex</a>< typename detail::promoted_numerical_type< T0, T1 >::type > </span><span><b><a href="/thrust/api/groups/group__complex__numbers.html#function-operator+">operator+</a></b>(const <a href="/thrust/api/classes/structcomplex.html">complex</a>< T0 > & x,</span>
<span>&nbsp;&nbsp;const T1 & y);</span>
<br>
<span>template &lt;typename T0,</span>
<span>&nbsp;&nbsp;typename T1&gt;</span>
<span>__host__ __device__ <a href="/thrust/api/classes/structcomplex.html">complex</a>< typename detail::promoted_numerical_type< T0, T1 >::type > </span><span><b><a href="/thrust/api/groups/group__complex__numbers.html#function-operator+">operator+</a></b>(const T0 & x,</span>
<span>&nbsp;&nbsp;const <a href="/thrust/api/classes/structcomplex.html">complex</a>< T1 > & y);</span>
<br>
<span>template &lt;typename T0,</span>
<span>&nbsp;&nbsp;typename T1&gt;</span>
<span>__host__ __device__ <a href="/thrust/api/classes/structcomplex.html">complex</a>< typename detail::promoted_numerical_type< T0, T1 >::type > </span><span><b><a href="/thrust/api/groups/group__complex__numbers.html#function-operator-">operator-</a></b>(const <a href="/thrust/api/classes/structcomplex.html">complex</a>< T0 > & x,</span>
<span>&nbsp;&nbsp;const <a href="/thrust/api/classes/structcomplex.html">complex</a>< T1 > & y);</span>
<br>
<span>template &lt;typename T0,</span>
<span>&nbsp;&nbsp;typename T1&gt;</span>
<span>__host__ __device__ <a href="/thrust/api/classes/structcomplex.html">complex</a>< typename detail::promoted_numerical_type< T0, T1 >::type > </span><span><b><a href="/thrust/api/groups/group__complex__numbers.html#function-operator-">operator-</a></b>(const <a href="/thrust/api/classes/structcomplex.html">complex</a>< T0 > & x,</span>
<span>&nbsp;&nbsp;const T1 & y);</span>
<br>
<span>template &lt;typename T0,</span>
<span>&nbsp;&nbsp;typename T1&gt;</span>
<span>__host__ __device__ <a href="/thrust/api/classes/structcomplex.html">complex</a>< typename detail::promoted_numerical_type< T0, T1 >::type > </span><span><b><a href="/thrust/api/groups/group__complex__numbers.html#function-operator-">operator-</a></b>(const T0 & x,</span>
<span>&nbsp;&nbsp;const <a href="/thrust/api/classes/structcomplex.html">complex</a>< T1 > & y);</span>
<br>
<span>template &lt;typename T0,</span>
<span>&nbsp;&nbsp;typename T1&gt;</span>
<span>__host__ __device__ <a href="/thrust/api/classes/structcomplex.html">complex</a>< typename detail::promoted_numerical_type< T0, T1 >::type > </span><span><b><a href="/thrust/api/groups/group__complex__numbers.html#function-operator*">operator&#42;</a></b>(const <a href="/thrust/api/classes/structcomplex.html">complex</a>< T0 > & x,</span>
<span>&nbsp;&nbsp;const <a href="/thrust/api/classes/structcomplex.html">complex</a>< T1 > & y);</span>
<br>
<span>template &lt;typename T0,</span>
<span>&nbsp;&nbsp;typename T1&gt;</span>
<span>__host__ __device__ <a href="/thrust/api/classes/structcomplex.html">complex</a>< typename detail::promoted_numerical_type< T0, T1 >::type > </span><span><b><a href="/thrust/api/groups/group__complex__numbers.html#function-operator*">operator&#42;</a></b>(const <a href="/thrust/api/classes/structcomplex.html">complex</a>< T0 > & x,</span>
<span>&nbsp;&nbsp;const T1 & y);</span>
<br>
<span>template &lt;typename T0,</span>
<span>&nbsp;&nbsp;typename T1&gt;</span>
<span>__host__ __device__ <a href="/thrust/api/classes/structcomplex.html">complex</a>< typename detail::promoted_numerical_type< T0, T1 >::type > </span><span><b><a href="/thrust/api/groups/group__complex__numbers.html#function-operator*">operator&#42;</a></b>(const T0 & x,</span>
<span>&nbsp;&nbsp;const <a href="/thrust/api/classes/structcomplex.html">complex</a>< T1 > & y);</span>
<br>
<span>template &lt;typename T0,</span>
<span>&nbsp;&nbsp;typename T1&gt;</span>
<span>__host__ __device__ <a href="/thrust/api/classes/structcomplex.html">complex</a>< typename detail::promoted_numerical_type< T0, T1 >::type > </span><span><b><a href="/thrust/api/groups/group__complex__numbers.html#function-operator/">operator/</a></b>(const <a href="/thrust/api/classes/structcomplex.html">complex</a>< T0 > & x,</span>
<span>&nbsp;&nbsp;const <a href="/thrust/api/classes/structcomplex.html">complex</a>< T1 > & y);</span>
<br>
<span>template &lt;typename T0,</span>
<span>&nbsp;&nbsp;typename T1&gt;</span>
<span>__host__ __device__ <a href="/thrust/api/classes/structcomplex.html">complex</a>< typename detail::promoted_numerical_type< T0, T1 >::type > </span><span><b><a href="/thrust/api/groups/group__complex__numbers.html#function-operator/">operator/</a></b>(const <a href="/thrust/api/classes/structcomplex.html">complex</a>< T0 > & x,</span>
<span>&nbsp;&nbsp;const T1 & y);</span>
<br>
<span>template &lt;typename T0,</span>
<span>&nbsp;&nbsp;typename T1&gt;</span>
<span>__host__ __device__ <a href="/thrust/api/classes/structcomplex.html">complex</a>< typename detail::promoted_numerical_type< T0, T1 >::type > </span><span><b><a href="/thrust/api/groups/group__complex__numbers.html#function-operator/">operator/</a></b>(const T0 & x,</span>
<span>&nbsp;&nbsp;const <a href="/thrust/api/classes/structcomplex.html">complex</a>< T1 > & y);</span>
<br>
<span>template &lt;typename T&gt;</span>
<span>__host__ __device__ <a href="/thrust/api/classes/structcomplex.html">complex</a>< T > </span><span><b><a href="/thrust/api/groups/group__complex__numbers.html#function-operator+">operator+</a></b>(const <a href="/thrust/api/classes/structcomplex.html">complex</a>< T > & y);</span>
<br>
<span>template &lt;typename T&gt;</span>
<span>__host__ __device__ <a href="/thrust/api/classes/structcomplex.html">complex</a>< T > </span><span><b><a href="/thrust/api/groups/group__complex__numbers.html#function-operator-">operator-</a></b>(const <a href="/thrust/api/classes/structcomplex.html">complex</a>< T > & y);</span>
<br>
<span>template &lt;typename T&gt;</span>
<span>__host__ __device__ <a href="/thrust/api/classes/structcomplex.html">complex</a>< T > </span><span><b><a href="/thrust/api/groups/group__complex__numbers.html#function-exp">exp</a></b>(const <a href="/thrust/api/classes/structcomplex.html">complex</a>< T > & z);</span>
<br>
<span>template &lt;typename T&gt;</span>
<span>__host__ __device__ <a href="/thrust/api/classes/structcomplex.html">complex</a>< T > </span><span><b><a href="/thrust/api/groups/group__complex__numbers.html#function-log">log</a></b>(const <a href="/thrust/api/classes/structcomplex.html">complex</a>< T > & z);</span>
<br>
<span>template &lt;typename T&gt;</span>
<span>__host__ __device__ <a href="/thrust/api/classes/structcomplex.html">complex</a>< T > </span><span><b><a href="/thrust/api/groups/group__complex__numbers.html#function-log10">log10</a></b>(const <a href="/thrust/api/classes/structcomplex.html">complex</a>< T > & z);</span>
<br>
<span>template &lt;typename T0,</span>
<span>&nbsp;&nbsp;typename T1&gt;</span>
<span>__host__ __device__ <a href="/thrust/api/classes/structcomplex.html">complex</a>< typename detail::promoted_numerical_type< T0, T1 >::type > </span><span><b><a href="/thrust/api/groups/group__complex__numbers.html#function-pow">pow</a></b>(const <a href="/thrust/api/classes/structcomplex.html">complex</a>< T0 > & x,</span>
<span>&nbsp;&nbsp;const <a href="/thrust/api/classes/structcomplex.html">complex</a>< T1 > & y);</span>
<br>
<span>template &lt;typename T0,</span>
<span>&nbsp;&nbsp;typename T1&gt;</span>
<span>__host__ __device__ <a href="/thrust/api/classes/structcomplex.html">complex</a>< typename detail::promoted_numerical_type< T0, T1 >::type > </span><span><b><a href="/thrust/api/groups/group__complex__numbers.html#function-pow">pow</a></b>(const <a href="/thrust/api/classes/structcomplex.html">complex</a>< T0 > & x,</span>
<span>&nbsp;&nbsp;const T1 & y);</span>
<br>
<span>template &lt;typename T0,</span>
<span>&nbsp;&nbsp;typename T1&gt;</span>
<span>__host__ __device__ <a href="/thrust/api/classes/structcomplex.html">complex</a>< typename detail::promoted_numerical_type< T0, T1 >::type > </span><span><b><a href="/thrust/api/groups/group__complex__numbers.html#function-pow">pow</a></b>(const T0 & x,</span>
<span>&nbsp;&nbsp;const <a href="/thrust/api/classes/structcomplex.html">complex</a>< T1 > & y);</span>
<br>
<span>template &lt;typename T&gt;</span>
<span>__host__ __device__ <a href="/thrust/api/classes/structcomplex.html">complex</a>< T > </span><span><b><a href="/thrust/api/groups/group__complex__numbers.html#function-sqrt">sqrt</a></b>(const <a href="/thrust/api/classes/structcomplex.html">complex</a>< T > & z);</span>
<br>
<span>template &lt;typename T&gt;</span>
<span>__host__ __device__ <a href="/thrust/api/classes/structcomplex.html">complex</a>< T > </span><span><b><a href="/thrust/api/groups/group__complex__numbers.html#function-cos">cos</a></b>(const <a href="/thrust/api/classes/structcomplex.html">complex</a>< T > & z);</span>
<br>
<span>template &lt;typename T&gt;</span>
<span>__host__ __device__ <a href="/thrust/api/classes/structcomplex.html">complex</a>< T > </span><span><b><a href="/thrust/api/groups/group__complex__numbers.html#function-sin">sin</a></b>(const <a href="/thrust/api/classes/structcomplex.html">complex</a>< T > & z);</span>
<br>
<span>template &lt;typename T&gt;</span>
<span>__host__ __device__ <a href="/thrust/api/classes/structcomplex.html">complex</a>< T > </span><span><b><a href="/thrust/api/groups/group__complex__numbers.html#function-tan">tan</a></b>(const <a href="/thrust/api/classes/structcomplex.html">complex</a>< T > & z);</span>
<br>
<span>template &lt;typename T&gt;</span>
<span>__host__ __device__ <a href="/thrust/api/classes/structcomplex.html">complex</a>< T > </span><span><b><a href="/thrust/api/groups/group__complex__numbers.html#function-cosh">cosh</a></b>(const <a href="/thrust/api/classes/structcomplex.html">complex</a>< T > & z);</span>
<br>
<span>template &lt;typename T&gt;</span>
<span>__host__ __device__ <a href="/thrust/api/classes/structcomplex.html">complex</a>< T > </span><span><b><a href="/thrust/api/groups/group__complex__numbers.html#function-sinh">sinh</a></b>(const <a href="/thrust/api/classes/structcomplex.html">complex</a>< T > & z);</span>
<br>
<span>template &lt;typename T&gt;</span>
<span>__host__ __device__ <a href="/thrust/api/classes/structcomplex.html">complex</a>< T > </span><span><b><a href="/thrust/api/groups/group__complex__numbers.html#function-tanh">tanh</a></b>(const <a href="/thrust/api/classes/structcomplex.html">complex</a>< T > & z);</span>
<br>
<span>template &lt;typename T&gt;</span>
<span>__host__ __device__ <a href="/thrust/api/classes/structcomplex.html">complex</a>< T > </span><span><b><a href="/thrust/api/groups/group__complex__numbers.html#function-acos">acos</a></b>(const <a href="/thrust/api/classes/structcomplex.html">complex</a>< T > & z);</span>
<br>
<span>template &lt;typename T&gt;</span>
<span>__host__ __device__ <a href="/thrust/api/classes/structcomplex.html">complex</a>< T > </span><span><b><a href="/thrust/api/groups/group__complex__numbers.html#function-asin">asin</a></b>(const <a href="/thrust/api/classes/structcomplex.html">complex</a>< T > & z);</span>
<br>
<span>template &lt;typename T&gt;</span>
<span>__host__ __device__ <a href="/thrust/api/classes/structcomplex.html">complex</a>< T > </span><span><b><a href="/thrust/api/groups/group__complex__numbers.html#function-atan">atan</a></b>(const <a href="/thrust/api/classes/structcomplex.html">complex</a>< T > & z);</span>
<br>
<span>template &lt;typename T&gt;</span>
<span>__host__ __device__ <a href="/thrust/api/classes/structcomplex.html">complex</a>< T > </span><span><b><a href="/thrust/api/groups/group__complex__numbers.html#function-acosh">acosh</a></b>(const <a href="/thrust/api/classes/structcomplex.html">complex</a>< T > & z);</span>
<br>
<span>template &lt;typename T&gt;</span>
<span>__host__ __device__ <a href="/thrust/api/classes/structcomplex.html">complex</a>< T > </span><span><b><a href="/thrust/api/groups/group__complex__numbers.html#function-asinh">asinh</a></b>(const <a href="/thrust/api/classes/structcomplex.html">complex</a>< T > & z);</span>
<br>
<span>template &lt;typename T&gt;</span>
<span>__host__ __device__ <a href="/thrust/api/classes/structcomplex.html">complex</a>< T > </span><span><b><a href="/thrust/api/groups/group__complex__numbers.html#function-atanh">atanh</a></b>(const <a href="/thrust/api/classes/structcomplex.html">complex</a>< T > & z);</span>
<br>
<span>template &lt;typename T,</span>
<span>&nbsp;&nbsp;typename CharT,</span>
<span>&nbsp;&nbsp;typename Traits&gt;</span>
<span>std::basic_ostream< CharT, Traits > & </span><span><b><a href="/thrust/api/groups/group__complex__numbers.html#function-operator<<">operator&lt;&lt;</a></b>(std::basic_ostream< CharT, Traits > & os,</span>
<span>&nbsp;&nbsp;const <a href="/thrust/api/classes/structcomplex.html">complex</a>< T > & z);</span>
<br>
<span>template &lt;typename T,</span>
<span>&nbsp;&nbsp;typename CharT,</span>
<span>&nbsp;&nbsp;typename Traits&gt;</span>
<span>__host__ std::basic_istream< CharT, Traits > & </span><span><b><a href="/thrust/api/groups/group__complex__numbers.html#function-operator>>">operator&gt;&gt;</a></b>(std::basic_istream< CharT, Traits > & is,</span>
<span>&nbsp;&nbsp;<a href="/thrust/api/classes/structcomplex.html">complex</a>< T > & z);</span>
<br>
<span>template &lt;typename T0,</span>
<span>&nbsp;&nbsp;typename T1&gt;</span>
<span>__host__ __device__ bool </span><span><b><a href="/thrust/api/groups/group__complex__numbers.html#function-operator==">operator==</a></b>(const <a href="/thrust/api/classes/structcomplex.html">complex</a>< T0 > & x,</span>
<span>&nbsp;&nbsp;const <a href="/thrust/api/classes/structcomplex.html">complex</a>< T1 > & y);</span>
<br>
<span>template &lt;typename T0,</span>
<span>&nbsp;&nbsp;typename T1&gt;</span>
<span>__host__ __device__ bool </span><span><b><a href="/thrust/api/groups/group__complex__numbers.html#function-operator==">operator==</a></b>(const <a href="/thrust/api/classes/structcomplex.html">complex</a>< T0 > & x,</span>
<span>&nbsp;&nbsp;const std::complex< T1 > & y);</span>
<br>
<span>template &lt;typename T0,</span>
<span>&nbsp;&nbsp;typename T1&gt;</span>
<span>__host__ __device__ bool </span><span><b><a href="/thrust/api/groups/group__complex__numbers.html#function-operator==">operator==</a></b>(const std::complex< T0 > & x,</span>
<span>&nbsp;&nbsp;const <a href="/thrust/api/classes/structcomplex.html">complex</a>< T1 > & y);</span>
<br>
<span>template &lt;typename T0,</span>
<span>&nbsp;&nbsp;typename T1&gt;</span>
<span>__host__ __device__ bool </span><span><b><a href="/thrust/api/groups/group__complex__numbers.html#function-operator==">operator==</a></b>(const T0 & x,</span>
<span>&nbsp;&nbsp;const <a href="/thrust/api/classes/structcomplex.html">complex</a>< T1 > & y);</span>
<br>
<span>template &lt;typename T0,</span>
<span>&nbsp;&nbsp;typename T1&gt;</span>
<span>__host__ __device__ bool </span><span><b><a href="/thrust/api/groups/group__complex__numbers.html#function-operator==">operator==</a></b>(const <a href="/thrust/api/classes/structcomplex.html">complex</a>< T0 > & x,</span>
<span>&nbsp;&nbsp;const T1 & y);</span>
<br>
<span>template &lt;typename T0,</span>
<span>&nbsp;&nbsp;typename T1&gt;</span>
<span>__host__ __device__ bool </span><span><b><a href="/thrust/api/groups/group__complex__numbers.html#function-operator!=">operator!=</a></b>(const <a href="/thrust/api/classes/structcomplex.html">complex</a>< T0 > & x,</span>
<span>&nbsp;&nbsp;const <a href="/thrust/api/classes/structcomplex.html">complex</a>< T1 > & y);</span>
<br>
<span>template &lt;typename T0,</span>
<span>&nbsp;&nbsp;typename T1&gt;</span>
<span>__host__ __device__ bool </span><span><b><a href="/thrust/api/groups/group__complex__numbers.html#function-operator!=">operator!=</a></b>(const <a href="/thrust/api/classes/structcomplex.html">complex</a>< T0 > & x,</span>
<span>&nbsp;&nbsp;const std::complex< T1 > & y);</span>
<br>
<span>template &lt;typename T0,</span>
<span>&nbsp;&nbsp;typename T1&gt;</span>
<span>__host__ __device__ bool </span><span><b><a href="/thrust/api/groups/group__complex__numbers.html#function-operator!=">operator!=</a></b>(const std::complex< T0 > & x,</span>
<span>&nbsp;&nbsp;const <a href="/thrust/api/classes/structcomplex.html">complex</a>< T1 > & y);</span>
<br>
<span>template &lt;typename T0,</span>
<span>&nbsp;&nbsp;typename T1&gt;</span>
<span>__host__ __device__ bool </span><span><b><a href="/thrust/api/groups/group__complex__numbers.html#function-operator!=">operator!=</a></b>(const T0 & x,</span>
<span>&nbsp;&nbsp;const <a href="/thrust/api/classes/structcomplex.html">complex</a>< T1 > & y);</span>
<br>
<span>template &lt;typename T0,</span>
<span>&nbsp;&nbsp;typename T1&gt;</span>
<span>__host__ __device__ bool </span><span><b><a href="/thrust/api/groups/group__complex__numbers.html#function-operator!=">operator!=</a></b>(const <a href="/thrust/api/classes/structcomplex.html">complex</a>< T0 > & x,</span>
<span>&nbsp;&nbsp;const T1 & y);</span>
</code>

## Member Classes

<h3 id="struct-complex">
<a href="/thrust/api/classes/structcomplex.html">Struct <code>complex</code>
</a>
</h3>


## Functions

<h3 id="function-abs">
Function <code>abs</code>
</h3>

<code class="doxybook">
<span>template &lt;typename T&gt;</span>
<span>__host__ __device__ T </span><span><b>abs</b>(const <a href="/thrust/api/classes/structcomplex.html">complex</a>< T > & z);</span></code>
Returns the magnitude (also known as absolute value) of a <code>complex</code>.

**Function Parameters**:
**`z`**: The <code>complex</code> from which to calculate the absolute value. 

<h3 id="function-arg">
Function <code>arg</code>
</h3>

<code class="doxybook">
<span>template &lt;typename T&gt;</span>
<span>__host__ __device__ T </span><span><b>arg</b>(const <a href="/thrust/api/classes/structcomplex.html">complex</a>< T > & z);</span></code>
Returns the phase angle (also known as argument) in radians of a <code>complex</code>.

**Function Parameters**:
**`z`**: The <code>complex</code> from which to calculate the phase angle. 

<h3 id="function-norm">
Function <code>norm</code>
</h3>

<code class="doxybook">
<span>template &lt;typename T&gt;</span>
<span>__host__ __device__ T </span><span><b>norm</b>(const <a href="/thrust/api/classes/structcomplex.html">complex</a>< T > & z);</span></code>
Returns the square of the magnitude of a <code>complex</code>.

**Function Parameters**:
**`z`**: The <code>complex</code> from which to calculate the norm. 

<h3 id="function-conj">
Function <code>conj</code>
</h3>

<code class="doxybook">
<span>template &lt;typename T&gt;</span>
<span>__host__ __device__ <a href="/thrust/api/classes/structcomplex.html">complex</a>< T > </span><span><b>conj</b>(const <a href="/thrust/api/classes/structcomplex.html">complex</a>< T > & z);</span></code>
Returns the complex conjugate of a <code>complex</code>.

**Function Parameters**:
**`z`**: The <code>complex</code> from which to calculate the complex conjugate. 

<h3 id="function-polar">
Function <code>polar</code>
</h3>

<code class="doxybook">
<span>template &lt;typename T0,</span>
<span>&nbsp;&nbsp;typename T1&gt;</span>
<span>__host__ __device__ <a href="/thrust/api/classes/structcomplex.html">complex</a>< typename detail::promoted_numerical_type< T0, T1 >::type > </span><span><b>polar</b>(const T0 & m,</span>
<span>&nbsp;&nbsp;const T1 & theta = T1());</span></code>
Returns a <code>complex</code> with the specified magnitude and phase.

**Function Parameters**:
* **`m`** The magnitude of the returned <code>complex</code>. 
* **`theta`** The phase of the returned <code>complex</code> in radians. 

<h3 id="function-proj">
Function <code>proj</code>
</h3>

<code class="doxybook">
<span>template &lt;typename T&gt;</span>
<span>__host__ __device__ <a href="/thrust/api/classes/structcomplex.html">complex</a>< T > </span><span><b>proj</b>(const T & z);</span></code>
Returns the projection of a <code>complex</code> on the Riemann sphere. For all finite <code>complex</code> it returns the argument. For <code>complexs</code> with a non finite part returns (INFINITY,+/-0) where the sign of the zero matches the sign of the imaginary part of the argument.

**Function Parameters**:
**`z`**: The <code>complex</code> argument. 

<h3 id="function-operator+">
Function <code>operator+</code>
</h3>

<code class="doxybook">
<span>template &lt;typename T0,</span>
<span>&nbsp;&nbsp;typename T1&gt;</span>
<span>__host__ __device__ <a href="/thrust/api/classes/structcomplex.html">complex</a>< typename detail::promoted_numerical_type< T0, T1 >::type > </span><span><b>operator+</b>(const <a href="/thrust/api/classes/structcomplex.html">complex</a>< T0 > & x,</span>
<span>&nbsp;&nbsp;const <a href="/thrust/api/classes/structcomplex.html">complex</a>< T1 > & y);</span></code>
Adds two <code>complex</code> numbers.

The value types of the two <code>complex</code> types should be compatible and the type of the returned <code>complex</code> is the promoted type of the two arguments.

**Function Parameters**:
* **`x`** The first <code>complex</code>. 
* **`y`** The second <code>complex</code>. 

<h3 id="function-operator+">
Function <code>operator+</code>
</h3>

<code class="doxybook">
<span>template &lt;typename T0,</span>
<span>&nbsp;&nbsp;typename T1&gt;</span>
<span>__host__ __device__ <a href="/thrust/api/classes/structcomplex.html">complex</a>< typename detail::promoted_numerical_type< T0, T1 >::type > </span><span><b>operator+</b>(const <a href="/thrust/api/classes/structcomplex.html">complex</a>< T0 > & x,</span>
<span>&nbsp;&nbsp;const T1 & y);</span></code>
Adds a scalar to a <code>complex</code> number.

The value type of the <code>complex</code> should be compatible with the scalar and the type of the returned <code>complex</code> is the promoted type of the two arguments.

**Function Parameters**:
* **`x`** The <code>complex</code>. 
* **`y`** The scalar. 

<h3 id="function-operator+">
Function <code>operator+</code>
</h3>

<code class="doxybook">
<span>template &lt;typename T0,</span>
<span>&nbsp;&nbsp;typename T1&gt;</span>
<span>__host__ __device__ <a href="/thrust/api/classes/structcomplex.html">complex</a>< typename detail::promoted_numerical_type< T0, T1 >::type > </span><span><b>operator+</b>(const T0 & x,</span>
<span>&nbsp;&nbsp;const <a href="/thrust/api/classes/structcomplex.html">complex</a>< T1 > & y);</span></code>
Adds a <code>complex</code> number to a scalar.

The value type of the <code>complex</code> should be compatible with the scalar and the type of the returned <code>complex</code> is the promoted type of the two arguments.

**Function Parameters**:
* **`x`** The scalar. 
* **`y`** The <code>complex</code>. 

<h3 id="function-operator-">
Function <code>operator-</code>
</h3>

<code class="doxybook">
<span>template &lt;typename T0,</span>
<span>&nbsp;&nbsp;typename T1&gt;</span>
<span>__host__ __device__ <a href="/thrust/api/classes/structcomplex.html">complex</a>< typename detail::promoted_numerical_type< T0, T1 >::type > </span><span><b>operator-</b>(const <a href="/thrust/api/classes/structcomplex.html">complex</a>< T0 > & x,</span>
<span>&nbsp;&nbsp;const <a href="/thrust/api/classes/structcomplex.html">complex</a>< T1 > & y);</span></code>
Subtracts two <code>complex</code> numbers.

The value types of the two <code>complex</code> types should be compatible and the type of the returned <code>complex</code> is the promoted type of the two arguments.

**Function Parameters**:
* **`x`** The first <code>complex</code> (minuend). 
* **`y`** The second <code>complex</code> (subtrahend). 

<h3 id="function-operator-">
Function <code>operator-</code>
</h3>

<code class="doxybook">
<span>template &lt;typename T0,</span>
<span>&nbsp;&nbsp;typename T1&gt;</span>
<span>__host__ __device__ <a href="/thrust/api/classes/structcomplex.html">complex</a>< typename detail::promoted_numerical_type< T0, T1 >::type > </span><span><b>operator-</b>(const <a href="/thrust/api/classes/structcomplex.html">complex</a>< T0 > & x,</span>
<span>&nbsp;&nbsp;const T1 & y);</span></code>
Subtracts a scalar from a <code>complex</code> number.

The value type of the <code>complex</code> should be compatible with the scalar and the type of the returned <code>complex</code> is the promoted type of the two arguments.

**Function Parameters**:
* **`x`** The <code>complex</code> (minuend). 
* **`y`** The scalar (subtrahend). 

<h3 id="function-operator-">
Function <code>operator-</code>
</h3>

<code class="doxybook">
<span>template &lt;typename T0,</span>
<span>&nbsp;&nbsp;typename T1&gt;</span>
<span>__host__ __device__ <a href="/thrust/api/classes/structcomplex.html">complex</a>< typename detail::promoted_numerical_type< T0, T1 >::type > </span><span><b>operator-</b>(const T0 & x,</span>
<span>&nbsp;&nbsp;const <a href="/thrust/api/classes/structcomplex.html">complex</a>< T1 > & y);</span></code>
Subtracts a <code>complex</code> number from a scalar.

The value type of the <code>complex</code> should be compatible with the scalar and the type of the returned <code>complex</code> is the promoted type of the two arguments.

**Function Parameters**:
* **`x`** The scalar (minuend). 
* **`y`** The <code>complex</code> (subtrahend). 

<h3 id="function-operator*">
Function <code>operator&#42;</code>
</h3>

<code class="doxybook">
<span>template &lt;typename T0,</span>
<span>&nbsp;&nbsp;typename T1&gt;</span>
<span>__host__ __device__ <a href="/thrust/api/classes/structcomplex.html">complex</a>< typename detail::promoted_numerical_type< T0, T1 >::type > </span><span><b>operator*</b>(const <a href="/thrust/api/classes/structcomplex.html">complex</a>< T0 > & x,</span>
<span>&nbsp;&nbsp;const <a href="/thrust/api/classes/structcomplex.html">complex</a>< T1 > & y);</span></code>
Multiplies two <code>complex</code> numbers.

The value types of the two <code>complex</code> types should be compatible and the type of the returned <code>complex</code> is the promoted type of the two arguments.

**Function Parameters**:
* **`x`** The first <code>complex</code>. 
* **`y`** The second <code>complex</code>. 

<h3 id="function-operator*">
Function <code>operator&#42;</code>
</h3>

<code class="doxybook">
<span>template &lt;typename T0,</span>
<span>&nbsp;&nbsp;typename T1&gt;</span>
<span>__host__ __device__ <a href="/thrust/api/classes/structcomplex.html">complex</a>< typename detail::promoted_numerical_type< T0, T1 >::type > </span><span><b>operator*</b>(const <a href="/thrust/api/classes/structcomplex.html">complex</a>< T0 > & x,</span>
<span>&nbsp;&nbsp;const T1 & y);</span></code>
Multiplies a <code>complex</code> number by a scalar.

**Function Parameters**:
* **`x`** The <code>complex</code>. 
* **`y`** The scalar. 

<h3 id="function-operator*">
Function <code>operator&#42;</code>
</h3>

<code class="doxybook">
<span>template &lt;typename T0,</span>
<span>&nbsp;&nbsp;typename T1&gt;</span>
<span>__host__ __device__ <a href="/thrust/api/classes/structcomplex.html">complex</a>< typename detail::promoted_numerical_type< T0, T1 >::type > </span><span><b>operator*</b>(const T0 & x,</span>
<span>&nbsp;&nbsp;const <a href="/thrust/api/classes/structcomplex.html">complex</a>< T1 > & y);</span></code>
Multiplies a scalar by a <code>complex</code> number.

The value type of the <code>complex</code> should be compatible with the scalar and the type of the returned <code>complex</code> is the promoted type of the two arguments.

**Function Parameters**:
* **`x`** The scalar. 
* **`y`** The <code>complex</code>. 

<h3 id="function-operator/">
Function <code>operator/</code>
</h3>

<code class="doxybook">
<span>template &lt;typename T0,</span>
<span>&nbsp;&nbsp;typename T1&gt;</span>
<span>__host__ __device__ <a href="/thrust/api/classes/structcomplex.html">complex</a>< typename detail::promoted_numerical_type< T0, T1 >::type > </span><span><b>operator/</b>(const <a href="/thrust/api/classes/structcomplex.html">complex</a>< T0 > & x,</span>
<span>&nbsp;&nbsp;const <a href="/thrust/api/classes/structcomplex.html">complex</a>< T1 > & y);</span></code>
Divides two <code>complex</code> numbers.

The value types of the two <code>complex</code> types should be compatible and the type of the returned <code>complex</code> is the promoted type of the two arguments.

**Function Parameters**:
* **`x`** The numerator (dividend). 
* **`y`** The denomimator (divisor). 

<h3 id="function-operator/">
Function <code>operator/</code>
</h3>

<code class="doxybook">
<span>template &lt;typename T0,</span>
<span>&nbsp;&nbsp;typename T1&gt;</span>
<span>__host__ __device__ <a href="/thrust/api/classes/structcomplex.html">complex</a>< typename detail::promoted_numerical_type< T0, T1 >::type > </span><span><b>operator/</b>(const <a href="/thrust/api/classes/structcomplex.html">complex</a>< T0 > & x,</span>
<span>&nbsp;&nbsp;const T1 & y);</span></code>
Divides a <code>complex</code> number by a scalar.

The value type of the <code>complex</code> should be compatible with the scalar and the type of the returned <code>complex</code> is the promoted type of the two arguments.

**Function Parameters**:
* **`x`** The complex numerator (dividend). 
* **`y`** The scalar denomimator (divisor). 

<h3 id="function-operator/">
Function <code>operator/</code>
</h3>

<code class="doxybook">
<span>template &lt;typename T0,</span>
<span>&nbsp;&nbsp;typename T1&gt;</span>
<span>__host__ __device__ <a href="/thrust/api/classes/structcomplex.html">complex</a>< typename detail::promoted_numerical_type< T0, T1 >::type > </span><span><b>operator/</b>(const T0 & x,</span>
<span>&nbsp;&nbsp;const <a href="/thrust/api/classes/structcomplex.html">complex</a>< T1 > & y);</span></code>
Divides a scalar by a <code>complex</code> number.

The value type of the <code>complex</code> should be compatible with the scalar and the type of the returned <code>complex</code> is the promoted type of the two arguments.

**Function Parameters**:
* **`x`** The scalar numerator (dividend). 
* **`y`** The complex denomimator (divisor). 

<h3 id="function-operator+">
Function <code>operator+</code>
</h3>

<code class="doxybook">
<span>template &lt;typename T&gt;</span>
<span>__host__ __device__ <a href="/thrust/api/classes/structcomplex.html">complex</a>< T > </span><span><b>operator+</b>(const <a href="/thrust/api/classes/structcomplex.html">complex</a>< T > & y);</span></code>
Unary plus, returns its <code>complex</code> argument.

**Function Parameters**:
**`y`**: The <code>complex</code> argument. 

<h3 id="function-operator-">
Function <code>operator-</code>
</h3>

<code class="doxybook">
<span>template &lt;typename T&gt;</span>
<span>__host__ __device__ <a href="/thrust/api/classes/structcomplex.html">complex</a>< T > </span><span><b>operator-</b>(const <a href="/thrust/api/classes/structcomplex.html">complex</a>< T > & y);</span></code>
Unary minus, returns the additive inverse (negation) of its <code>complex</code> argument.

**Function Parameters**:
**`y`**: The <code>complex</code> argument. 

<h3 id="function-exp">
Function <code>exp</code>
</h3>

<code class="doxybook">
<span>template &lt;typename T&gt;</span>
<span>__host__ __device__ <a href="/thrust/api/classes/structcomplex.html">complex</a>< T > </span><span><b>exp</b>(const <a href="/thrust/api/classes/structcomplex.html">complex</a>< T > & z);</span></code>
Returns the complex exponential of a <code>complex</code> number.

**Function Parameters**:
**`z`**: The <code>complex</code> argument. 

<h3 id="function-log">
Function <code>log</code>
</h3>

<code class="doxybook">
<span>template &lt;typename T&gt;</span>
<span>__host__ __device__ <a href="/thrust/api/classes/structcomplex.html">complex</a>< T > </span><span><b>log</b>(const <a href="/thrust/api/classes/structcomplex.html">complex</a>< T > & z);</span></code>
Returns the complex natural logarithm of a <code>complex</code> number.

**Function Parameters**:
**`z`**: The <code>complex</code> argument. 

<h3 id="function-log10">
Function <code>log10</code>
</h3>

<code class="doxybook">
<span>template &lt;typename T&gt;</span>
<span>__host__ __device__ <a href="/thrust/api/classes/structcomplex.html">complex</a>< T > </span><span><b>log10</b>(const <a href="/thrust/api/classes/structcomplex.html">complex</a>< T > & z);</span></code>
Returns the complex base 10 logarithm of a <code>complex</code> number.

**Function Parameters**:
**`z`**: The <code>complex</code> argument. 

<h3 id="function-pow">
Function <code>pow</code>
</h3>

<code class="doxybook">
<span>template &lt;typename T0,</span>
<span>&nbsp;&nbsp;typename T1&gt;</span>
<span>__host__ __device__ <a href="/thrust/api/classes/structcomplex.html">complex</a>< typename detail::promoted_numerical_type< T0, T1 >::type > </span><span><b>pow</b>(const <a href="/thrust/api/classes/structcomplex.html">complex</a>< T0 > & x,</span>
<span>&nbsp;&nbsp;const <a href="/thrust/api/classes/structcomplex.html">complex</a>< T1 > & y);</span></code>
Returns a <code>complex</code> number raised to another.

The value types of the two <code>complex</code> types should be compatible and the type of the returned <code>complex</code> is the promoted type of the two arguments.

**Function Parameters**:
* **`x`** The base. 
* **`y`** The exponent. 

<h3 id="function-pow">
Function <code>pow</code>
</h3>

<code class="doxybook">
<span>template &lt;typename T0,</span>
<span>&nbsp;&nbsp;typename T1&gt;</span>
<span>__host__ __device__ <a href="/thrust/api/classes/structcomplex.html">complex</a>< typename detail::promoted_numerical_type< T0, T1 >::type > </span><span><b>pow</b>(const <a href="/thrust/api/classes/structcomplex.html">complex</a>< T0 > & x,</span>
<span>&nbsp;&nbsp;const T1 & y);</span></code>
Returns a <code>complex</code> number raised to a scalar.

The value type of the <code>complex</code> should be compatible with the scalar and the type of the returned <code>complex</code> is the promoted type of the two arguments.

**Function Parameters**:
* **`x`** The base. 
* **`y`** The exponent. 

<h3 id="function-pow">
Function <code>pow</code>
</h3>

<code class="doxybook">
<span>template &lt;typename T0,</span>
<span>&nbsp;&nbsp;typename T1&gt;</span>
<span>__host__ __device__ <a href="/thrust/api/classes/structcomplex.html">complex</a>< typename detail::promoted_numerical_type< T0, T1 >::type > </span><span><b>pow</b>(const T0 & x,</span>
<span>&nbsp;&nbsp;const <a href="/thrust/api/classes/structcomplex.html">complex</a>< T1 > & y);</span></code>
Returns a scalar raised to a <code>complex</code> number.

The value type of the <code>complex</code> should be compatible with the scalar and the type of the returned <code>complex</code> is the promoted type of the two arguments.

**Function Parameters**:
* **`x`** The base. 
* **`y`** The exponent. 

<h3 id="function-sqrt">
Function <code>sqrt</code>
</h3>

<code class="doxybook">
<span>template &lt;typename T&gt;</span>
<span>__host__ __device__ <a href="/thrust/api/classes/structcomplex.html">complex</a>< T > </span><span><b>sqrt</b>(const <a href="/thrust/api/classes/structcomplex.html">complex</a>< T > & z);</span></code>
Returns the complex square root of a <code>complex</code> number.

**Function Parameters**:
**`z`**: The <code>complex</code> argument. 

<h3 id="function-cos">
Function <code>cos</code>
</h3>

<code class="doxybook">
<span>template &lt;typename T&gt;</span>
<span>__host__ __device__ <a href="/thrust/api/classes/structcomplex.html">complex</a>< T > </span><span><b>cos</b>(const <a href="/thrust/api/classes/structcomplex.html">complex</a>< T > & z);</span></code>
Returns the complex cosine of a <code>complex</code> number.

**Function Parameters**:
**`z`**: The <code>complex</code> argument. 

<h3 id="function-sin">
Function <code>sin</code>
</h3>

<code class="doxybook">
<span>template &lt;typename T&gt;</span>
<span>__host__ __device__ <a href="/thrust/api/classes/structcomplex.html">complex</a>< T > </span><span><b>sin</b>(const <a href="/thrust/api/classes/structcomplex.html">complex</a>< T > & z);</span></code>
Returns the complex sine of a <code>complex</code> number.

**Function Parameters**:
**`z`**: The <code>complex</code> argument. 

<h3 id="function-tan">
Function <code>tan</code>
</h3>

<code class="doxybook">
<span>template &lt;typename T&gt;</span>
<span>__host__ __device__ <a href="/thrust/api/classes/structcomplex.html">complex</a>< T > </span><span><b>tan</b>(const <a href="/thrust/api/classes/structcomplex.html">complex</a>< T > & z);</span></code>
Returns the complex tangent of a <code>complex</code> number.

**Function Parameters**:
**`z`**: The <code>complex</code> argument. 

<h3 id="function-cosh">
Function <code>cosh</code>
</h3>

<code class="doxybook">
<span>template &lt;typename T&gt;</span>
<span>__host__ __device__ <a href="/thrust/api/classes/structcomplex.html">complex</a>< T > </span><span><b>cosh</b>(const <a href="/thrust/api/classes/structcomplex.html">complex</a>< T > & z);</span></code>
Returns the complex hyperbolic cosine of a <code>complex</code> number.

**Function Parameters**:
**`z`**: The <code>complex</code> argument. 

<h3 id="function-sinh">
Function <code>sinh</code>
</h3>

<code class="doxybook">
<span>template &lt;typename T&gt;</span>
<span>__host__ __device__ <a href="/thrust/api/classes/structcomplex.html">complex</a>< T > </span><span><b>sinh</b>(const <a href="/thrust/api/classes/structcomplex.html">complex</a>< T > & z);</span></code>
Returns the complex hyperbolic sine of a <code>complex</code> number.

**Function Parameters**:
**`z`**: The <code>complex</code> argument. 

<h3 id="function-tanh">
Function <code>tanh</code>
</h3>

<code class="doxybook">
<span>template &lt;typename T&gt;</span>
<span>__host__ __device__ <a href="/thrust/api/classes/structcomplex.html">complex</a>< T > </span><span><b>tanh</b>(const <a href="/thrust/api/classes/structcomplex.html">complex</a>< T > & z);</span></code>
Returns the complex hyperbolic tangent of a <code>complex</code> number.

**Function Parameters**:
**`z`**: The <code>complex</code> argument. 

<h3 id="function-acos">
Function <code>acos</code>
</h3>

<code class="doxybook">
<span>template &lt;typename T&gt;</span>
<span>__host__ __device__ <a href="/thrust/api/classes/structcomplex.html">complex</a>< T > </span><span><b>acos</b>(const <a href="/thrust/api/classes/structcomplex.html">complex</a>< T > & z);</span></code>
Returns the complex arc cosine of a <code>complex</code> number.

The range of the real part of the result is [0, Pi] and the range of the imaginary part is [-inf, +inf]

**Function Parameters**:
**`z`**: The <code>complex</code> argument. 

<h3 id="function-asin">
Function <code>asin</code>
</h3>

<code class="doxybook">
<span>template &lt;typename T&gt;</span>
<span>__host__ __device__ <a href="/thrust/api/classes/structcomplex.html">complex</a>< T > </span><span><b>asin</b>(const <a href="/thrust/api/classes/structcomplex.html">complex</a>< T > & z);</span></code>
Returns the complex arc sine of a <code>complex</code> number.

The range of the real part of the result is [-Pi/2, Pi/2] and the range of the imaginary part is [-inf, +inf]

**Function Parameters**:
**`z`**: The <code>complex</code> argument. 

<h3 id="function-atan">
Function <code>atan</code>
</h3>

<code class="doxybook">
<span>template &lt;typename T&gt;</span>
<span>__host__ __device__ <a href="/thrust/api/classes/structcomplex.html">complex</a>< T > </span><span><b>atan</b>(const <a href="/thrust/api/classes/structcomplex.html">complex</a>< T > & z);</span></code>
Returns the complex arc tangent of a <code>complex</code> number.

The range of the real part of the result is [-Pi/2, Pi/2] and the range of the imaginary part is [-inf, +inf]

**Function Parameters**:
**`z`**: The <code>complex</code> argument. 

<h3 id="function-acosh">
Function <code>acosh</code>
</h3>

<code class="doxybook">
<span>template &lt;typename T&gt;</span>
<span>__host__ __device__ <a href="/thrust/api/classes/structcomplex.html">complex</a>< T > </span><span><b>acosh</b>(const <a href="/thrust/api/classes/structcomplex.html">complex</a>< T > & z);</span></code>
Returns the complex inverse hyperbolic cosine of a <code>complex</code> number.

The range of the real part of the result is [0, +inf] and the range of the imaginary part is [-Pi, Pi]

**Function Parameters**:
**`z`**: The <code>complex</code> argument. 

<h3 id="function-asinh">
Function <code>asinh</code>
</h3>

<code class="doxybook">
<span>template &lt;typename T&gt;</span>
<span>__host__ __device__ <a href="/thrust/api/classes/structcomplex.html">complex</a>< T > </span><span><b>asinh</b>(const <a href="/thrust/api/classes/structcomplex.html">complex</a>< T > & z);</span></code>
Returns the complex inverse hyperbolic sine of a <code>complex</code> number.

The range of the real part of the result is [-inf, +inf] and the range of the imaginary part is [-Pi/2, Pi/2]

**Function Parameters**:
**`z`**: The <code>complex</code> argument. 

<h3 id="function-atanh">
Function <code>atanh</code>
</h3>

<code class="doxybook">
<span>template &lt;typename T&gt;</span>
<span>__host__ __device__ <a href="/thrust/api/classes/structcomplex.html">complex</a>< T > </span><span><b>atanh</b>(const <a href="/thrust/api/classes/structcomplex.html">complex</a>< T > & z);</span></code>
Returns the complex inverse hyperbolic tangent of a <code>complex</code> number.

The range of the real part of the result is [-inf, +inf] and the range of the imaginary part is [-Pi/2, Pi/2]

**Function Parameters**:
**`z`**: The <code>complex</code> argument. 

<h3 id="function-operator<<">
Function <code>operator&lt;&lt;</code>
</h3>

<code class="doxybook">
<span>template &lt;typename T,</span>
<span>&nbsp;&nbsp;typename CharT,</span>
<span>&nbsp;&nbsp;typename Traits&gt;</span>
<span>std::basic_ostream< CharT, Traits > & </span><span><b>operator<<</b>(std::basic_ostream< CharT, Traits > & os,</span>
<span>&nbsp;&nbsp;const <a href="/thrust/api/classes/structcomplex.html">complex</a>< T > & z);</span></code>
Writes to an output stream a <code>complex</code> number in the form (real, imaginary).

**Function Parameters**:
* **`os`** The output stream. 
* **`z`** The <code>complex</code> number to output. 

<h3 id="function-operator>>">
Function <code>operator&gt;&gt;</code>
</h3>

<code class="doxybook">
<span>template &lt;typename T,</span>
<span>&nbsp;&nbsp;typename CharT,</span>
<span>&nbsp;&nbsp;typename Traits&gt;</span>
<span>__host__ std::basic_istream< CharT, Traits > & </span><span><b>operator>></b>(std::basic_istream< CharT, Traits > & is,</span>
<span>&nbsp;&nbsp;<a href="/thrust/api/classes/structcomplex.html">complex</a>< T > & z);</span></code>
Reads a <code>complex</code> number from an input stream.

The recognized formats are:

* real
* (real)
* (real, imaginary)
The values read must be convertible to the <code>complex's</code><code>value&#95;type</code>

**Function Parameters**:
* **`is`** The input stream. 
* **`z`** The <code>complex</code> number to set. 

<h3 id="function-operator==">
Function <code>operator==</code>
</h3>

<code class="doxybook">
<span>template &lt;typename T0,</span>
<span>&nbsp;&nbsp;typename T1&gt;</span>
<span>__host__ __device__ bool </span><span><b>operator==</b>(const <a href="/thrust/api/classes/structcomplex.html">complex</a>< T0 > & x,</span>
<span>&nbsp;&nbsp;const <a href="/thrust/api/classes/structcomplex.html">complex</a>< T1 > & y);</span></code>
Returns true if two <code>complex</code> numbers are equal and false otherwise.

**Function Parameters**:
* **`x`** The first <code>complex</code>. 
* **`y`** The second <code>complex</code>. 

<h3 id="function-operator==">
Function <code>operator==</code>
</h3>

<code class="doxybook">
<span>template &lt;typename T0,</span>
<span>&nbsp;&nbsp;typename T1&gt;</span>
<span>__host__ __device__ bool </span><span><b>operator==</b>(const <a href="/thrust/api/classes/structcomplex.html">complex</a>< T0 > & x,</span>
<span>&nbsp;&nbsp;const std::complex< T1 > & y);</span></code>
Returns true if two <code>complex</code> numbers are equal and false otherwise.

**Function Parameters**:
* **`x`** The first <code>complex</code>. 
* **`y`** The second <code>complex</code>. 

<h3 id="function-operator==">
Function <code>operator==</code>
</h3>

<code class="doxybook">
<span>template &lt;typename T0,</span>
<span>&nbsp;&nbsp;typename T1&gt;</span>
<span>__host__ __device__ bool </span><span><b>operator==</b>(const std::complex< T0 > & x,</span>
<span>&nbsp;&nbsp;const <a href="/thrust/api/classes/structcomplex.html">complex</a>< T1 > & y);</span></code>
Returns true if two <code>complex</code> numbers are equal and false otherwise.

**Function Parameters**:
* **`x`** The first <code>complex</code>. 
* **`y`** The second <code>complex</code>. 

<h3 id="function-operator==">
Function <code>operator==</code>
</h3>

<code class="doxybook">
<span>template &lt;typename T0,</span>
<span>&nbsp;&nbsp;typename T1&gt;</span>
<span>__host__ __device__ bool </span><span><b>operator==</b>(const T0 & x,</span>
<span>&nbsp;&nbsp;const <a href="/thrust/api/classes/structcomplex.html">complex</a>< T1 > & y);</span></code>
Returns true if the imaginary part of the <code>complex</code> number is zero and the real part is equal to the scalar. Returns false otherwise.

**Function Parameters**:
* **`x`** The scalar. 
* **`y`** The <code>complex</code>. 

<h3 id="function-operator==">
Function <code>operator==</code>
</h3>

<code class="doxybook">
<span>template &lt;typename T0,</span>
<span>&nbsp;&nbsp;typename T1&gt;</span>
<span>__host__ __device__ bool </span><span><b>operator==</b>(const <a href="/thrust/api/classes/structcomplex.html">complex</a>< T0 > & x,</span>
<span>&nbsp;&nbsp;const T1 & y);</span></code>
Returns true if the imaginary part of the <code>complex</code> number is zero and the real part is equal to the scalar. Returns false otherwise.

**Function Parameters**:
* **`x`** The <code>complex</code>. 
* **`y`** The scalar. 

<h3 id="function-operator!=">
Function <code>operator!=</code>
</h3>

<code class="doxybook">
<span>template &lt;typename T0,</span>
<span>&nbsp;&nbsp;typename T1&gt;</span>
<span>__host__ __device__ bool </span><span><b>operator!=</b>(const <a href="/thrust/api/classes/structcomplex.html">complex</a>< T0 > & x,</span>
<span>&nbsp;&nbsp;const <a href="/thrust/api/classes/structcomplex.html">complex</a>< T1 > & y);</span></code>
Returns true if two <code>complex</code> numbers are different and false otherwise.

**Function Parameters**:
* **`x`** The first <code>complex</code>. 
* **`y`** The second <code>complex</code>. 

<h3 id="function-operator!=">
Function <code>operator!=</code>
</h3>

<code class="doxybook">
<span>template &lt;typename T0,</span>
<span>&nbsp;&nbsp;typename T1&gt;</span>
<span>__host__ __device__ bool </span><span><b>operator!=</b>(const <a href="/thrust/api/classes/structcomplex.html">complex</a>< T0 > & x,</span>
<span>&nbsp;&nbsp;const std::complex< T1 > & y);</span></code>
Returns true if two <code>complex</code> numbers are different and false otherwise.

**Function Parameters**:
* **`x`** The first <code>complex</code>. 
* **`y`** The second <code>complex</code>. 

<h3 id="function-operator!=">
Function <code>operator!=</code>
</h3>

<code class="doxybook">
<span>template &lt;typename T0,</span>
<span>&nbsp;&nbsp;typename T1&gt;</span>
<span>__host__ __device__ bool </span><span><b>operator!=</b>(const std::complex< T0 > & x,</span>
<span>&nbsp;&nbsp;const <a href="/thrust/api/classes/structcomplex.html">complex</a>< T1 > & y);</span></code>
Returns true if two <code>complex</code> numbers are different and false otherwise.

**Function Parameters**:
* **`x`** The first <code>complex</code>. 
* **`y`** The second <code>complex</code>. 

<h3 id="function-operator!=">
Function <code>operator!=</code>
</h3>

<code class="doxybook">
<span>template &lt;typename T0,</span>
<span>&nbsp;&nbsp;typename T1&gt;</span>
<span>__host__ __device__ bool </span><span><b>operator!=</b>(const T0 & x,</span>
<span>&nbsp;&nbsp;const <a href="/thrust/api/classes/structcomplex.html">complex</a>< T1 > & y);</span></code>
Returns true if the imaginary part of the <code>complex</code> number is not zero or the real part is different from the scalar. Returns false otherwise.

**Function Parameters**:
* **`x`** The scalar. 
* **`y`** The <code>complex</code>. 

<h3 id="function-operator!=">
Function <code>operator!=</code>
</h3>

<code class="doxybook">
<span>template &lt;typename T0,</span>
<span>&nbsp;&nbsp;typename T1&gt;</span>
<span>__host__ __device__ bool </span><span><b>operator!=</b>(const <a href="/thrust/api/classes/structcomplex.html">complex</a>< T0 > & x,</span>
<span>&nbsp;&nbsp;const T1 & y);</span></code>
Returns true if the imaginary part of the <code>complex</code> number is not zero or the real part is different from the scalar. Returns false otherwise.

**Function Parameters**:
* **`x`** The <code>complex</code>. 
* **`y`** The scalar. 


