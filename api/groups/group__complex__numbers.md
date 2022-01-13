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
<span>struct <b><a href="{{ site.baseurl }}/api/classes/structthrust_1_1complex.html">thrust::complex</a></b>;</span>
<br>
<span>typedef <i>see below</i> <b><a href="{{ site.baseurl }}/api/groups/group__complex__numbers.html#typedef-value-type">thrust::complex::value&#95;type</a></b>;</span>
<br>
<span>__host__ __device__ </span><span><b><a href="{{ site.baseurl }}/api/groups/group__complex__numbers.html#function-complex">thrust::complex::complex</a></b>(const T & re);</span>
<br>
<span>__host__ __device__ </span><span><b><a href="{{ site.baseurl }}/api/groups/group__complex__numbers.html#function-complex">thrust::complex::complex</a></b>(const T & re,</span>
<span>&nbsp;&nbsp;const T & im);</span>
<br>
<span>template &lt;typename U&gt;</span>
<span>__host__ __device__ </span><span><b><a href="{{ site.baseurl }}/api/groups/group__complex__numbers.html#function-complex">thrust::complex::complex</a></b>(const complex< U > & z);</span>
<br>
<span>__host__ __device__ </span><span><b><a href="{{ site.baseurl }}/api/groups/group__complex__numbers.html#function-complex">thrust::complex::complex</a></b>(const std::complex< T > & z);</span>
<br>
<span>template &lt;typename U&gt;</span>
<span>__host__ __device__ </span><span><b><a href="{{ site.baseurl }}/api/groups/group__complex__numbers.html#function-complex">thrust::complex::complex</a></b>(const std::complex< U > & z);</span>
<br>
<span>__host__ __device__ complex & </span><span><b><a href="{{ site.baseurl }}/api/groups/group__complex__numbers.html#function-operator=">thrust::complex::operator=</a></b>(const T & re);</span>
<br>
<span>complex & </span><span><b><a href="{{ site.baseurl }}/api/groups/group__complex__numbers.html#function-operator=">thrust::complex::operator=</a></b>(const complex< T > & z) = default;</span>
<br>
<span>template &lt;typename U&gt;</span>
<span>__host__ __device__ complex & </span><span><b><a href="{{ site.baseurl }}/api/groups/group__complex__numbers.html#function-operator=">thrust::complex::operator=</a></b>(const complex< U > & z);</span>
<br>
<span>__host__ __device__ complex & </span><span><b><a href="{{ site.baseurl }}/api/groups/group__complex__numbers.html#function-operator=">thrust::complex::operator=</a></b>(const std::complex< T > & z);</span>
<br>
<span>template &lt;typename U&gt;</span>
<span>__host__ __device__ complex & </span><span><b><a href="{{ site.baseurl }}/api/groups/group__complex__numbers.html#function-operator=">thrust::complex::operator=</a></b>(const std::complex< U > & z);</span>
<br>
<span>template &lt;typename U&gt;</span>
<span>__host__ __device__ complex< T > & </span><span><b><a href="{{ site.baseurl }}/api/groups/group__complex__numbers.html#function-operator+=">thrust::complex::operator+=</a></b>(const complex< U > & z);</span>
<br>
<span>template &lt;typename U&gt;</span>
<span>__host__ __device__ complex< T > & </span><span><b><a href="{{ site.baseurl }}/api/groups/group__complex__numbers.html#function-operator-=">thrust::complex::operator-=</a></b>(const complex< U > & z);</span>
<br>
<span>template &lt;typename U&gt;</span>
<span>__host__ __device__ complex< T > & </span><span><b><a href="{{ site.baseurl }}/api/groups/group__complex__numbers.html#function-operator*=">thrust::complex::operator&#42;=</a></b>(const complex< U > & z);</span>
<br>
<span>template &lt;typename U&gt;</span>
<span>__host__ __device__ complex< T > & </span><span><b><a href="{{ site.baseurl }}/api/groups/group__complex__numbers.html#function-operator/=">thrust::complex::operator/=</a></b>(const complex< U > & z);</span>
<br>
<span>template &lt;typename U&gt;</span>
<span>__host__ __device__ complex< T > & </span><span><b><a href="{{ site.baseurl }}/api/groups/group__complex__numbers.html#function-operator+=">thrust::complex::operator+=</a></b>(const U & z);</span>
<br>
<span>template &lt;typename U&gt;</span>
<span>__host__ __device__ complex< T > & </span><span><b><a href="{{ site.baseurl }}/api/groups/group__complex__numbers.html#function-operator-=">thrust::complex::operator-=</a></b>(const U & z);</span>
<br>
<span>template &lt;typename U&gt;</span>
<span>__host__ __device__ complex< T > & </span><span><b><a href="{{ site.baseurl }}/api/groups/group__complex__numbers.html#function-operator*=">thrust::complex::operator&#42;=</a></b>(const U & z);</span>
<br>
<span>template &lt;typename U&gt;</span>
<span>__host__ __device__ complex< T > & </span><span><b><a href="{{ site.baseurl }}/api/groups/group__complex__numbers.html#function-operator/=">thrust::complex::operator/=</a></b>(const U & z);</span>
<br>
<span>__host__ __device__ T </span><span><b><a href="{{ site.baseurl }}/api/groups/group__complex__numbers.html#function-real">thrust::complex::real</a></b>() const;</span>
<br>
<span>__host__ __device__ T </span><span><b><a href="{{ site.baseurl }}/api/groups/group__complex__numbers.html#function-imag">thrust::complex::imag</a></b>() const;</span>
<br>
<span>__host__ __device__ T </span><span><b><a href="{{ site.baseurl }}/api/groups/group__complex__numbers.html#function-real">thrust::complex::real</a></b>() const;</span>
<br>
<span>__host__ __device__ T </span><span><b><a href="{{ site.baseurl }}/api/groups/group__complex__numbers.html#function-imag">thrust::complex::imag</a></b>() const;</span>
<br>
<span>__host__ __device__ void </span><span><b><a href="{{ site.baseurl }}/api/groups/group__complex__numbers.html#function-real">thrust::complex::real</a></b>(T re);</span>
<br>
<span>__host__ __device__ void </span><span><b><a href="{{ site.baseurl }}/api/groups/group__complex__numbers.html#function-imag">thrust::complex::imag</a></b>(T im);</span>
<br>
<span>__host__ __device__ void </span><span><b><a href="{{ site.baseurl }}/api/groups/group__complex__numbers.html#function-real">thrust::complex::real</a></b>(T re);</span>
<br>
<span>__host__ __device__ void </span><span><b><a href="{{ site.baseurl }}/api/groups/group__complex__numbers.html#function-imag">thrust::complex::imag</a></b>(T im);</span>
<br>
<span>__host__ </span><span><b><a href="{{ site.baseurl }}/api/groups/group__complex__numbers.html#function-operator-stdcomplex<-t->">thrust::complex::complex&lt; T &gt;</a></b>() const;</span>
<br>
<span>template &lt;typename T&gt;</span>
<span>__host__ __device__ T </span><span><b><a href="{{ site.baseurl }}/api/groups/group__complex__numbers.html#function-abs">thrust::abs</a></b>(const complex< T > & z);</span>
<br>
<span>template &lt;typename T&gt;</span>
<span>__host__ __device__ T </span><span><b><a href="{{ site.baseurl }}/api/groups/group__complex__numbers.html#function-arg">thrust::arg</a></b>(const complex< T > & z);</span>
<br>
<span>template &lt;typename T&gt;</span>
<span>__host__ __device__ T </span><span><b><a href="{{ site.baseurl }}/api/groups/group__complex__numbers.html#function-norm">thrust::norm</a></b>(const complex< T > & z);</span>
<br>
<span>template &lt;typename T&gt;</span>
<span>__host__ __device__ complex< T > </span><span><b><a href="{{ site.baseurl }}/api/groups/group__complex__numbers.html#function-conj">thrust::conj</a></b>(const complex< T > & z);</span>
<br>
<span>template &lt;typename T0,</span>
<span>&nbsp;&nbsp;typename T1&gt;</span>
<span>__host__ __device__ complex< typename detail::promoted_numerical_type< T0, T1 >::type > </span><span><b><a href="{{ site.baseurl }}/api/groups/group__complex__numbers.html#function-polar">thrust::polar</a></b>(const T0 & m,</span>
<span>&nbsp;&nbsp;const T1 & theta = T1());</span>
<br>
<span>template &lt;typename T&gt;</span>
<span>__host__ __device__ complex< T > </span><span><b><a href="{{ site.baseurl }}/api/groups/group__complex__numbers.html#function-proj">thrust::proj</a></b>(const T & z);</span>
<br>
<span>template &lt;typename T0,</span>
<span>&nbsp;&nbsp;typename T1&gt;</span>
<span>__host__ __device__ complex< typename detail::promoted_numerical_type< T0, T1 >::type > </span><span><b><a href="{{ site.baseurl }}/api/groups/group__complex__numbers.html#function-operator+">thrust::operator+</a></b>(const complex< T0 > & x,</span>
<span>&nbsp;&nbsp;const complex< T1 > & y);</span>
<br>
<span>template &lt;typename T0,</span>
<span>&nbsp;&nbsp;typename T1&gt;</span>
<span>__host__ __device__ complex< typename detail::promoted_numerical_type< T0, T1 >::type > </span><span><b><a href="{{ site.baseurl }}/api/groups/group__complex__numbers.html#function-operator+">thrust::operator+</a></b>(const complex< T0 > & x,</span>
<span>&nbsp;&nbsp;const T1 & y);</span>
<br>
<span>template &lt;typename T0,</span>
<span>&nbsp;&nbsp;typename T1&gt;</span>
<span>__host__ __device__ complex< typename detail::promoted_numerical_type< T0, T1 >::type > </span><span><b><a href="{{ site.baseurl }}/api/groups/group__complex__numbers.html#function-operator+">thrust::operator+</a></b>(const T0 & x,</span>
<span>&nbsp;&nbsp;const complex< T1 > & y);</span>
<br>
<span>template &lt;typename T0,</span>
<span>&nbsp;&nbsp;typename T1&gt;</span>
<span>__host__ __device__ complex< typename detail::promoted_numerical_type< T0, T1 >::type > </span><span><b><a href="{{ site.baseurl }}/api/groups/group__complex__numbers.html#function-operator-">thrust::operator-</a></b>(const complex< T0 > & x,</span>
<span>&nbsp;&nbsp;const complex< T1 > & y);</span>
<br>
<span>template &lt;typename T0,</span>
<span>&nbsp;&nbsp;typename T1&gt;</span>
<span>__host__ __device__ complex< typename detail::promoted_numerical_type< T0, T1 >::type > </span><span><b><a href="{{ site.baseurl }}/api/groups/group__complex__numbers.html#function-operator-">thrust::operator-</a></b>(const complex< T0 > & x,</span>
<span>&nbsp;&nbsp;const T1 & y);</span>
<br>
<span>template &lt;typename T0,</span>
<span>&nbsp;&nbsp;typename T1&gt;</span>
<span>__host__ __device__ complex< typename detail::promoted_numerical_type< T0, T1 >::type > </span><span><b><a href="{{ site.baseurl }}/api/groups/group__complex__numbers.html#function-operator-">thrust::operator-</a></b>(const T0 & x,</span>
<span>&nbsp;&nbsp;const complex< T1 > & y);</span>
<br>
<span>template &lt;typename T0,</span>
<span>&nbsp;&nbsp;typename T1&gt;</span>
<span>__host__ __device__ complex< typename detail::promoted_numerical_type< T0, T1 >::type > </span><span><b><a href="{{ site.baseurl }}/api/groups/group__complex__numbers.html#function-operator*">thrust::operator&#42;</a></b>(const complex< T0 > & x,</span>
<span>&nbsp;&nbsp;const complex< T1 > & y);</span>
<br>
<span>template &lt;typename T0,</span>
<span>&nbsp;&nbsp;typename T1&gt;</span>
<span>__host__ __device__ complex< typename detail::promoted_numerical_type< T0, T1 >::type > </span><span><b><a href="{{ site.baseurl }}/api/groups/group__complex__numbers.html#function-operator*">thrust::operator&#42;</a></b>(const complex< T0 > & x,</span>
<span>&nbsp;&nbsp;const T1 & y);</span>
<br>
<span>template &lt;typename T0,</span>
<span>&nbsp;&nbsp;typename T1&gt;</span>
<span>__host__ __device__ complex< typename detail::promoted_numerical_type< T0, T1 >::type > </span><span><b><a href="{{ site.baseurl }}/api/groups/group__complex__numbers.html#function-operator*">thrust::operator&#42;</a></b>(const T0 & x,</span>
<span>&nbsp;&nbsp;const complex< T1 > & y);</span>
<br>
<span>template &lt;typename T0,</span>
<span>&nbsp;&nbsp;typename T1&gt;</span>
<span>__host__ __device__ complex< typename detail::promoted_numerical_type< T0, T1 >::type > </span><span><b><a href="{{ site.baseurl }}/api/groups/group__complex__numbers.html#function-operator/">thrust::operator/</a></b>(const complex< T0 > & x,</span>
<span>&nbsp;&nbsp;const complex< T1 > & y);</span>
<br>
<span>template &lt;typename T0,</span>
<span>&nbsp;&nbsp;typename T1&gt;</span>
<span>__host__ __device__ complex< typename detail::promoted_numerical_type< T0, T1 >::type > </span><span><b><a href="{{ site.baseurl }}/api/groups/group__complex__numbers.html#function-operator/">thrust::operator/</a></b>(const complex< T0 > & x,</span>
<span>&nbsp;&nbsp;const T1 & y);</span>
<br>
<span>template &lt;typename T0,</span>
<span>&nbsp;&nbsp;typename T1&gt;</span>
<span>__host__ __device__ complex< typename detail::promoted_numerical_type< T0, T1 >::type > </span><span><b><a href="{{ site.baseurl }}/api/groups/group__complex__numbers.html#function-operator/">thrust::operator/</a></b>(const T0 & x,</span>
<span>&nbsp;&nbsp;const complex< T1 > & y);</span>
<br>
<span>template &lt;typename T&gt;</span>
<span>__host__ __device__ complex< T > </span><span><b><a href="{{ site.baseurl }}/api/groups/group__complex__numbers.html#function-operator+">thrust::operator+</a></b>(const complex< T > & y);</span>
<br>
<span>template &lt;typename T&gt;</span>
<span>__host__ __device__ complex< T > </span><span><b><a href="{{ site.baseurl }}/api/groups/group__complex__numbers.html#function-operator-">thrust::operator-</a></b>(const complex< T > & y);</span>
<br>
<span>template &lt;typename T&gt;</span>
<span>__host__ __device__ complex< T > </span><span><b><a href="{{ site.baseurl }}/api/groups/group__complex__numbers.html#function-exp">thrust::exp</a></b>(const complex< T > & z);</span>
<br>
<span>template &lt;typename T&gt;</span>
<span>__host__ __device__ complex< T > </span><span><b><a href="{{ site.baseurl }}/api/groups/group__complex__numbers.html#function-log">thrust::log</a></b>(const complex< T > & z);</span>
<br>
<span>template &lt;typename T&gt;</span>
<span>__host__ __device__ complex< T > </span><span><b><a href="{{ site.baseurl }}/api/groups/group__complex__numbers.html#function-log10">thrust::log10</a></b>(const complex< T > & z);</span>
<br>
<span>template &lt;typename T0,</span>
<span>&nbsp;&nbsp;typename T1&gt;</span>
<span>__host__ __device__ complex< typename detail::promoted_numerical_type< T0, T1 >::type > </span><span><b><a href="{{ site.baseurl }}/api/groups/group__complex__numbers.html#function-pow">thrust::pow</a></b>(const complex< T0 > & x,</span>
<span>&nbsp;&nbsp;const complex< T1 > & y);</span>
<br>
<span>template &lt;typename T0,</span>
<span>&nbsp;&nbsp;typename T1&gt;</span>
<span>__host__ __device__ complex< typename detail::promoted_numerical_type< T0, T1 >::type > </span><span><b><a href="{{ site.baseurl }}/api/groups/group__complex__numbers.html#function-pow">thrust::pow</a></b>(const complex< T0 > & x,</span>
<span>&nbsp;&nbsp;const T1 & y);</span>
<br>
<span>template &lt;typename T0,</span>
<span>&nbsp;&nbsp;typename T1&gt;</span>
<span>__host__ __device__ complex< typename detail::promoted_numerical_type< T0, T1 >::type > </span><span><b><a href="{{ site.baseurl }}/api/groups/group__complex__numbers.html#function-pow">thrust::pow</a></b>(const T0 & x,</span>
<span>&nbsp;&nbsp;const complex< T1 > & y);</span>
<br>
<span>template &lt;typename T&gt;</span>
<span>__host__ __device__ complex< T > </span><span><b><a href="{{ site.baseurl }}/api/groups/group__complex__numbers.html#function-sqrt">thrust::sqrt</a></b>(const complex< T > & z);</span>
<br>
<span>template &lt;typename T&gt;</span>
<span>__host__ __device__ complex< T > </span><span><b><a href="{{ site.baseurl }}/api/groups/group__complex__numbers.html#function-cos">thrust::cos</a></b>(const complex< T > & z);</span>
<br>
<span>template &lt;typename T&gt;</span>
<span>__host__ __device__ complex< T > </span><span><b><a href="{{ site.baseurl }}/api/groups/group__complex__numbers.html#function-sin">thrust::sin</a></b>(const complex< T > & z);</span>
<br>
<span>template &lt;typename T&gt;</span>
<span>__host__ __device__ complex< T > </span><span><b><a href="{{ site.baseurl }}/api/groups/group__complex__numbers.html#function-tan">thrust::tan</a></b>(const complex< T > & z);</span>
<br>
<span>template &lt;typename T&gt;</span>
<span>__host__ __device__ complex< T > </span><span><b><a href="{{ site.baseurl }}/api/groups/group__complex__numbers.html#function-cosh">thrust::cosh</a></b>(const complex< T > & z);</span>
<br>
<span>template &lt;typename T&gt;</span>
<span>__host__ __device__ complex< T > </span><span><b><a href="{{ site.baseurl }}/api/groups/group__complex__numbers.html#function-sinh">thrust::sinh</a></b>(const complex< T > & z);</span>
<br>
<span>template &lt;typename T&gt;</span>
<span>__host__ __device__ complex< T > </span><span><b><a href="{{ site.baseurl }}/api/groups/group__complex__numbers.html#function-tanh">thrust::tanh</a></b>(const complex< T > & z);</span>
<br>
<span>template &lt;typename T&gt;</span>
<span>__host__ __device__ complex< T > </span><span><b><a href="{{ site.baseurl }}/api/groups/group__complex__numbers.html#function-acos">thrust::acos</a></b>(const complex< T > & z);</span>
<br>
<span>template &lt;typename T&gt;</span>
<span>__host__ __device__ complex< T > </span><span><b><a href="{{ site.baseurl }}/api/groups/group__complex__numbers.html#function-asin">thrust::asin</a></b>(const complex< T > & z);</span>
<br>
<span>template &lt;typename T&gt;</span>
<span>__host__ __device__ complex< T > </span><span><b><a href="{{ site.baseurl }}/api/groups/group__complex__numbers.html#function-atan">thrust::atan</a></b>(const complex< T > & z);</span>
<br>
<span>template &lt;typename T&gt;</span>
<span>__host__ __device__ complex< T > </span><span><b><a href="{{ site.baseurl }}/api/groups/group__complex__numbers.html#function-acosh">thrust::acosh</a></b>(const complex< T > & z);</span>
<br>
<span>template &lt;typename T&gt;</span>
<span>__host__ __device__ complex< T > </span><span><b><a href="{{ site.baseurl }}/api/groups/group__complex__numbers.html#function-asinh">thrust::asinh</a></b>(const complex< T > & z);</span>
<br>
<span>template &lt;typename T&gt;</span>
<span>__host__ __device__ complex< T > </span><span><b><a href="{{ site.baseurl }}/api/groups/group__complex__numbers.html#function-atanh">thrust::atanh</a></b>(const complex< T > & z);</span>
<br>
<span>template &lt;typename T,</span>
<span>&nbsp;&nbsp;typename CharT,</span>
<span>&nbsp;&nbsp;typename Traits&gt;</span>
<span>std::basic_ostream< CharT, Traits > & </span><span><b><a href="{{ site.baseurl }}/api/groups/group__complex__numbers.html#function-operator<<">thrust::operator&lt;&lt;</a></b>(std::basic_ostream< CharT, Traits > & os,</span>
<span>&nbsp;&nbsp;const complex< T > & z);</span>
<br>
<span>template &lt;typename T,</span>
<span>&nbsp;&nbsp;typename CharT,</span>
<span>&nbsp;&nbsp;typename Traits&gt;</span>
<span>__host__ std::basic_istream< CharT, Traits > & </span><span><b><a href="{{ site.baseurl }}/api/groups/group__complex__numbers.html#function-operator>>">thrust::operator&gt;&gt;</a></b>(std::basic_istream< CharT, Traits > & is,</span>
<span>&nbsp;&nbsp;complex< T > & z);</span>
<br>
<span>template &lt;typename T0,</span>
<span>&nbsp;&nbsp;typename T1&gt;</span>
<span>__host__ __device__ bool </span><span><b><a href="{{ site.baseurl }}/api/groups/group__complex__numbers.html#function-operator==">thrust::operator==</a></b>(const complex< T0 > & x,</span>
<span>&nbsp;&nbsp;const complex< T1 > & y);</span>
<br>
<span>template &lt;typename T0,</span>
<span>&nbsp;&nbsp;typename T1&gt;</span>
<span>__host__ __device__ bool </span><span><b><a href="{{ site.baseurl }}/api/groups/group__complex__numbers.html#function-operator==">thrust::operator==</a></b>(const complex< T0 > & x,</span>
<span>&nbsp;&nbsp;const std::complex< T1 > & y);</span>
<br>
<span>template &lt;typename T0,</span>
<span>&nbsp;&nbsp;typename T1&gt;</span>
<span>__host__ __device__ bool </span><span><b><a href="{{ site.baseurl }}/api/groups/group__complex__numbers.html#function-operator==">thrust::operator==</a></b>(const std::complex< T0 > & x,</span>
<span>&nbsp;&nbsp;const complex< T1 > & y);</span>
<br>
<span>template &lt;typename T0,</span>
<span>&nbsp;&nbsp;typename T1&gt;</span>
<span>__host__ __device__ bool </span><span><b><a href="{{ site.baseurl }}/api/groups/group__complex__numbers.html#function-operator==">thrust::operator==</a></b>(const T0 & x,</span>
<span>&nbsp;&nbsp;const complex< T1 > & y);</span>
<br>
<span>template &lt;typename T0,</span>
<span>&nbsp;&nbsp;typename T1&gt;</span>
<span>__host__ __device__ bool </span><span><b><a href="{{ site.baseurl }}/api/groups/group__complex__numbers.html#function-operator==">thrust::operator==</a></b>(const complex< T0 > & x,</span>
<span>&nbsp;&nbsp;const T1 & y);</span>
<br>
<span>template &lt;typename T0,</span>
<span>&nbsp;&nbsp;typename T1&gt;</span>
<span>__host__ __device__ bool </span><span><b><a href="{{ site.baseurl }}/api/groups/group__complex__numbers.html#function-operator!=">thrust::operator!=</a></b>(const complex< T0 > & x,</span>
<span>&nbsp;&nbsp;const complex< T1 > & y);</span>
<br>
<span>template &lt;typename T0,</span>
<span>&nbsp;&nbsp;typename T1&gt;</span>
<span>__host__ __device__ bool </span><span><b><a href="{{ site.baseurl }}/api/groups/group__complex__numbers.html#function-operator!=">thrust::operator!=</a></b>(const complex< T0 > & x,</span>
<span>&nbsp;&nbsp;const std::complex< T1 > & y);</span>
<br>
<span>template &lt;typename T0,</span>
<span>&nbsp;&nbsp;typename T1&gt;</span>
<span>__host__ __device__ bool </span><span><b><a href="{{ site.baseurl }}/api/groups/group__complex__numbers.html#function-operator!=">thrust::operator!=</a></b>(const std::complex< T0 > & x,</span>
<span>&nbsp;&nbsp;const complex< T1 > & y);</span>
<br>
<span>template &lt;typename T0,</span>
<span>&nbsp;&nbsp;typename T1&gt;</span>
<span>__host__ __device__ bool </span><span><b><a href="{{ site.baseurl }}/api/groups/group__complex__numbers.html#function-operator!=">thrust::operator!=</a></b>(const T0 & x,</span>
<span>&nbsp;&nbsp;const complex< T1 > & y);</span>
<br>
<span>template &lt;typename T0,</span>
<span>&nbsp;&nbsp;typename T1&gt;</span>
<span>__host__ __device__ bool </span><span><b><a href="{{ site.baseurl }}/api/groups/group__complex__numbers.html#function-operator!=">thrust::operator!=</a></b>(const complex< T0 > & x,</span>
<span>&nbsp;&nbsp;const T1 & y);</span>
</code>

## Member Classes

<h3 id="struct-thrustcomplex">
<a href="{{ site.baseurl }}/api/classes/structthrust_1_1complex.html">Struct <code>thrust::complex</code>
</a>
</h3>


## Types

<h3 id="typedef-value-type">
Typedef <code>thrust::complex::value&#95;type</code>
</h3>

<code class="doxybook">
<span>typedef T<b>value_type</b>;</span></code>
<code>value&#95;type</code> is the type of <code>complex's</code> real and imaginary parts. 


## Functions

<h3 id="function-complex">
Function <code>thrust::complex::complex</code>
</h3>

<code class="doxybook">
<span>__host__ __device__ </span><span><b>complex</b>(const T & re);</span></code>
Construct a complex number with an imaginary part of 0.

**Function Parameters**:
**`re`**: The real part of the number. 

<h3 id="function-complex">
Function <code>thrust::complex::complex</code>
</h3>

<code class="doxybook">
<span>__host__ __device__ </span><span><b>complex</b>(const T & re,</span>
<span>&nbsp;&nbsp;const T & im);</span></code>
Construct a complex number from its real and imaginary parts.

**Function Parameters**:
* **`re`** The real part of the number. 
* **`im`** The imaginary part of the number. 

<h3 id="function-complex">
Function <code>thrust::complex::complex</code>
</h3>

<code class="doxybook">
<span><b>complex</b>() = default;</span></code>
Default construct a complex number. 

<h3 id="function-complex">
Function <code>thrust::complex::complex</code>
</h3>

<code class="doxybook">
<span><b>complex</b>(const complex< T > & z) = default;</span></code>
This copy constructor copies from a <code>complex</code> with a type that is convertible to this <code>complex's</code><code>value&#95;type</code>.

**Function Parameters**:
**`z`**: The <code>complex</code> to copy from. 

<h3 id="function-complex">
Function <code>thrust::complex::complex</code>
</h3>

<code class="doxybook">
<span>template &lt;typename U&gt;</span>
<span>__host__ __device__ </span><span><b>complex</b>(const complex< U > & z);</span></code>
This converting copy constructor copies from a <code>complex</code> with a type that is convertible to this <code>complex's</code><code>value&#95;type</code>.

**Template Parameters**:
**`U`**: is convertible to <code>value&#95;type</code>. 

**Function Parameters**:
**`z`**: The <code>complex</code> to copy from.

<h3 id="function-complex">
Function <code>thrust::complex::complex</code>
</h3>

<code class="doxybook">
<span>__host__ __device__ </span><span><b>complex</b>(const std::complex< T > & z);</span></code>
This converting copy constructor copies from a <code>std::complex</code> with a type that is convertible to this <code>complex's</code><code>value&#95;type</code>.

**Function Parameters**:
**`z`**: The <code>complex</code> to copy from. 

<h3 id="function-complex">
Function <code>thrust::complex::complex</code>
</h3>

<code class="doxybook">
<span>template &lt;typename U&gt;</span>
<span>__host__ __device__ </span><span><b>complex</b>(const std::complex< U > & z);</span></code>
This converting copy constructor copies from a <code>std::complex</code> with a type that is convertible to this <code>complex's</code><code>value&#95;type</code>.

**Template Parameters**:
**`U`**: is convertible to <code>value&#95;type</code>. 

**Function Parameters**:
**`z`**: The <code>complex</code> to copy from.

<h3 id="function-operator=">
Function <code>thrust::complex::operator=</code>
</h3>

<code class="doxybook">
<span>__host__ __device__ complex & </span><span><b>operator=</b>(const T & re);</span></code>
Assign <code>re</code> to the real part of this <code>complex</code> and set the imaginary part to 0.

**Function Parameters**:
**`re`**: The real part of the number. 

<h3 id="function-operator=">
Function <code>thrust::complex::operator=</code>
</h3>

<code class="doxybook">
<span>complex & </span><span><b>operator=</b>(const complex< T > & z) = default;</span></code>
Assign <code>z.real()</code> and <code>z.imag()</code> to the real and imaginary parts of this <code>complex</code> respectively.

**Function Parameters**:
**`z`**: The <code>complex</code> to copy from. 

<h3 id="function-operator=">
Function <code>thrust::complex::operator=</code>
</h3>

<code class="doxybook">
<span>template &lt;typename U&gt;</span>
<span>__host__ __device__ complex & </span><span><b>operator=</b>(const complex< U > & z);</span></code>
Assign <code>z.real()</code> and <code>z.imag()</code> to the real and imaginary parts of this <code>complex</code> respectively.

**Template Parameters**:
**`U`**: is convertible to <code>value&#95;type</code>. 

**Function Parameters**:
**`z`**: The <code>complex</code> to copy from.

<h3 id="function-operator=">
Function <code>thrust::complex::operator=</code>
</h3>

<code class="doxybook">
<span>__host__ __device__ complex & </span><span><b>operator=</b>(const std::complex< T > & z);</span></code>
Assign <code>z.real()</code> and <code>z.imag()</code> to the real and imaginary parts of this <code>complex</code> respectively.

**Function Parameters**:
**`z`**: The <code>complex</code> to copy from. 

<h3 id="function-operator=">
Function <code>thrust::complex::operator=</code>
</h3>

<code class="doxybook">
<span>template &lt;typename U&gt;</span>
<span>__host__ __device__ complex & </span><span><b>operator=</b>(const std::complex< U > & z);</span></code>
Assign <code>z.real()</code> and <code>z.imag()</code> to the real and imaginary parts of this <code>complex</code> respectively.

**Template Parameters**:
**`U`**: is convertible to <code>value&#95;type</code>. 

**Function Parameters**:
**`z`**: The <code>complex</code> to copy from.

<h3 id="function-operator+=">
Function <code>thrust::complex::operator+=</code>
</h3>

<code class="doxybook">
<span>template &lt;typename U&gt;</span>
<span>__host__ __device__ complex< T > & </span><span><b>operator+=</b>(const complex< U > & z);</span></code>
Adds a <code>complex</code> to this <code>complex</code> and assigns the result to this <code>complex</code>.

**Template Parameters**:
**`U`**: is convertible to <code>value&#95;type</code>. 

**Function Parameters**:
**`z`**: The <code>complex</code> to be added.

<h3 id="function-operator-=">
Function <code>thrust::complex::operator-=</code>
</h3>

<code class="doxybook">
<span>template &lt;typename U&gt;</span>
<span>__host__ __device__ complex< T > & </span><span><b>operator-=</b>(const complex< U > & z);</span></code>
Subtracts a <code>complex</code> from this <code>complex</code> and assigns the result to this <code>complex</code>.

**Template Parameters**:
**`U`**: is convertible to <code>value&#95;type</code>. 

**Function Parameters**:
**`z`**: The <code>complex</code> to be subtracted.

<h3 id="function-operator*=">
Function <code>thrust::complex::operator&#42;=</code>
</h3>

<code class="doxybook">
<span>template &lt;typename U&gt;</span>
<span>__host__ __device__ complex< T > & </span><span><b>operator*=</b>(const complex< U > & z);</span></code>
Multiplies this <code>complex</code> by another <code>complex</code> and assigns the result to this <code>complex</code>.

**Template Parameters**:
**`U`**: is convertible to <code>value&#95;type</code>. 

**Function Parameters**:
**`z`**: The <code>complex</code> to be multiplied.

<h3 id="function-operator/=">
Function <code>thrust::complex::operator/=</code>
</h3>

<code class="doxybook">
<span>template &lt;typename U&gt;</span>
<span>__host__ __device__ complex< T > & </span><span><b>operator/=</b>(const complex< U > & z);</span></code>
Divides this <code>complex</code> by another <code>complex</code> and assigns the result to this <code>complex</code>.

**Template Parameters**:
**`U`**: is convertible to <code>value&#95;type</code>. 

**Function Parameters**:
**`z`**: The <code>complex</code> to be divided.

<h3 id="function-operator+=">
Function <code>thrust::complex::operator+=</code>
</h3>

<code class="doxybook">
<span>template &lt;typename U&gt;</span>
<span>__host__ __device__ complex< T > & </span><span><b>operator+=</b>(const U & z);</span></code>
Adds a scalar to this <code>complex</code> and assigns the result to this <code>complex</code>.

**Template Parameters**:
**`U`**: is convertible to <code>value&#95;type</code>. 

**Function Parameters**:
**`z`**: The <code>complex</code> to be added.

<h3 id="function-operator-=">
Function <code>thrust::complex::operator-=</code>
</h3>

<code class="doxybook">
<span>template &lt;typename U&gt;</span>
<span>__host__ __device__ complex< T > & </span><span><b>operator-=</b>(const U & z);</span></code>
Subtracts a scalar from this <code>complex</code> and assigns the result to this <code>complex</code>.

**Template Parameters**:
**`U`**: is convertible to <code>value&#95;type</code>. 

**Function Parameters**:
**`z`**: The scalar to be subtracted.

<h3 id="function-operator*=">
Function <code>thrust::complex::operator&#42;=</code>
</h3>

<code class="doxybook">
<span>template &lt;typename U&gt;</span>
<span>__host__ __device__ complex< T > & </span><span><b>operator*=</b>(const U & z);</span></code>
Multiplies this <code>complex</code> by a scalar and assigns the result to this <code>complex</code>.

**Template Parameters**:
**`U`**: is convertible to <code>value&#95;type</code>. 

**Function Parameters**:
**`z`**: The scalar to be multiplied.

<h3 id="function-operator/=">
Function <code>thrust::complex::operator/=</code>
</h3>

<code class="doxybook">
<span>template &lt;typename U&gt;</span>
<span>__host__ __device__ complex< T > & </span><span><b>operator/=</b>(const U & z);</span></code>
Divides this <code>complex</code> by a scalar and assigns the result to this <code>complex</code>.

**Template Parameters**:
**`U`**: is convertible to <code>value&#95;type</code>. 

**Function Parameters**:
**`z`**: The scalar to be divided.

<h3 id="function-real">
Function <code>thrust::complex::real</code>
</h3>

<code class="doxybook">
<span>__host__ __device__ T </span><span><b>real</b>() const;</span></code>
Returns the real part of this <code>complex</code>. 

<h3 id="function-imag">
Function <code>thrust::complex::imag</code>
</h3>

<code class="doxybook">
<span>__host__ __device__ T </span><span><b>imag</b>() const;</span></code>
Returns the imaginary part of this <code>complex</code>. 

<h3 id="function-real">
Function <code>thrust::complex::real</code>
</h3>

<code class="doxybook">
<span>__host__ __device__ T </span><span><b>real</b>() const;</span></code>
Returns the real part of this <code>complex</code>. 

<h3 id="function-imag">
Function <code>thrust::complex::imag</code>
</h3>

<code class="doxybook">
<span>__host__ __device__ T </span><span><b>imag</b>() const;</span></code>
Returns the imaginary part of this <code>complex</code>. 

<h3 id="function-real">
Function <code>thrust::complex::real</code>
</h3>

<code class="doxybook">
<span>__host__ __device__ void </span><span><b>real</b>(T re);</span></code>
Sets the real part of this <code>complex</code>.

**Function Parameters**:
**`re`**: The new real part of this <code>complex</code>. 

<h3 id="function-imag">
Function <code>thrust::complex::imag</code>
</h3>

<code class="doxybook">
<span>__host__ __device__ void </span><span><b>imag</b>(T im);</span></code>
Sets the imaginary part of this <code>complex</code>.

**Function Parameters**:
**`im`**: The new imaginary part of this <code>complex.e</code>

<h3 id="function-real">
Function <code>thrust::complex::real</code>
</h3>

<code class="doxybook">
<span>__host__ __device__ void </span><span><b>real</b>(T re);</span></code>
Sets the real part of this <code>complex</code>.

**Function Parameters**:
**`re`**: The new real part of this <code>complex</code>. 

<h3 id="function-imag">
Function <code>thrust::complex::imag</code>
</h3>

<code class="doxybook">
<span>__host__ __device__ void </span><span><b>imag</b>(T im);</span></code>
Sets the imaginary part of this <code>complex</code>.

**Function Parameters**:
**`im`**: The new imaginary part of this <code>complex</code>. 

<h3 id="function-operator-stdcomplex<-t->">
Function <code>thrust::complex::complex&lt; T &gt;</code>
</h3>

<code class="doxybook">
<span>__host__ </span><span><b>operator std::complex< T ></b>() const;</span></code>
Casts this <code>complex</code> to a <code>std::complex</code> of the same type. 

<h3 id="function-abs">
Function <code>thrust::abs</code>
</h3>

<code class="doxybook">
<span>template &lt;typename T&gt;</span>
<span>__host__ __device__ T </span><span><b>abs</b>(const complex< T > & z);</span></code>
Returns the magnitude (also known as absolute value) of a <code>complex</code>.

**Function Parameters**:
**`z`**: The <code>complex</code> from which to calculate the absolute value. 

<h3 id="function-arg">
Function <code>thrust::arg</code>
</h3>

<code class="doxybook">
<span>template &lt;typename T&gt;</span>
<span>__host__ __device__ T </span><span><b>arg</b>(const complex< T > & z);</span></code>
Returns the phase angle (also known as argument) in radians of a <code>complex</code>.

**Function Parameters**:
**`z`**: The <code>complex</code> from which to calculate the phase angle. 

<h3 id="function-norm">
Function <code>thrust::norm</code>
</h3>

<code class="doxybook">
<span>template &lt;typename T&gt;</span>
<span>__host__ __device__ T </span><span><b>norm</b>(const complex< T > & z);</span></code>
Returns the square of the magnitude of a <code>complex</code>.

**Function Parameters**:
**`z`**: The <code>complex</code> from which to calculate the norm. 

<h3 id="function-conj">
Function <code>thrust::conj</code>
</h3>

<code class="doxybook">
<span>template &lt;typename T&gt;</span>
<span>__host__ __device__ complex< T > </span><span><b>conj</b>(const complex< T > & z);</span></code>
Returns the complex conjugate of a <code>complex</code>.

**Function Parameters**:
**`z`**: The <code>complex</code> from which to calculate the complex conjugate. 

<h3 id="function-polar">
Function <code>thrust::polar</code>
</h3>

<code class="doxybook">
<span>template &lt;typename T0,</span>
<span>&nbsp;&nbsp;typename T1&gt;</span>
<span>__host__ __device__ complex< typename detail::promoted_numerical_type< T0, T1 >::type > </span><span><b>polar</b>(const T0 & m,</span>
<span>&nbsp;&nbsp;const T1 & theta = T1());</span></code>
Returns a <code>complex</code> with the specified magnitude and phase.

**Function Parameters**:
* **`m`** The magnitude of the returned <code>complex</code>. 
* **`theta`** The phase of the returned <code>complex</code> in radians. 

<h3 id="function-proj">
Function <code>thrust::proj</code>
</h3>

<code class="doxybook">
<span>template &lt;typename T&gt;</span>
<span>__host__ __device__ complex< T > </span><span><b>proj</b>(const T & z);</span></code>
Returns the projection of a <code>complex</code> on the Riemann sphere. For all finite <code>complex</code> it returns the argument. For <code>complexs</code> with a non finite part returns (INFINITY,+/-0) where the sign of the zero matches the sign of the imaginary part of the argument.

**Function Parameters**:
**`z`**: The <code>complex</code> argument. 

<h3 id="function-operator+">
Function <code>thrust::operator+</code>
</h3>

<code class="doxybook">
<span>template &lt;typename T0,</span>
<span>&nbsp;&nbsp;typename T1&gt;</span>
<span>__host__ __device__ complex< typename detail::promoted_numerical_type< T0, T1 >::type > </span><span><b>operator+</b>(const complex< T0 > & x,</span>
<span>&nbsp;&nbsp;const complex< T1 > & y);</span></code>
Adds two <code>complex</code> numbers.

The value types of the two <code>complex</code> types should be compatible and the type of the returned <code>complex</code> is the promoted type of the two arguments.

**Function Parameters**:
* **`x`** The first <code>complex</code>. 
* **`y`** The second <code>complex</code>. 

<h3 id="function-operator+">
Function <code>thrust::operator+</code>
</h3>

<code class="doxybook">
<span>template &lt;typename T0,</span>
<span>&nbsp;&nbsp;typename T1&gt;</span>
<span>__host__ __device__ complex< typename detail::promoted_numerical_type< T0, T1 >::type > </span><span><b>operator+</b>(const complex< T0 > & x,</span>
<span>&nbsp;&nbsp;const T1 & y);</span></code>
Adds a scalar to a <code>complex</code> number.

The value type of the <code>complex</code> should be compatible with the scalar and the type of the returned <code>complex</code> is the promoted type of the two arguments.

**Function Parameters**:
* **`x`** The <code>complex</code>. 
* **`y`** The scalar. 

<h3 id="function-operator+">
Function <code>thrust::operator+</code>
</h3>

<code class="doxybook">
<span>template &lt;typename T0,</span>
<span>&nbsp;&nbsp;typename T1&gt;</span>
<span>__host__ __device__ complex< typename detail::promoted_numerical_type< T0, T1 >::type > </span><span><b>operator+</b>(const T0 & x,</span>
<span>&nbsp;&nbsp;const complex< T1 > & y);</span></code>
Adds a <code>complex</code> number to a scalar.

The value type of the <code>complex</code> should be compatible with the scalar and the type of the returned <code>complex</code> is the promoted type of the two arguments.

**Function Parameters**:
* **`x`** The scalar. 
* **`y`** The <code>complex</code>. 

<h3 id="function-operator-">
Function <code>thrust::operator-</code>
</h3>

<code class="doxybook">
<span>template &lt;typename T0,</span>
<span>&nbsp;&nbsp;typename T1&gt;</span>
<span>__host__ __device__ complex< typename detail::promoted_numerical_type< T0, T1 >::type > </span><span><b>operator-</b>(const complex< T0 > & x,</span>
<span>&nbsp;&nbsp;const complex< T1 > & y);</span></code>
Subtracts two <code>complex</code> numbers.

The value types of the two <code>complex</code> types should be compatible and the type of the returned <code>complex</code> is the promoted type of the two arguments.

**Function Parameters**:
* **`x`** The first <code>complex</code> (minuend). 
* **`y`** The second <code>complex</code> (subtrahend). 

<h3 id="function-operator-">
Function <code>thrust::operator-</code>
</h3>

<code class="doxybook">
<span>template &lt;typename T0,</span>
<span>&nbsp;&nbsp;typename T1&gt;</span>
<span>__host__ __device__ complex< typename detail::promoted_numerical_type< T0, T1 >::type > </span><span><b>operator-</b>(const complex< T0 > & x,</span>
<span>&nbsp;&nbsp;const T1 & y);</span></code>
Subtracts a scalar from a <code>complex</code> number.

The value type of the <code>complex</code> should be compatible with the scalar and the type of the returned <code>complex</code> is the promoted type of the two arguments.

**Function Parameters**:
* **`x`** The <code>complex</code> (minuend). 
* **`y`** The scalar (subtrahend). 

<h3 id="function-operator-">
Function <code>thrust::operator-</code>
</h3>

<code class="doxybook">
<span>template &lt;typename T0,</span>
<span>&nbsp;&nbsp;typename T1&gt;</span>
<span>__host__ __device__ complex< typename detail::promoted_numerical_type< T0, T1 >::type > </span><span><b>operator-</b>(const T0 & x,</span>
<span>&nbsp;&nbsp;const complex< T1 > & y);</span></code>
Subtracts a <code>complex</code> number from a scalar.

The value type of the <code>complex</code> should be compatible with the scalar and the type of the returned <code>complex</code> is the promoted type of the two arguments.

**Function Parameters**:
* **`x`** The scalar (minuend). 
* **`y`** The <code>complex</code> (subtrahend). 

<h3 id="function-operator*">
Function <code>thrust::operator&#42;</code>
</h3>

<code class="doxybook">
<span>template &lt;typename T0,</span>
<span>&nbsp;&nbsp;typename T1&gt;</span>
<span>__host__ __device__ complex< typename detail::promoted_numerical_type< T0, T1 >::type > </span><span><b>operator*</b>(const complex< T0 > & x,</span>
<span>&nbsp;&nbsp;const complex< T1 > & y);</span></code>
Multiplies two <code>complex</code> numbers.

The value types of the two <code>complex</code> types should be compatible and the type of the returned <code>complex</code> is the promoted type of the two arguments.

**Function Parameters**:
* **`x`** The first <code>complex</code>. 
* **`y`** The second <code>complex</code>. 

<h3 id="function-operator*">
Function <code>thrust::operator&#42;</code>
</h3>

<code class="doxybook">
<span>template &lt;typename T0,</span>
<span>&nbsp;&nbsp;typename T1&gt;</span>
<span>__host__ __device__ complex< typename detail::promoted_numerical_type< T0, T1 >::type > </span><span><b>operator*</b>(const complex< T0 > & x,</span>
<span>&nbsp;&nbsp;const T1 & y);</span></code>
Multiplies a <code>complex</code> number by a scalar.

**Function Parameters**:
* **`x`** The <code>complex</code>. 
* **`y`** The scalar. 

<h3 id="function-operator*">
Function <code>thrust::operator&#42;</code>
</h3>

<code class="doxybook">
<span>template &lt;typename T0,</span>
<span>&nbsp;&nbsp;typename T1&gt;</span>
<span>__host__ __device__ complex< typename detail::promoted_numerical_type< T0, T1 >::type > </span><span><b>operator*</b>(const T0 & x,</span>
<span>&nbsp;&nbsp;const complex< T1 > & y);</span></code>
Multiplies a scalar by a <code>complex</code> number.

The value type of the <code>complex</code> should be compatible with the scalar and the type of the returned <code>complex</code> is the promoted type of the two arguments.

**Function Parameters**:
* **`x`** The scalar. 
* **`y`** The <code>complex</code>. 

<h3 id="function-operator/">
Function <code>thrust::operator/</code>
</h3>

<code class="doxybook">
<span>template &lt;typename T0,</span>
<span>&nbsp;&nbsp;typename T1&gt;</span>
<span>__host__ __device__ complex< typename detail::promoted_numerical_type< T0, T1 >::type > </span><span><b>operator/</b>(const complex< T0 > & x,</span>
<span>&nbsp;&nbsp;const complex< T1 > & y);</span></code>
Divides two <code>complex</code> numbers.

The value types of the two <code>complex</code> types should be compatible and the type of the returned <code>complex</code> is the promoted type of the two arguments.

**Function Parameters**:
* **`x`** The numerator (dividend). 
* **`y`** The denomimator (divisor). 

<h3 id="function-operator/">
Function <code>thrust::operator/</code>
</h3>

<code class="doxybook">
<span>template &lt;typename T0,</span>
<span>&nbsp;&nbsp;typename T1&gt;</span>
<span>__host__ __device__ complex< typename detail::promoted_numerical_type< T0, T1 >::type > </span><span><b>operator/</b>(const complex< T0 > & x,</span>
<span>&nbsp;&nbsp;const T1 & y);</span></code>
Divides a <code>complex</code> number by a scalar.

The value type of the <code>complex</code> should be compatible with the scalar and the type of the returned <code>complex</code> is the promoted type of the two arguments.

**Function Parameters**:
* **`x`** The complex numerator (dividend). 
* **`y`** The scalar denomimator (divisor). 

<h3 id="function-operator/">
Function <code>thrust::operator/</code>
</h3>

<code class="doxybook">
<span>template &lt;typename T0,</span>
<span>&nbsp;&nbsp;typename T1&gt;</span>
<span>__host__ __device__ complex< typename detail::promoted_numerical_type< T0, T1 >::type > </span><span><b>operator/</b>(const T0 & x,</span>
<span>&nbsp;&nbsp;const complex< T1 > & y);</span></code>
Divides a scalar by a <code>complex</code> number.

The value type of the <code>complex</code> should be compatible with the scalar and the type of the returned <code>complex</code> is the promoted type of the two arguments.

**Function Parameters**:
* **`x`** The scalar numerator (dividend). 
* **`y`** The complex denomimator (divisor). 

<h3 id="function-operator+">
Function <code>thrust::operator+</code>
</h3>

<code class="doxybook">
<span>template &lt;typename T&gt;</span>
<span>__host__ __device__ complex< T > </span><span><b>operator+</b>(const complex< T > & y);</span></code>
Unary plus, returns its <code>complex</code> argument.

**Function Parameters**:
**`y`**: The <code>complex</code> argument. 

<h3 id="function-operator-">
Function <code>thrust::operator-</code>
</h3>

<code class="doxybook">
<span>template &lt;typename T&gt;</span>
<span>__host__ __device__ complex< T > </span><span><b>operator-</b>(const complex< T > & y);</span></code>
Unary minus, returns the additive inverse (negation) of its <code>complex</code> argument.

**Function Parameters**:
**`y`**: The <code>complex</code> argument. 

<h3 id="function-exp">
Function <code>thrust::exp</code>
</h3>

<code class="doxybook">
<span>template &lt;typename T&gt;</span>
<span>__host__ __device__ complex< T > </span><span><b>exp</b>(const complex< T > & z);</span></code>
Returns the complex exponential of a <code>complex</code> number.

**Function Parameters**:
**`z`**: The <code>complex</code> argument. 

<h3 id="function-log">
Function <code>thrust::log</code>
</h3>

<code class="doxybook">
<span>template &lt;typename T&gt;</span>
<span>__host__ __device__ complex< T > </span><span><b>log</b>(const complex< T > & z);</span></code>
Returns the complex natural logarithm of a <code>complex</code> number.

**Function Parameters**:
**`z`**: The <code>complex</code> argument. 

<h3 id="function-log10">
Function <code>thrust::log10</code>
</h3>

<code class="doxybook">
<span>template &lt;typename T&gt;</span>
<span>__host__ __device__ complex< T > </span><span><b>log10</b>(const complex< T > & z);</span></code>
Returns the complex base 10 logarithm of a <code>complex</code> number.

**Function Parameters**:
**`z`**: The <code>complex</code> argument. 

<h3 id="function-pow">
Function <code>thrust::pow</code>
</h3>

<code class="doxybook">
<span>template &lt;typename T0,</span>
<span>&nbsp;&nbsp;typename T1&gt;</span>
<span>__host__ __device__ complex< typename detail::promoted_numerical_type< T0, T1 >::type > </span><span><b>pow</b>(const complex< T0 > & x,</span>
<span>&nbsp;&nbsp;const complex< T1 > & y);</span></code>
Returns a <code>complex</code> number raised to another.

The value types of the two <code>complex</code> types should be compatible and the type of the returned <code>complex</code> is the promoted type of the two arguments.

**Function Parameters**:
* **`x`** The base. 
* **`y`** The exponent. 

<h3 id="function-pow">
Function <code>thrust::pow</code>
</h3>

<code class="doxybook">
<span>template &lt;typename T0,</span>
<span>&nbsp;&nbsp;typename T1&gt;</span>
<span>__host__ __device__ complex< typename detail::promoted_numerical_type< T0, T1 >::type > </span><span><b>pow</b>(const complex< T0 > & x,</span>
<span>&nbsp;&nbsp;const T1 & y);</span></code>
Returns a <code>complex</code> number raised to a scalar.

The value type of the <code>complex</code> should be compatible with the scalar and the type of the returned <code>complex</code> is the promoted type of the two arguments.

**Function Parameters**:
* **`x`** The base. 
* **`y`** The exponent. 

<h3 id="function-pow">
Function <code>thrust::pow</code>
</h3>

<code class="doxybook">
<span>template &lt;typename T0,</span>
<span>&nbsp;&nbsp;typename T1&gt;</span>
<span>__host__ __device__ complex< typename detail::promoted_numerical_type< T0, T1 >::type > </span><span><b>pow</b>(const T0 & x,</span>
<span>&nbsp;&nbsp;const complex< T1 > & y);</span></code>
Returns a scalar raised to a <code>complex</code> number.

The value type of the <code>complex</code> should be compatible with the scalar and the type of the returned <code>complex</code> is the promoted type of the two arguments.

**Function Parameters**:
* **`x`** The base. 
* **`y`** The exponent. 

<h3 id="function-sqrt">
Function <code>thrust::sqrt</code>
</h3>

<code class="doxybook">
<span>template &lt;typename T&gt;</span>
<span>__host__ __device__ complex< T > </span><span><b>sqrt</b>(const complex< T > & z);</span></code>
Returns the complex square root of a <code>complex</code> number.

**Function Parameters**:
**`z`**: The <code>complex</code> argument. 

<h3 id="function-cos">
Function <code>thrust::cos</code>
</h3>

<code class="doxybook">
<span>template &lt;typename T&gt;</span>
<span>__host__ __device__ complex< T > </span><span><b>cos</b>(const complex< T > & z);</span></code>
Returns the complex cosine of a <code>complex</code> number.

**Function Parameters**:
**`z`**: The <code>complex</code> argument. 

<h3 id="function-sin">
Function <code>thrust::sin</code>
</h3>

<code class="doxybook">
<span>template &lt;typename T&gt;</span>
<span>__host__ __device__ complex< T > </span><span><b>sin</b>(const complex< T > & z);</span></code>
Returns the complex sine of a <code>complex</code> number.

**Function Parameters**:
**`z`**: The <code>complex</code> argument. 

<h3 id="function-tan">
Function <code>thrust::tan</code>
</h3>

<code class="doxybook">
<span>template &lt;typename T&gt;</span>
<span>__host__ __device__ complex< T > </span><span><b>tan</b>(const complex< T > & z);</span></code>
Returns the complex tangent of a <code>complex</code> number.

**Function Parameters**:
**`z`**: The <code>complex</code> argument. 

<h3 id="function-cosh">
Function <code>thrust::cosh</code>
</h3>

<code class="doxybook">
<span>template &lt;typename T&gt;</span>
<span>__host__ __device__ complex< T > </span><span><b>cosh</b>(const complex< T > & z);</span></code>
Returns the complex hyperbolic cosine of a <code>complex</code> number.

**Function Parameters**:
**`z`**: The <code>complex</code> argument. 

<h3 id="function-sinh">
Function <code>thrust::sinh</code>
</h3>

<code class="doxybook">
<span>template &lt;typename T&gt;</span>
<span>__host__ __device__ complex< T > </span><span><b>sinh</b>(const complex< T > & z);</span></code>
Returns the complex hyperbolic sine of a <code>complex</code> number.

**Function Parameters**:
**`z`**: The <code>complex</code> argument. 

<h3 id="function-tanh">
Function <code>thrust::tanh</code>
</h3>

<code class="doxybook">
<span>template &lt;typename T&gt;</span>
<span>__host__ __device__ complex< T > </span><span><b>tanh</b>(const complex< T > & z);</span></code>
Returns the complex hyperbolic tangent of a <code>complex</code> number.

**Function Parameters**:
**`z`**: The <code>complex</code> argument. 

<h3 id="function-acos">
Function <code>thrust::acos</code>
</h3>

<code class="doxybook">
<span>template &lt;typename T&gt;</span>
<span>__host__ __device__ complex< T > </span><span><b>acos</b>(const complex< T > & z);</span></code>
Returns the complex arc cosine of a <code>complex</code> number.

The range of the real part of the result is [0, Pi] and the range of the imaginary part is [-inf, +inf]

**Function Parameters**:
**`z`**: The <code>complex</code> argument. 

<h3 id="function-asin">
Function <code>thrust::asin</code>
</h3>

<code class="doxybook">
<span>template &lt;typename T&gt;</span>
<span>__host__ __device__ complex< T > </span><span><b>asin</b>(const complex< T > & z);</span></code>
Returns the complex arc sine of a <code>complex</code> number.

The range of the real part of the result is [-Pi/2, Pi/2] and the range of the imaginary part is [-inf, +inf]

**Function Parameters**:
**`z`**: The <code>complex</code> argument. 

<h3 id="function-atan">
Function <code>thrust::atan</code>
</h3>

<code class="doxybook">
<span>template &lt;typename T&gt;</span>
<span>__host__ __device__ complex< T > </span><span><b>atan</b>(const complex< T > & z);</span></code>
Returns the complex arc tangent of a <code>complex</code> number.

The range of the real part of the result is [-Pi/2, Pi/2] and the range of the imaginary part is [-inf, +inf]

**Function Parameters**:
**`z`**: The <code>complex</code> argument. 

<h3 id="function-acosh">
Function <code>thrust::acosh</code>
</h3>

<code class="doxybook">
<span>template &lt;typename T&gt;</span>
<span>__host__ __device__ complex< T > </span><span><b>acosh</b>(const complex< T > & z);</span></code>
Returns the complex inverse hyperbolic cosine of a <code>complex</code> number.

The range of the real part of the result is [0, +inf] and the range of the imaginary part is [-Pi, Pi]

**Function Parameters**:
**`z`**: The <code>complex</code> argument. 

<h3 id="function-asinh">
Function <code>thrust::asinh</code>
</h3>

<code class="doxybook">
<span>template &lt;typename T&gt;</span>
<span>__host__ __device__ complex< T > </span><span><b>asinh</b>(const complex< T > & z);</span></code>
Returns the complex inverse hyperbolic sine of a <code>complex</code> number.

The range of the real part of the result is [-inf, +inf] and the range of the imaginary part is [-Pi/2, Pi/2]

**Function Parameters**:
**`z`**: The <code>complex</code> argument. 

<h3 id="function-atanh">
Function <code>thrust::atanh</code>
</h3>

<code class="doxybook">
<span>template &lt;typename T&gt;</span>
<span>__host__ __device__ complex< T > </span><span><b>atanh</b>(const complex< T > & z);</span></code>
Returns the complex inverse hyperbolic tangent of a <code>complex</code> number.

The range of the real part of the result is [-inf, +inf] and the range of the imaginary part is [-Pi/2, Pi/2]

**Function Parameters**:
**`z`**: The <code>complex</code> argument. 

<h3 id="function-operator<<">
Function <code>thrust::operator&lt;&lt;</code>
</h3>

<code class="doxybook">
<span>template &lt;typename T,</span>
<span>&nbsp;&nbsp;typename CharT,</span>
<span>&nbsp;&nbsp;typename Traits&gt;</span>
<span>std::basic_ostream< CharT, Traits > & </span><span><b>operator<<</b>(std::basic_ostream< CharT, Traits > & os,</span>
<span>&nbsp;&nbsp;const complex< T > & z);</span></code>
Writes to an output stream a <code>complex</code> number in the form (real, imaginary).

**Function Parameters**:
* **`os`** The output stream. 
* **`z`** The <code>complex</code> number to output. 

<h3 id="function-operator>>">
Function <code>thrust::operator&gt;&gt;</code>
</h3>

<code class="doxybook">
<span>template &lt;typename T,</span>
<span>&nbsp;&nbsp;typename CharT,</span>
<span>&nbsp;&nbsp;typename Traits&gt;</span>
<span>__host__ std::basic_istream< CharT, Traits > & </span><span><b>operator>></b>(std::basic_istream< CharT, Traits > & is,</span>
<span>&nbsp;&nbsp;complex< T > & z);</span></code>
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
Function <code>thrust::operator==</code>
</h3>

<code class="doxybook">
<span>template &lt;typename T0,</span>
<span>&nbsp;&nbsp;typename T1&gt;</span>
<span>__host__ __device__ bool </span><span><b>operator==</b>(const complex< T0 > & x,</span>
<span>&nbsp;&nbsp;const complex< T1 > & y);</span></code>
Returns true if two <code>complex</code> numbers are equal and false otherwise.

**Function Parameters**:
* **`x`** The first <code>complex</code>. 
* **`y`** The second <code>complex</code>. 

<h3 id="function-operator==">
Function <code>thrust::operator==</code>
</h3>

<code class="doxybook">
<span>template &lt;typename T0,</span>
<span>&nbsp;&nbsp;typename T1&gt;</span>
<span>__host__ __device__ bool </span><span><b>operator==</b>(const complex< T0 > & x,</span>
<span>&nbsp;&nbsp;const std::complex< T1 > & y);</span></code>
Returns true if two <code>complex</code> numbers are equal and false otherwise.

**Function Parameters**:
* **`x`** The first <code>complex</code>. 
* **`y`** The second <code>complex</code>. 

<h3 id="function-operator==">
Function <code>thrust::operator==</code>
</h3>

<code class="doxybook">
<span>template &lt;typename T0,</span>
<span>&nbsp;&nbsp;typename T1&gt;</span>
<span>__host__ __device__ bool </span><span><b>operator==</b>(const std::complex< T0 > & x,</span>
<span>&nbsp;&nbsp;const complex< T1 > & y);</span></code>
Returns true if two <code>complex</code> numbers are equal and false otherwise.

**Function Parameters**:
* **`x`** The first <code>complex</code>. 
* **`y`** The second <code>complex</code>. 

<h3 id="function-operator==">
Function <code>thrust::operator==</code>
</h3>

<code class="doxybook">
<span>template &lt;typename T0,</span>
<span>&nbsp;&nbsp;typename T1&gt;</span>
<span>__host__ __device__ bool </span><span><b>operator==</b>(const T0 & x,</span>
<span>&nbsp;&nbsp;const complex< T1 > & y);</span></code>
Returns true if the imaginary part of the <code>complex</code> number is zero and the real part is equal to the scalar. Returns false otherwise.

**Function Parameters**:
* **`x`** The scalar. 
* **`y`** The <code>complex</code>. 

<h3 id="function-operator==">
Function <code>thrust::operator==</code>
</h3>

<code class="doxybook">
<span>template &lt;typename T0,</span>
<span>&nbsp;&nbsp;typename T1&gt;</span>
<span>__host__ __device__ bool </span><span><b>operator==</b>(const complex< T0 > & x,</span>
<span>&nbsp;&nbsp;const T1 & y);</span></code>
Returns true if the imaginary part of the <code>complex</code> number is zero and the real part is equal to the scalar. Returns false otherwise.

**Function Parameters**:
* **`x`** The <code>complex</code>. 
* **`y`** The scalar. 

<h3 id="function-operator!=">
Function <code>thrust::operator!=</code>
</h3>

<code class="doxybook">
<span>template &lt;typename T0,</span>
<span>&nbsp;&nbsp;typename T1&gt;</span>
<span>__host__ __device__ bool </span><span><b>operator!=</b>(const complex< T0 > & x,</span>
<span>&nbsp;&nbsp;const complex< T1 > & y);</span></code>
Returns true if two <code>complex</code> numbers are different and false otherwise.

**Function Parameters**:
* **`x`** The first <code>complex</code>. 
* **`y`** The second <code>complex</code>. 

<h3 id="function-operator!=">
Function <code>thrust::operator!=</code>
</h3>

<code class="doxybook">
<span>template &lt;typename T0,</span>
<span>&nbsp;&nbsp;typename T1&gt;</span>
<span>__host__ __device__ bool </span><span><b>operator!=</b>(const complex< T0 > & x,</span>
<span>&nbsp;&nbsp;const std::complex< T1 > & y);</span></code>
Returns true if two <code>complex</code> numbers are different and false otherwise.

**Function Parameters**:
* **`x`** The first <code>complex</code>. 
* **`y`** The second <code>complex</code>. 

<h3 id="function-operator!=">
Function <code>thrust::operator!=</code>
</h3>

<code class="doxybook">
<span>template &lt;typename T0,</span>
<span>&nbsp;&nbsp;typename T1&gt;</span>
<span>__host__ __device__ bool </span><span><b>operator!=</b>(const std::complex< T0 > & x,</span>
<span>&nbsp;&nbsp;const complex< T1 > & y);</span></code>
Returns true if two <code>complex</code> numbers are different and false otherwise.

**Function Parameters**:
* **`x`** The first <code>complex</code>. 
* **`y`** The second <code>complex</code>. 

<h3 id="function-operator!=">
Function <code>thrust::operator!=</code>
</h3>

<code class="doxybook">
<span>template &lt;typename T0,</span>
<span>&nbsp;&nbsp;typename T1&gt;</span>
<span>__host__ __device__ bool </span><span><b>operator!=</b>(const T0 & x,</span>
<span>&nbsp;&nbsp;const complex< T1 > & y);</span></code>
Returns true if the imaginary part of the <code>complex</code> number is not zero or the real part is different from the scalar. Returns false otherwise.

**Function Parameters**:
* **`x`** The scalar. 
* **`y`** The <code>complex</code>. 

<h3 id="function-operator!=">
Function <code>thrust::operator!=</code>
</h3>

<code class="doxybook">
<span>template &lt;typename T0,</span>
<span>&nbsp;&nbsp;typename T1&gt;</span>
<span>__host__ __device__ bool </span><span><b>operator!=</b>(const complex< T0 > & x,</span>
<span>&nbsp;&nbsp;const T1 & y);</span></code>
Returns true if the imaginary part of the <code>complex</code> number is not zero or the real part is different from the scalar. Returns false otherwise.

**Function Parameters**:
* **`x`** The <code>complex</code>. 
* **`y`** The scalar. 


