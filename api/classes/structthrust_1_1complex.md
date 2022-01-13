---
title: thrust::complex
parent: Complex Numbers
grand_parent: Numerics
nav_exclude: true
has_children: true
has_toc: false
---

# Struct `thrust::complex`

<code>complex</code> is the Thrust equivalent to <code>std::complex</code>. It is functionally identical to it, but can also be used in device code which <code>std::complex</code> currently cannot.

**Template Parameters**:
**`T`**: The type used to hold the real and imaginary parts. Should be <code>float</code> or <code>double</code>. Others types are not supported. 

<code class="doxybook">
<span>#include <thrust/complex.h></span><br>
<span>template &lt;typename T&gt;</span>
<span>struct thrust::complex {</span>
<span>public:</span><span>&nbsp;&nbsp;typedef <i>see below</i> <b><a href="{{ site.baseurl }}/api/groups/group__complex__numbers.html#typedef-value-type">value&#95;type</a></b>;</span>
<br>
<span>&nbsp;&nbsp;__host__ __device__ </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/groups/group__complex__numbers.html#function-complex">complex</a></b>(const T & re);</span>
<br>
<span>&nbsp;&nbsp;__host__ __device__ </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/groups/group__complex__numbers.html#function-complex">complex</a></b>(const T & re,</span>
<span>&nbsp;&nbsp;&nbsp;&nbsp;const T & im);</span>
<br>
<span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/groups/group__complex__numbers.html#function-complex">complex</a></b>() = default;</span>
<br>
<span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/groups/group__complex__numbers.html#function-complex">complex</a></b>(const <a href="{{ site.baseurl }}/api/classes/structthrust_1_1complex.html">complex</a>< T > & z) = default;</span>
<br>
<span>&nbsp;&nbsp;template &lt;typename U&gt;</span>
<span>&nbsp;&nbsp;__host__ __device__ </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/groups/group__complex__numbers.html#function-complex">complex</a></b>(const <a href="{{ site.baseurl }}/api/classes/structthrust_1_1complex.html">complex</a>< U > & z);</span>
<br>
<span>&nbsp;&nbsp;__host__ __device__ </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/groups/group__complex__numbers.html#function-complex">complex</a></b>(const std::complex< T > & z);</span>
<br>
<span>&nbsp;&nbsp;template &lt;typename U&gt;</span>
<span>&nbsp;&nbsp;__host__ __device__ </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/groups/group__complex__numbers.html#function-complex">complex</a></b>(const std::complex< U > & z);</span>
<br>
<span>&nbsp;&nbsp;__host__ __device__ <a href="{{ site.baseurl }}/api/classes/structthrust_1_1complex.html">complex</a> & </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/groups/group__complex__numbers.html#function-operator=">operator=</a></b>(const T & re);</span>
<br>
<span>&nbsp;&nbsp;<a href="{{ site.baseurl }}/api/classes/structthrust_1_1complex.html">complex</a> & </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/groups/group__complex__numbers.html#function-operator=">operator=</a></b>(const <a href="{{ site.baseurl }}/api/classes/structthrust_1_1complex.html">complex</a>< T > & z) = default;</span>
<br>
<span>&nbsp;&nbsp;template &lt;typename U&gt;</span>
<span>&nbsp;&nbsp;__host__ __device__ <a href="{{ site.baseurl }}/api/classes/structthrust_1_1complex.html">complex</a> & </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/groups/group__complex__numbers.html#function-operator=">operator=</a></b>(const <a href="{{ site.baseurl }}/api/classes/structthrust_1_1complex.html">complex</a>< U > & z);</span>
<br>
<span>&nbsp;&nbsp;__host__ __device__ <a href="{{ site.baseurl }}/api/classes/structthrust_1_1complex.html">complex</a> & </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/groups/group__complex__numbers.html#function-operator=">operator=</a></b>(const std::complex< T > & z);</span>
<br>
<span>&nbsp;&nbsp;template &lt;typename U&gt;</span>
<span>&nbsp;&nbsp;__host__ __device__ <a href="{{ site.baseurl }}/api/classes/structthrust_1_1complex.html">complex</a> & </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/groups/group__complex__numbers.html#function-operator=">operator=</a></b>(const std::complex< U > & z);</span>
<br>
<span>&nbsp;&nbsp;template &lt;typename U&gt;</span>
<span>&nbsp;&nbsp;__host__ __device__ <a href="{{ site.baseurl }}/api/classes/structthrust_1_1complex.html">complex</a>< T > & </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/groups/group__complex__numbers.html#function-operator+=">operator+=</a></b>(const <a href="{{ site.baseurl }}/api/classes/structthrust_1_1complex.html">complex</a>< U > & z);</span>
<br>
<span>&nbsp;&nbsp;template &lt;typename U&gt;</span>
<span>&nbsp;&nbsp;__host__ __device__ <a href="{{ site.baseurl }}/api/classes/structthrust_1_1complex.html">complex</a>< T > & </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/groups/group__complex__numbers.html#function-operator-=">operator-=</a></b>(const <a href="{{ site.baseurl }}/api/classes/structthrust_1_1complex.html">complex</a>< U > & z);</span>
<br>
<span>&nbsp;&nbsp;template &lt;typename U&gt;</span>
<span>&nbsp;&nbsp;__host__ __device__ <a href="{{ site.baseurl }}/api/classes/structthrust_1_1complex.html">complex</a>< T > & </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/groups/group__complex__numbers.html#function-operator*=">operator&#42;=</a></b>(const <a href="{{ site.baseurl }}/api/classes/structthrust_1_1complex.html">complex</a>< U > & z);</span>
<br>
<span>&nbsp;&nbsp;template &lt;typename U&gt;</span>
<span>&nbsp;&nbsp;__host__ __device__ <a href="{{ site.baseurl }}/api/classes/structthrust_1_1complex.html">complex</a>< T > & </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/groups/group__complex__numbers.html#function-operator/=">operator/=</a></b>(const <a href="{{ site.baseurl }}/api/classes/structthrust_1_1complex.html">complex</a>< U > & z);</span>
<br>
<span>&nbsp;&nbsp;template &lt;typename U&gt;</span>
<span>&nbsp;&nbsp;__host__ __device__ <a href="{{ site.baseurl }}/api/classes/structthrust_1_1complex.html">complex</a>< T > & </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/groups/group__complex__numbers.html#function-operator+=">operator+=</a></b>(const U & z);</span>
<br>
<span>&nbsp;&nbsp;template &lt;typename U&gt;</span>
<span>&nbsp;&nbsp;__host__ __device__ <a href="{{ site.baseurl }}/api/classes/structthrust_1_1complex.html">complex</a>< T > & </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/groups/group__complex__numbers.html#function-operator-=">operator-=</a></b>(const U & z);</span>
<br>
<span>&nbsp;&nbsp;template &lt;typename U&gt;</span>
<span>&nbsp;&nbsp;__host__ __device__ <a href="{{ site.baseurl }}/api/classes/structthrust_1_1complex.html">complex</a>< T > & </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/groups/group__complex__numbers.html#function-operator*=">operator&#42;=</a></b>(const U & z);</span>
<br>
<span>&nbsp;&nbsp;template &lt;typename U&gt;</span>
<span>&nbsp;&nbsp;__host__ __device__ <a href="{{ site.baseurl }}/api/classes/structthrust_1_1complex.html">complex</a>< T > & </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/groups/group__complex__numbers.html#function-operator/=">operator/=</a></b>(const U & z);</span>
<br>
<span>&nbsp;&nbsp;__host__ __device__ T </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/groups/group__complex__numbers.html#function-real">real</a></b>() const;</span>
<br>
<span>&nbsp;&nbsp;__host__ __device__ T </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/groups/group__complex__numbers.html#function-imag">imag</a></b>() const;</span>
<br>
<span>&nbsp;&nbsp;__host__ __device__ T </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/groups/group__complex__numbers.html#function-real">real</a></b>() const;</span>
<br>
<span>&nbsp;&nbsp;__host__ __device__ T </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/groups/group__complex__numbers.html#function-imag">imag</a></b>() const;</span>
<br>
<span>&nbsp;&nbsp;__host__ __device__ void </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/groups/group__complex__numbers.html#function-real">real</a></b>(T re);</span>
<br>
<span>&nbsp;&nbsp;__host__ __device__ void </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/groups/group__complex__numbers.html#function-imag">imag</a></b>(T im);</span>
<br>
<span>&nbsp;&nbsp;__host__ __device__ void </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/groups/group__complex__numbers.html#function-real">real</a></b>(T re);</span>
<br>
<span>&nbsp;&nbsp;__host__ __device__ void </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/groups/group__complex__numbers.html#function-imag">imag</a></b>(T im);</span>
<br>
<span>&nbsp;&nbsp;__host__ </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/groups/group__complex__numbers.html#function-operator-stdcomplex<-t->">complex&lt; T &gt;</a></b>() const;</span>
<span>};</span>
</code>

## Member Types

<h3 id="typedef-value-type">
Typedef <code>thrust::complex::value&#95;type</code>
</h3>

<code class="doxybook">
<span>typedef T<b>value_type</b>;</span></code>
<code>value&#95;type</code> is the type of <code>complex's</code> real and imaginary parts. 


## Member Functions

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
<span><b>complex</b>(const <a href="{{ site.baseurl }}/api/classes/structthrust_1_1complex.html">complex</a>< T > & z) = default;</span></code>
This copy constructor copies from a <code>complex</code> with a type that is convertible to this <code>complex's</code><code>value&#95;type</code>.

**Function Parameters**:
**`z`**: The <code>complex</code> to copy from. 

<h3 id="function-complex">
Function <code>thrust::complex::complex</code>
</h3>

<code class="doxybook">
<span>template &lt;typename U&gt;</span>
<span>__host__ __device__ </span><span><b>complex</b>(const <a href="{{ site.baseurl }}/api/classes/structthrust_1_1complex.html">complex</a>< U > & z);</span></code>
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
<span>__host__ __device__ <a href="{{ site.baseurl }}/api/classes/structthrust_1_1complex.html">complex</a> & </span><span><b>operator=</b>(const T & re);</span></code>
Assign <code>re</code> to the real part of this <code>complex</code> and set the imaginary part to 0.

**Function Parameters**:
**`re`**: The real part of the number. 

<h3 id="function-operator=">
Function <code>thrust::complex::operator=</code>
</h3>

<code class="doxybook">
<span><a href="{{ site.baseurl }}/api/classes/structthrust_1_1complex.html">complex</a> & </span><span><b>operator=</b>(const <a href="{{ site.baseurl }}/api/classes/structthrust_1_1complex.html">complex</a>< T > & z) = default;</span></code>
Assign <code>z.real()</code> and <code>z.imag()</code> to the real and imaginary parts of this <code>complex</code> respectively.

**Function Parameters**:
**`z`**: The <code>complex</code> to copy from. 

<h3 id="function-operator=">
Function <code>thrust::complex::operator=</code>
</h3>

<code class="doxybook">
<span>template &lt;typename U&gt;</span>
<span>__host__ __device__ <a href="{{ site.baseurl }}/api/classes/structthrust_1_1complex.html">complex</a> & </span><span><b>operator=</b>(const <a href="{{ site.baseurl }}/api/classes/structthrust_1_1complex.html">complex</a>< U > & z);</span></code>
Assign <code>z.real()</code> and <code>z.imag()</code> to the real and imaginary parts of this <code>complex</code> respectively.

**Template Parameters**:
**`U`**: is convertible to <code>value&#95;type</code>. 

**Function Parameters**:
**`z`**: The <code>complex</code> to copy from.

<h3 id="function-operator=">
Function <code>thrust::complex::operator=</code>
</h3>

<code class="doxybook">
<span>__host__ __device__ <a href="{{ site.baseurl }}/api/classes/structthrust_1_1complex.html">complex</a> & </span><span><b>operator=</b>(const std::complex< T > & z);</span></code>
Assign <code>z.real()</code> and <code>z.imag()</code> to the real and imaginary parts of this <code>complex</code> respectively.

**Function Parameters**:
**`z`**: The <code>complex</code> to copy from. 

<h3 id="function-operator=">
Function <code>thrust::complex::operator=</code>
</h3>

<code class="doxybook">
<span>template &lt;typename U&gt;</span>
<span>__host__ __device__ <a href="{{ site.baseurl }}/api/classes/structthrust_1_1complex.html">complex</a> & </span><span><b>operator=</b>(const std::complex< U > & z);</span></code>
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
<span>__host__ __device__ <a href="{{ site.baseurl }}/api/classes/structthrust_1_1complex.html">complex</a>< T > & </span><span><b>operator+=</b>(const <a href="{{ site.baseurl }}/api/classes/structthrust_1_1complex.html">complex</a>< U > & z);</span></code>
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
<span>__host__ __device__ <a href="{{ site.baseurl }}/api/classes/structthrust_1_1complex.html">complex</a>< T > & </span><span><b>operator-=</b>(const <a href="{{ site.baseurl }}/api/classes/structthrust_1_1complex.html">complex</a>< U > & z);</span></code>
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
<span>__host__ __device__ <a href="{{ site.baseurl }}/api/classes/structthrust_1_1complex.html">complex</a>< T > & </span><span><b>operator*=</b>(const <a href="{{ site.baseurl }}/api/classes/structthrust_1_1complex.html">complex</a>< U > & z);</span></code>
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
<span>__host__ __device__ <a href="{{ site.baseurl }}/api/classes/structthrust_1_1complex.html">complex</a>< T > & </span><span><b>operator/=</b>(const <a href="{{ site.baseurl }}/api/classes/structthrust_1_1complex.html">complex</a>< U > & z);</span></code>
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
<span>__host__ __device__ <a href="{{ site.baseurl }}/api/classes/structthrust_1_1complex.html">complex</a>< T > & </span><span><b>operator+=</b>(const U & z);</span></code>
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
<span>__host__ __device__ <a href="{{ site.baseurl }}/api/classes/structthrust_1_1complex.html">complex</a>< T > & </span><span><b>operator-=</b>(const U & z);</span></code>
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
<span>__host__ __device__ <a href="{{ site.baseurl }}/api/classes/structthrust_1_1complex.html">complex</a>< T > & </span><span><b>operator*=</b>(const U & z);</span></code>
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
<span>__host__ __device__ <a href="{{ site.baseurl }}/api/classes/structthrust_1_1complex.html">complex</a>< T > & </span><span><b>operator/=</b>(const U & z);</span></code>
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


