---
title: thrust::system::error_code
summary: The class error_code describes an object used to hold error code values, such as those originating from the operating system or other low-level application program interfaces. 
parent: System Diagnostics
grand_parent: System
nav_exclude: true
has_children: true
has_toc: false
---

# Class `thrust::system::error_code`

<code class="doxybook">
<span>#include <thrust/system/error_code.h></span><br>
<span>class thrust::system::error&#95;code {</span>
<span>public:</span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1system_1_1error__code.html#function-error-code">error&#95;code</a></b>(void);</span>
<br>
<span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1system_1_1error__code.html#function-error-code">error&#95;code</a></b>(int val,</span>
<span>&nbsp;&nbsp;&nbsp;&nbsp;const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1system_1_1error__category.html">error_category</a> & cat);</span>
<br>
<span>&nbsp;&nbsp;template &lt;typename ErrorCodeEnum&gt;</span>
<span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1system_1_1error__code.html#function-error-code">error&#95;code</a></b>(ErrorCodeEnum e);</span>
<br>
<span>&nbsp;&nbsp;void </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1system_1_1error__code.html#function-assign">assign</a></b>(int val,</span>
<span>&nbsp;&nbsp;&nbsp;&nbsp;const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1system_1_1error__category.html">error_category</a> & cat);</span>
<br>
<span>&nbsp;&nbsp;template &lt;typename ErrorCodeEnum&gt;</span>
<span>&nbsp;&nbsp;<a href="{{ site.baseurl }}/api/classes/classthrust_1_1system_1_1error__code.html">error_code</a> & </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1system_1_1error__code.html#function-operator=">operator=</a></b>(ErrorCodeEnum e);</span>
<br>
<span>&nbsp;&nbsp;void </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1system_1_1error__code.html#function-clear">clear</a></b>(void);</span>
<br>
<span>&nbsp;&nbsp;int </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1system_1_1error__code.html#function-value">value</a></b>(void) const;</span>
<br>
<span>&nbsp;&nbsp;const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1system_1_1error__category.html">error_category</a> & </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1system_1_1error__code.html#function-category">category</a></b>(void) const;</span>
<br>
<span>&nbsp;&nbsp;<a href="{{ site.baseurl }}/api/classes/classthrust_1_1system_1_1error__condition.html">error_condition</a> </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1system_1_1error__code.html#function-default-error-condition">default&#95;error&#95;condition</a></b>(void) const;</span>
<br>
<span>&nbsp;&nbsp;std::string </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1system_1_1error__code.html#function-message">message</a></b>(void) const;</span>
<br>
<span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1system_1_1error__code.html#function-operator-bool">operator bool</a></b>(void) const;</span>
<span>};</span>
</code>

## Member Functions

<h3 id="function-error-code">
Function <code>thrust::system::error&#95;code::error&#95;code</code>
</h3>

<code class="doxybook">
<span><b>error_code</b>(void);</span></code>
Effects: Constructs an object of type <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1system_1_1error__code.html">error&#95;code</a></code>. 

**Postconditions**:
<code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1system_1_1error__code.html#function-value">value()</a> == 0</code> and <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1system_1_1error__code.html#function-category">category()</a> == &<a href="{{ site.baseurl }}/api/groups/group__system__diagnostics.html#function-system-category">system&#95;category()</a></code>. 

<h3 id="function-error-code">
Function <code>thrust::system::error&#95;code::error&#95;code</code>
</h3>

<code class="doxybook">
<span><b>error_code</b>(int val,</span>
<span>&nbsp;&nbsp;const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1system_1_1error__category.html">error_category</a> & cat);</span></code>
Effects: Constructs an object of type <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1system_1_1error__code.html">error&#95;code</a></code>. 

**Postconditions**:
<code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1system_1_1error__code.html#function-value">value()</a> == val</code> and <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1system_1_1error__code.html#function-category">category()</a> == &cat</code>. 

<h3 id="function-error-code">
Function <code>thrust::system::error&#95;code::error&#95;code</code>
</h3>

<code class="doxybook">
<span>template &lt;typename ErrorCodeEnum&gt;</span>
<span><b>error_code</b>(ErrorCodeEnum e);</span></code>
Effects: Constructs an object of type <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1system_1_1error__code.html">error&#95;code</a></code>. 

**Postconditions**:
<code>&#42;this == make&#95;error&#95;code(e)</code>. 

<h3 id="function-assign">
Function <code>thrust::system::error&#95;code::assign</code>
</h3>

<code class="doxybook">
<span>void </span><span><b>assign</b>(int val,</span>
<span>&nbsp;&nbsp;const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1system_1_1error__category.html">error_category</a> & cat);</span></code>
**Postconditions**:
<code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1system_1_1error__code.html#function-value">value()</a> == val</code> and <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1system_1_1error__code.html#function-category">category()</a> == &cat</code>. 

<h3 id="function-operator=">
Function <code>thrust::system::error&#95;code::operator=</code>
</h3>

<code class="doxybook">
<span>template &lt;typename ErrorCodeEnum&gt;</span>
<span><a href="{{ site.baseurl }}/api/classes/classthrust_1_1system_1_1error__code.html">error_code</a> & </span><span><b>operator=</b>(ErrorCodeEnum e);</span></code>
**Postconditions**:
<code>&#42;this == make&#95;error&#95;code(e)</code>. 

<h3 id="function-clear">
Function <code>thrust::system::error&#95;code::clear</code>
</h3>

<code class="doxybook">
<span>void </span><span><b>clear</b>(void);</span></code>
**Postconditions**:
<code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1system_1_1error__code.html#function-value">value()</a> == 0</code> and <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1system_1_1error__code.html#function-category">category()</a> == <a href="{{ site.baseurl }}/api/groups/group__system__diagnostics.html#function-system-category">system&#95;category()</a></code>. 

<h3 id="function-value">
Function <code>thrust::system::error&#95;code::value</code>
</h3>

<code class="doxybook">
<span>int </span><span><b>value</b>(void) const;</span></code>
**Returns**:
An integral value of this <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1system_1_1error__code.html">error&#95;code</a></code> object. 

<h3 id="function-category">
Function <code>thrust::system::error&#95;code::category</code>
</h3>

<code class="doxybook">
<span>const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1system_1_1error__category.html">error_category</a> & </span><span><b>category</b>(void) const;</span></code>
**Returns**:
An <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1system_1_1error__category.html">error&#95;category</a></code> describing the category of this <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1system_1_1error__code.html">error&#95;code</a></code> object. 

<h3 id="function-default-error-condition">
Function <code>thrust::system::error&#95;code::default&#95;error&#95;condition</code>
</h3>

<code class="doxybook">
<span><a href="{{ site.baseurl }}/api/classes/classthrust_1_1system_1_1error__condition.html">error_condition</a> </span><span><b>default_error_condition</b>(void) const;</span></code>
**Returns**:
<code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1system_1_1error__code.html#function-category">category()</a>.<a href="{{ site.baseurl }}/api/classes/classthrust_1_1system_1_1error__code.html#function-default-error-condition">default&#95;error&#95;condition()</a></code>. 

<h3 id="function-message">
Function <code>thrust::system::error&#95;code::message</code>
</h3>

<code class="doxybook">
<span>std::string </span><span><b>message</b>(void) const;</span></code>
**Returns**:
<code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1system_1_1error__code.html#function-category">category()</a>.message(value())</code>. 

<h3 id="function-operator-bool">
Function <code>thrust::system::error&#95;code::operator bool</code>
</h3>

<code class="doxybook">
<span><b>operator bool</b>(void) const;</span></code>
**Returns**:
<code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1system_1_1error__code.html#function-value">value()</a> != 0</code>. 


