---
title: thrust::system::error_category
summary: The class error_category serves as a base class for types used to identify the source and encoding of a particular category of error code. Classes may be derived from error_category to support categories of errors in addition to those defined in the C++ International Standard. 
parent: System Diagnostics
grand_parent: System
nav_exclude: true
has_children: true
has_toc: false
---

# Class `thrust::system::error_category`

<code class="doxybook">
<span>#include <thrust/system/error_code.h></span><br>
<span>class thrust::system::error&#95;category {</span>
<span>public:</span><span>&nbsp;&nbsp;virtual </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1system_1_1error__category.html#function-~error-category">~error&#95;category</a></b>(void);</span>
<br>
<span>&nbsp;&nbsp;virtual const char * </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1system_1_1error__category.html#function-name">name</a></b>(void) const = 0;</span>
<br>
<span>&nbsp;&nbsp;virtual <a href="{{ site.baseurl }}/api/classes/classthrust_1_1system_1_1error__condition.html">error_condition</a> </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1system_1_1error__category.html#function-default-error-condition">default&#95;error&#95;condition</a></b>(int ev) const;</span>
<br>
<span>&nbsp;&nbsp;virtual bool </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1system_1_1error__category.html#function-equivalent">equivalent</a></b>(int code,</span>
<span>&nbsp;&nbsp;&nbsp;&nbsp;const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1system_1_1error__condition.html">error_condition</a> & condition) const;</span>
<br>
<span>&nbsp;&nbsp;virtual bool </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1system_1_1error__category.html#function-equivalent">equivalent</a></b>(const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1system_1_1error__code.html">error_code</a> & code,</span>
<span>&nbsp;&nbsp;&nbsp;&nbsp;int condition) const;</span>
<br>
<span>&nbsp;&nbsp;virtual std::string </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1system_1_1error__category.html#function-message">message</a></b>(int ev) const = 0;</span>
<br>
<span>&nbsp;&nbsp;bool </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1system_1_1error__category.html#function-operator==">operator==</a></b>(const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1system_1_1error__category.html">error_category</a> & rhs) const;</span>
<br>
<span>&nbsp;&nbsp;bool </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1system_1_1error__category.html#function-operator!=">operator!=</a></b>(const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1system_1_1error__category.html">error_category</a> & rhs) const;</span>
<br>
<span>&nbsp;&nbsp;bool </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1system_1_1error__category.html#function-operator<">operator&lt;</a></b>(const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1system_1_1error__category.html">error_category</a> & rhs) const;</span>
<span>};</span>
</code>

## Member Functions

<h3 id="function-~error-category">
Function <code>thrust::system::error&#95;category::~error&#95;category</code>
</h3>

<code class="doxybook">
<span>virtual </span><span><b>~error_category</b>(void);</span></code>
Destructor does nothing. 

<h3 id="function-name">
Function <code>thrust::system::error&#95;category::name</code>
</h3>

<code class="doxybook">
<span>virtual const char * </span><span><b>name</b>(void) const = 0;</span></code>
**Returns**:
A string naming the error category. 

<h3 id="function-default-error-condition">
Function <code>thrust::system::error&#95;category::default&#95;error&#95;condition</code>
</h3>

<code class="doxybook">
<span>virtual <a href="{{ site.baseurl }}/api/classes/classthrust_1_1system_1_1error__condition.html">error_condition</a> </span><span><b>default_error_condition</b>(int ev) const;</span></code>
**Returns**:
<code>error&#95;condition(ev, &#42;this)</code>. 

<h3 id="function-equivalent">
Function <code>thrust::system::error&#95;category::equivalent</code>
</h3>

<code class="doxybook">
<span>virtual bool </span><span><b>equivalent</b>(int code,</span>
<span>&nbsp;&nbsp;const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1system_1_1error__condition.html">error_condition</a> & condition) const;</span></code>
**Returns**:
<code>default&#95;error&#95;condition(code) == condition</code>

<h3 id="function-equivalent">
Function <code>thrust::system::error&#95;category::equivalent</code>
</h3>

<code class="doxybook">
<span>virtual bool </span><span><b>equivalent</b>(const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1system_1_1error__code.html">error_code</a> & code,</span>
<span>&nbsp;&nbsp;int condition) const;</span></code>
**Returns**:
<code>&#42;this == code.category() && code.value() == condition</code>

<h3 id="function-message">
Function <code>thrust::system::error&#95;category::message</code>
</h3>

<code class="doxybook">
<span>virtual std::string </span><span><b>message</b>(int ev) const = 0;</span></code>
**Returns**:
A string that describes the error condition denoted by <code>ev</code>. 

<h3 id="function-operator==">
Function <code>thrust::system::error&#95;category::operator==</code>
</h3>

<code class="doxybook">
<span>bool </span><span><b>operator==</b>(const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1system_1_1error__category.html">error_category</a> & rhs) const;</span></code>
**Returns**:
<code>&#42;this == &rhs</code>

<h3 id="function-operator!=">
Function <code>thrust::system::error&#95;category::operator!=</code>
</h3>

<code class="doxybook">
<span>bool </span><span><b>operator!=</b>(const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1system_1_1error__category.html">error_category</a> & rhs) const;</span></code>
**Returns**:
<code>!(&#42;this == rhs)</code>

<h3 id="function-operator<">
Function <code>thrust::system::error&#95;category::operator&lt;</code>
</h3>

<code class="doxybook">
<span>bool </span><span><b>operator<</b>(const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1system_1_1error__category.html">error_category</a> & rhs) const;</span></code>
**Note**:
<code>less</code> provides a total ordering for pointers. 

**Returns**:
<code>less&lt;const error&#95;category&#42;&gt;()(this, &rhs)</code>


