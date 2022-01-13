---
title: thrust::system::error_condition
summary: The class error_condition describes an object used to hold values identifying error conditions. 
parent: System Diagnostics
grand_parent: System
nav_exclude: true
has_children: true
has_toc: false
---

# Class `thrust::system::error_condition`

The class <code>error&#95;condition</code> describes an object used to hold values identifying error conditions. 

**Note**:
<code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1system_1_1error__condition.html">error&#95;condition</a></code> values are portable abstractions, while <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1system_1_1error__code.html">error&#95;code</a></code> values are implementation specific. 

<code class="doxybook">
<span>#include <thrust/system/error_code.h></span><br>
<span>class thrust::system::error&#95;condition {</span>
<span>public:</span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1system_1_1error__condition.html#function-error-condition">error&#95;condition</a></b>(void);</span>
<br>
<span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1system_1_1error__condition.html#function-error-condition">error&#95;condition</a></b>(int val,</span>
<span>&nbsp;&nbsp;&nbsp;&nbsp;const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1system_1_1error__category.html">error_category</a> & cat);</span>
<br>
<span>&nbsp;&nbsp;template &lt;typename ErrorConditionEnum&gt;</span>
<span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1system_1_1error__condition.html#function-error-condition">error&#95;condition</a></b>(ErrorConditionEnum e);</span>
<br>
<span>&nbsp;&nbsp;void </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1system_1_1error__condition.html#function-assign">assign</a></b>(int val,</span>
<span>&nbsp;&nbsp;&nbsp;&nbsp;const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1system_1_1error__category.html">error_category</a> & cat);</span>
<br>
<span>&nbsp;&nbsp;template &lt;typename ErrorConditionEnum&gt;</span>
<span>&nbsp;&nbsp;<a href="{{ site.baseurl }}/api/classes/classthrust_1_1system_1_1error__condition.html">error_condition</a> & </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1system_1_1error__condition.html#function-operator=">operator=</a></b>(ErrorConditionEnum e);</span>
<br>
<span>&nbsp;&nbsp;void </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1system_1_1error__condition.html#function-clear">clear</a></b>(void);</span>
<br>
<span>&nbsp;&nbsp;int </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1system_1_1error__condition.html#function-value">value</a></b>(void) const;</span>
<br>
<span>&nbsp;&nbsp;const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1system_1_1error__category.html">error_category</a> & </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1system_1_1error__condition.html#function-category">category</a></b>(void) const;</span>
<br>
<span>&nbsp;&nbsp;std::string </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1system_1_1error__condition.html#function-message">message</a></b>(void) const;</span>
<br>
<span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1system_1_1error__condition.html#function-operator-bool">operator bool</a></b>(void) const;</span>
<span>};</span>
</code>

## Member Functions

<h3 id="function-error-condition">
Function <code>thrust::system::error&#95;condition::error&#95;condition</code>
</h3>

<code class="doxybook">
<span><b>error_condition</b>(void);</span></code>
Constructs an object of type <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1system_1_1error__condition.html">error&#95;condition</a></code>. 

**Postconditions**:
* <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1system_1_1error__condition.html#function-value">value()</a> == 0</code>. 
* <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1system_1_1error__condition.html#function-category">category()</a> == <a href="{{ site.baseurl }}/api/groups/group__system__diagnostics.html#function-generic-category">generic&#95;category()</a></code>. 

<h3 id="function-error-condition">
Function <code>thrust::system::error&#95;condition::error&#95;condition</code>
</h3>

<code class="doxybook">
<span><b>error_condition</b>(int val,</span>
<span>&nbsp;&nbsp;const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1system_1_1error__category.html">error_category</a> & cat);</span></code>
Constructs an object of type <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1system_1_1error__condition.html">error&#95;condition</a></code>. 

**Postconditions**:
* <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1system_1_1error__condition.html#function-value">value()</a> == val</code>. 
* <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1system_1_1error__condition.html#function-category">category()</a> == cat</code>. 

<h3 id="function-error-condition">
Function <code>thrust::system::error&#95;condition::error&#95;condition</code>
</h3>

<code class="doxybook">
<span>template &lt;typename ErrorConditionEnum&gt;</span>
<span><b>error_condition</b>(ErrorConditionEnum e);</span></code>
Constructs an object of type <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1system_1_1error__condition.html">error&#95;condition</a></code>. 

**Note**:
This constructor shall not participate in overload resolution unless <code>is&#95;error&#95;condition&#95;enum&lt;ErrorConditionEnum&gt;::value</code> is <code>true</code>. 

**Postconditions**:
<code>&#42;this == make&#95;error&#95;condition(e)</code>. 

<h3 id="function-assign">
Function <code>thrust::system::error&#95;condition::assign</code>
</h3>

<code class="doxybook">
<span>void </span><span><b>assign</b>(int val,</span>
<span>&nbsp;&nbsp;const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1system_1_1error__category.html">error_category</a> & cat);</span></code>
Assigns to this <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1system_1_1error__code.html">error&#95;code</a></code> object from an error value and an <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1system_1_1error__category.html">error&#95;category</a></code>. 

**Function Parameters**:
* **`val`** The new value to return from <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1system_1_1error__condition.html#function-value">value()</a></code>. 
* **`cat`** The new <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1system_1_1error__category.html">error&#95;category</a></code> to return from <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1system_1_1error__condition.html#function-category">category()</a></code>. 

**Postconditions**:
* <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1system_1_1error__condition.html#function-value">value()</a> == val</code>. 
* <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1system_1_1error__condition.html#function-category">category()</a> == cat</code>. 

<h3 id="function-operator=">
Function <code>thrust::system::error&#95;condition::operator=</code>
</h3>

<code class="doxybook">
<span>template &lt;typename ErrorConditionEnum&gt;</span>
<span><a href="{{ site.baseurl }}/api/classes/classthrust_1_1system_1_1error__condition.html">error_condition</a> & </span><span><b>operator=</b>(ErrorConditionEnum e);</span></code>
Assigns to this <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1system_1_1error__code.html">error&#95;code</a></code> object from an error condition enumeration. 

**Note**:
This operator shall not participate in overload resolution unless <code>is&#95;error&#95;condition&#95;enum&lt;ErrorConditionEnum&gt;::value</code> is <code>true</code>. 

**Postconditions**:
<code>&#42;this == make&#95;error&#95;condition(e)</code>. 

**Returns**:
*this 

<h3 id="function-clear">
Function <code>thrust::system::error&#95;condition::clear</code>
</h3>

<code class="doxybook">
<span>void </span><span><b>clear</b>(void);</span></code>
Clears this <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1system_1_1error__code.html">error&#95;code</a></code> object. 

**Postconditions**:
* <code>value == 0</code>
* <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1system_1_1error__condition.html#function-category">category()</a> == <a href="{{ site.baseurl }}/api/groups/group__system__diagnostics.html#function-generic-category">generic&#95;category()</a></code>. 

<h3 id="function-value">
Function <code>thrust::system::error&#95;condition::value</code>
</h3>

<code class="doxybook">
<span>int </span><span><b>value</b>(void) const;</span></code>
**Returns**:
The value encoded by this <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1system_1_1error__condition.html">error&#95;condition</a></code>. 

<h3 id="function-category">
Function <code>thrust::system::error&#95;condition::category</code>
</h3>

<code class="doxybook">
<span>const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1system_1_1error__category.html">error_category</a> & </span><span><b>category</b>(void) const;</span></code>
**Returns**:
A <code>const</code> reference to the <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1system_1_1error__category.html">error&#95;category</a></code> encoded by this <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1system_1_1error__condition.html">error&#95;condition</a></code>. 

<h3 id="function-message">
Function <code>thrust::system::error&#95;condition::message</code>
</h3>

<code class="doxybook">
<span>std::string </span><span><b>message</b>(void) const;</span></code>
**Returns**:
<code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1system_1_1error__condition.html#function-category">category()</a>.message(value())</code>. 

<h3 id="function-operator-bool">
Function <code>thrust::system::error&#95;condition::operator bool</code>
</h3>

<code class="doxybook">
<span><b>operator bool</b>(void) const;</span></code>
**Returns**:
<code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1system_1_1error__condition.html#function-value">value()</a> != 0</code>. 


