---
title: System Diagnostics
parent: System
grand_parent: API
nav_exclude: false
has_children: true
has_toc: false
---

# System Diagnostics

<code class="doxybook">
<span>namespace <b><a href="/api/namespaces/namespacesystem_1_1errc.html">system::errc</a></b> { <i>…</i> }</span>
<br>
<span>template &lt;typename T&gt;</span>
<span>struct <b><a href="/api/classes/structsystem_1_1is__error__code__enum.html">system::is&#95;error&#95;code&#95;enum</a></b>;</span>
<br>
<span>template &lt;typename T&gt;</span>
<span>struct <b><a href="/api/classes/structsystem_1_1is__error__condition__enum.html">system::is&#95;error&#95;condition&#95;enum</a></b>;</span>
<br>
<span>struct <b><a href="/api/classes/structsystem_1_1is__error__condition__enum_3_01errc_1_1errc__t_01_4.html">system::is&#95;error&#95;condition&#95;enum&lt; errc::errc&#95;t &gt;</a></b>;</span>
<br>
<span class="doxybook-comment">/* The class <code>error&#95;category</code> serves as a base class for types used to identify the source and encoding of a particular category of error code. Classes may be derived from <code>error&#95;category</code> to support categories of errors in addition to those defined in the C++ International Standard.  */</span><span>class <b><a href="/api/classes/classsystem_1_1error__category.html">system::error&#95;category</a></b>;</span>
<br>
<span class="doxybook-comment">/* The class <code>error&#95;code</code> describes an object used to hold error code values, such as those originating from the operating system or other low-level application program interfaces.  */</span><span>class <b><a href="/api/classes/classsystem_1_1error__code.html">system::error&#95;code</a></b>;</span>
<br>
<span class="doxybook-comment">/* The class <code>error&#95;condition</code> describes an object used to hold values identifying error conditions.  */</span><span>class <b><a href="/api/classes/classsystem_1_1error__condition.html">system::error&#95;condition</a></b>;</span>
<br>
<span class="doxybook-comment">/* The class <code>system&#95;error</code> describes an exception object used to report error conditions that have an associated <code><a href="/api/classes/classsystem_1_1error__code.html">error&#95;code</a></code>. Such error conditions typically originate from the operating system or other low-level application program interfaces.  */</span><span>class <b><a href="/api/classes/classsystem_1_1system__error.html">system::system&#95;error</a></b>;</span>
<br>
<span>const <a href="/api/classes/classsystem_1_1error__category.html">error_category</a> & </span><span><b><a href="/api/groups/group__system__diagnostics.html#function-generic_category">system::generic&#95;category</a></b>(void);</span>
<br>
<span>const <a href="/api/classes/classsystem_1_1error__category.html">error_category</a> & </span><span><b><a href="/api/groups/group__system__diagnostics.html#function-system_category">system::system&#95;category</a></b>(void);</span>
<br>
<span><a href="/api/classes/classsystem_1_1error__code.html">error_code</a> </span><span><b><a href="/api/groups/group__system__diagnostics.html#function-make_error_code">system::make&#95;error&#95;code</a></b>(errc::errc_t e);</span>
<br>
<span>bool </span><span><b><a href="/api/groups/group__system__diagnostics.html#function-operator<">system::operator&lt;</a></b>(const <a href="/api/classes/classsystem_1_1error__code.html">error_code</a> & lhs,</span>
<span>&nbsp;&nbsp;const <a href="/api/classes/classsystem_1_1error__code.html">error_code</a> & rhs);</span>
<br>
<span>template &lt;typename charT,</span>
<span>&nbsp;&nbsp;typename traits&gt;</span>
<span>std::basic_ostream< charT, traits > & </span><span><b><a href="/api/groups/group__system__diagnostics.html#function-operator<<">system::operator&lt;&lt;</a></b>(std::basic_ostream< charT, traits > & os,</span>
<span>&nbsp;&nbsp;const <a href="/api/classes/classsystem_1_1error__code.html">error_code</a> & ec);</span>
<br>
<span><a href="/api/classes/classsystem_1_1error__condition.html">error_condition</a> </span><span><b><a href="/api/groups/group__system__diagnostics.html#function-make_error_condition">system::make&#95;error&#95;condition</a></b>(errc::errc_t e);</span>
<br>
<span>bool </span><span><b><a href="/api/groups/group__system__diagnostics.html#function-operator<">system::operator&lt;</a></b>(const <a href="/api/classes/classsystem_1_1error__condition.html">error_condition</a> & lhs,</span>
<span>&nbsp;&nbsp;const <a href="/api/classes/classsystem_1_1error__condition.html">error_condition</a> & rhs);</span>
<br>
<span>bool </span><span><b><a href="/api/groups/group__system__diagnostics.html#function-operator==">system::operator==</a></b>(const <a href="/api/classes/classsystem_1_1error__code.html">error_code</a> & lhs,</span>
<span>&nbsp;&nbsp;const <a href="/api/classes/classsystem_1_1error__code.html">error_code</a> & rhs);</span>
<br>
<span>bool </span><span><b><a href="/api/groups/group__system__diagnostics.html#function-operator==">system::operator==</a></b>(const <a href="/api/classes/classsystem_1_1error__code.html">error_code</a> & lhs,</span>
<span>&nbsp;&nbsp;const <a href="/api/classes/classsystem_1_1error__condition.html">error_condition</a> & rhs);</span>
<br>
<span>bool </span><span><b><a href="/api/groups/group__system__diagnostics.html#function-operator==">system::operator==</a></b>(const <a href="/api/classes/classsystem_1_1error__condition.html">error_condition</a> & lhs,</span>
<span>&nbsp;&nbsp;const <a href="/api/classes/classsystem_1_1error__code.html">error_code</a> & rhs);</span>
<br>
<span>bool </span><span><b><a href="/api/groups/group__system__diagnostics.html#function-operator==">system::operator==</a></b>(const <a href="/api/classes/classsystem_1_1error__condition.html">error_condition</a> & lhs,</span>
<span>&nbsp;&nbsp;const <a href="/api/classes/classsystem_1_1error__condition.html">error_condition</a> & rhs);</span>
<br>
<span>bool </span><span><b><a href="/api/groups/group__system__diagnostics.html#function-operator!=">system::operator!=</a></b>(const <a href="/api/classes/classsystem_1_1error__code.html">error_code</a> & lhs,</span>
<span>&nbsp;&nbsp;const <a href="/api/classes/classsystem_1_1error__code.html">error_code</a> & rhs);</span>
<br>
<span>bool </span><span><b><a href="/api/groups/group__system__diagnostics.html#function-operator!=">system::operator!=</a></b>(const <a href="/api/classes/classsystem_1_1error__code.html">error_code</a> & lhs,</span>
<span>&nbsp;&nbsp;const <a href="/api/classes/classsystem_1_1error__condition.html">error_condition</a> & rhs);</span>
<br>
<span>bool </span><span><b><a href="/api/groups/group__system__diagnostics.html#function-operator!=">system::operator!=</a></b>(const <a href="/api/classes/classsystem_1_1error__condition.html">error_condition</a> & lhs,</span>
<span>&nbsp;&nbsp;const <a href="/api/classes/classsystem_1_1error__code.html">error_code</a> & rhs);</span>
<br>
<span>bool </span><span><b><a href="/api/groups/group__system__diagnostics.html#function-operator!=">system::operator!=</a></b>(const <a href="/api/classes/classsystem_1_1error__condition.html">error_condition</a> & lhs,</span>
<span>&nbsp;&nbsp;const <a href="/api/classes/classsystem_1_1error__condition.html">error_condition</a> & rhs);</span>
<br>
<span>const <a href="/api/classes/classsystem_1_1error__code.html">error_code</a> & </span><span><b><a href="/api/groups/group__system__diagnostics.html#function-code">system::system&#95;error::code</a></b>(void) const;</span>
<br>
<span>const char * </span><span><b><a href="/api/groups/group__system__diagnostics.html#function-what">system::system&#95;error::what</a></b>(void) const;</span>
</code>

## Member Classes

<h3 id="struct-system::is_error_code_enum">
<a href="/api/classes/structsystem_1_1is__error__code__enum.html">Struct <code>system::is&#95;error&#95;code&#95;enum</code>
</a>
</h3>

**Inherits From**:
`false_type`

<h3 id="struct-system::is_error_condition_enum">
<a href="/api/classes/structsystem_1_1is__error__condition__enum.html">Struct <code>system::is&#95;error&#95;condition&#95;enum</code>
</a>
</h3>

**Inherits From**:
`false_type`

<h3 id="struct-system::is_error_condition_enum< errc::errc_t >">
<a href="/api/classes/structsystem_1_1is__error__condition__enum_3_01errc_1_1errc__t_01_4.html">Struct <code>system::is&#95;error&#95;condition&#95;enum&lt; errc::errc&#95;t &gt;</code>
</a>
</h3>

**Inherits From**:
`true_type`

<h3 id="class-system::error_category">
<a href="/api/classes/classsystem_1_1error__category.html">Class <code>system::error&#95;category</code>
</a>
</h3>

The class <code>error&#95;category</code> serves as a base class for types used to identify the source and encoding of a particular category of error code. Classes may be derived from <code>error&#95;category</code> to support categories of errors in addition to those defined in the C++ International Standard. 

<h3 id="class-system::error_code">
<a href="/api/classes/classsystem_1_1error__code.html">Class <code>system::error&#95;code</code>
</a>
</h3>

The class <code>error&#95;code</code> describes an object used to hold error code values, such as those originating from the operating system or other low-level application program interfaces. 

<h3 id="class-system::error_condition">
<a href="/api/classes/classsystem_1_1error__condition.html">Class <code>system::error&#95;condition</code>
</a>
</h3>

The class <code>error&#95;condition</code> describes an object used to hold values identifying error conditions. 

<h3 id="class-system::system_error">
<a href="/api/classes/classsystem_1_1system__error.html">Class <code>system::system&#95;error</code>
</a>
</h3>

The class <code>system&#95;error</code> describes an exception object used to report error conditions that have an associated <code><a href="/api/classes/classsystem_1_1error__code.html">error&#95;code</a></code>. Such error conditions typically originate from the operating system or other low-level application program interfaces. 

**Inherits From**:
`runtime_error`


## Functions

<h3 id="function-generic_category">
Function <code>system::generic&#95;category</code>
</h3>

<code class="doxybook">
<span>const <a href="/api/classes/classsystem_1_1error__category.html">error_category</a> & </span><span><b>generic_category</b>(void);</span></code>
**Note**:
The object's <code>default&#95;error&#95;condition</code> and <code>equivalent</code> virtual functions shall behave as specified for the class <code><a href="/api/classes/classsystem_1_1error__category.html">error&#95;category</a></code>. The object's <code>name</code> virtual function shall return a pointer to the string <code>"generic"</code>. 

**Returns**:
A reference to an object of a type derived from class <code><a href="/api/classes/classsystem_1_1error__category.html">error&#95;category</a></code>. 

<h3 id="function-system_category">
Function <code>system::system&#95;category</code>
</h3>

<code class="doxybook">
<span>const <a href="/api/classes/classsystem_1_1error__category.html">error_category</a> & </span><span><b>system_category</b>(void);</span></code>

If the argument <code>ev</code> corresponds to a POSIX <code>errno</code> value <code>posv</code>, the function shall return <code>error&#95;condition(ev,generic&#95;category())</code>. Otherwise, the function shall return <code>error&#95;condition(ev,system&#95;category())</code>. What constitutes correspondence for any given operating system is unspecified. 

**Note**:
The object's <code>equivalent</code> virtual functions shall behave as specified for class <code><a href="/api/classes/classsystem_1_1error__category.html">error&#95;category</a></code>. The object's <code>name</code> virtual function shall return a pointer to the string <code>"system"</code>. The object's <code>default&#95;error&#95;condition</code> virtual function shall behave as follows:

**Returns**:
A reference to an object of a type derived from class <code><a href="/api/classes/classsystem_1_1error__category.html">error&#95;category</a></code>. 

<h3 id="function-make_error_code">
Function <code>system::make&#95;error&#95;code</code>
</h3>

<code class="doxybook">
<span><a href="/api/classes/classsystem_1_1error__code.html">error_code</a> </span><span><b>make_error_code</b>(errc::errc_t e);</span></code>
**Returns**:
<code><a href="/api/classes/classsystem_1_1error__code.html">error&#95;code</a>(static&#95;cast&lt;int&gt;(e), <a href="/api/groups/group__system__diagnostics.html#function-generic_category">generic&#95;category()</a>)</code>

<h3 id="function-operator<">
Function <code>system::operator&lt;</code>
</h3>

<code class="doxybook">
<span>bool </span><span><b>operator<</b>(const <a href="/api/classes/classsystem_1_1error__code.html">error_code</a> & lhs,</span>
<span>&nbsp;&nbsp;const <a href="/api/classes/classsystem_1_1error__code.html">error_code</a> & rhs);</span></code>
**Returns**:
<code>lhs.category() &lt; rhs.category() || lhs.category() == rhs.category() && lhs.value() &lt; rhs.value()</code>. 

<h3 id="function-operator<<">
Function <code>system::operator&lt;&lt;</code>
</h3>

<code class="doxybook">
<span>template &lt;typename charT,</span>
<span>&nbsp;&nbsp;typename traits&gt;</span>
<span>std::basic_ostream< charT, traits > & </span><span><b>operator<<</b>(std::basic_ostream< charT, traits > & os,</span>
<span>&nbsp;&nbsp;const <a href="/api/classes/classsystem_1_1error__code.html">error_code</a> & ec);</span></code>
Effects: <code>os &lt;&lt; ec.category().name() &lt;&lt; ':' &lt;&lt; ec.value()</code>. 

<h3 id="function-make_error_condition">
Function <code>system::make&#95;error&#95;condition</code>
</h3>

<code class="doxybook">
<span><a href="/api/classes/classsystem_1_1error__condition.html">error_condition</a> </span><span><b>make_error_condition</b>(errc::errc_t e);</span></code>
**Returns**:
<code><a href="/api/classes/classsystem_1_1error__condition.html">error&#95;condition</a>(static&#95;cast&lt;int&gt;(e), <a href="/api/groups/group__system__diagnostics.html#function-generic_category">generic&#95;category()</a>)</code>. 

<h3 id="function-operator<">
Function <code>system::operator&lt;</code>
</h3>

<code class="doxybook">
<span>bool </span><span><b>operator<</b>(const <a href="/api/classes/classsystem_1_1error__condition.html">error_condition</a> & lhs,</span>
<span>&nbsp;&nbsp;const <a href="/api/classes/classsystem_1_1error__condition.html">error_condition</a> & rhs);</span></code>
**Returns**:
<code>lhs.category() &lt; rhs.category() || lhs.category() == rhs.category() && lhs.value() &lt; rhs.value()</code>. 

<h3 id="function-operator==">
Function <code>system::operator==</code>
</h3>

<code class="doxybook">
<span>bool </span><span><b>operator==</b>(const <a href="/api/classes/classsystem_1_1error__code.html">error_code</a> & lhs,</span>
<span>&nbsp;&nbsp;const <a href="/api/classes/classsystem_1_1error__code.html">error_code</a> & rhs);</span></code>
**Returns**:
<code>lhs.category() == rhs.category() && lhs.value() == rhs.value()</code>. 

<h3 id="function-operator==">
Function <code>system::operator==</code>
</h3>

<code class="doxybook">
<span>bool </span><span><b>operator==</b>(const <a href="/api/classes/classsystem_1_1error__code.html">error_code</a> & lhs,</span>
<span>&nbsp;&nbsp;const <a href="/api/classes/classsystem_1_1error__condition.html">error_condition</a> & rhs);</span></code>
**Returns**:
<code>lhs.category().equivalent(lhs.value(), rhs) || rhs.category().equivalent(lhs,rhs.value())</code>. 

<h3 id="function-operator==">
Function <code>system::operator==</code>
</h3>

<code class="doxybook">
<span>bool </span><span><b>operator==</b>(const <a href="/api/classes/classsystem_1_1error__condition.html">error_condition</a> & lhs,</span>
<span>&nbsp;&nbsp;const <a href="/api/classes/classsystem_1_1error__code.html">error_code</a> & rhs);</span></code>
**Returns**:
<code>rhs.category().equivalent(lhs.value(), lhs) || lhs.category().equivalent(rhs, lhs.value())</code>. 

<h3 id="function-operator==">
Function <code>system::operator==</code>
</h3>

<code class="doxybook">
<span>bool </span><span><b>operator==</b>(const <a href="/api/classes/classsystem_1_1error__condition.html">error_condition</a> & lhs,</span>
<span>&nbsp;&nbsp;const <a href="/api/classes/classsystem_1_1error__condition.html">error_condition</a> & rhs);</span></code>
**Returns**:
<code>lhs.category() == rhs.category() && lhs.value() == rhs.value()</code>

<h3 id="function-operator!=">
Function <code>system::operator!=</code>
</h3>

<code class="doxybook">
<span>bool </span><span><b>operator!=</b>(const <a href="/api/classes/classsystem_1_1error__code.html">error_code</a> & lhs,</span>
<span>&nbsp;&nbsp;const <a href="/api/classes/classsystem_1_1error__code.html">error_code</a> & rhs);</span></code>
**Returns**:
<code>!(lhs == rhs)</code>

<h3 id="function-operator!=">
Function <code>system::operator!=</code>
</h3>

<code class="doxybook">
<span>bool </span><span><b>operator!=</b>(const <a href="/api/classes/classsystem_1_1error__code.html">error_code</a> & lhs,</span>
<span>&nbsp;&nbsp;const <a href="/api/classes/classsystem_1_1error__condition.html">error_condition</a> & rhs);</span></code>
**Returns**:
<code>!(lhs == rhs)</code>

<h3 id="function-operator!=">
Function <code>system::operator!=</code>
</h3>

<code class="doxybook">
<span>bool </span><span><b>operator!=</b>(const <a href="/api/classes/classsystem_1_1error__condition.html">error_condition</a> & lhs,</span>
<span>&nbsp;&nbsp;const <a href="/api/classes/classsystem_1_1error__code.html">error_code</a> & rhs);</span></code>
**Returns**:
<code>!(lhs == rhs)</code>

<h3 id="function-operator!=">
Function <code>system::operator!=</code>
</h3>

<code class="doxybook">
<span>bool </span><span><b>operator!=</b>(const <a href="/api/classes/classsystem_1_1error__condition.html">error_condition</a> & lhs,</span>
<span>&nbsp;&nbsp;const <a href="/api/classes/classsystem_1_1error__condition.html">error_condition</a> & rhs);</span></code>
**Returns**:
<code>!(lhs == rhs)</code>

<h3 id="function-system_error">
Function <code>system::system&#95;error::system&#95;error</code>
</h3>

<code class="doxybook">
<span><b>system_error</b>(<a href="/api/classes/classsystem_1_1error__code.html">error_code</a> ec,</span>
<span>&nbsp;&nbsp;const std::string & what_arg);</span></code>
Constructs an object of class <code><a href="/api/classes/classsystem_1_1system__error.html">system&#95;error</a></code>. 

**Function Parameters**:
* **`ec`** The value returned by <code><a href="/api/groups/group__system__diagnostics.html#function-code">code()</a></code>. 
* **`what_arg`** A string to include in the result returned by <code><a href="/api/groups/group__system__diagnostics.html#function-what">what()</a></code>. 

**Postconditions**:
* <code><a href="/api/groups/group__system__diagnostics.html#function-code">code()</a> == ec</code>. 
* <code>std::string(what()).find(what&#95;arg) != string::npos</code>. 

<h3 id="function-system_error">
Function <code>system::system&#95;error::system&#95;error</code>
</h3>

<code class="doxybook">
<span><b>system_error</b>(<a href="/api/classes/classsystem_1_1error__code.html">error_code</a> ec,</span>
<span>&nbsp;&nbsp;const char * what_arg);</span></code>
Constructs an object of class <code><a href="/api/classes/classsystem_1_1system__error.html">system&#95;error</a></code>. 

**Function Parameters**:
* **`ec`** The value returned by <code><a href="/api/groups/group__system__diagnostics.html#function-code">code()</a></code>. 
* **`what_arg`** A string to include in the result returned by <code><a href="/api/groups/group__system__diagnostics.html#function-what">what()</a></code>. 

**Postconditions**:
* <code><a href="/api/groups/group__system__diagnostics.html#function-code">code()</a> == ec</code>. 
* <code>std::string(what()).find(what&#95;arg) != string::npos</code>. 

<h3 id="function-system_error">
Function <code>system::system&#95;error::system&#95;error</code>
</h3>

<code class="doxybook">
<span><b>system_error</b>(<a href="/api/classes/classsystem_1_1error__code.html">error_code</a> ec);</span></code>
Constructs an object of class <code><a href="/api/classes/classsystem_1_1system__error.html">system&#95;error</a></code>. 

**Function Parameters**:
**`ec`**: The value returned by <code><a href="/api/groups/group__system__diagnostics.html#function-code">code()</a></code>. 

**Postconditions**:
<code><a href="/api/groups/group__system__diagnostics.html#function-code">code()</a> == ec</code>. 

<h3 id="function-system_error">
Function <code>system::system&#95;error::system&#95;error</code>
</h3>

<code class="doxybook">
<span><b>system_error</b>(int ev,</span>
<span>&nbsp;&nbsp;const <a href="/api/classes/classsystem_1_1error__category.html">error_category</a> & ecat,</span>
<span>&nbsp;&nbsp;const std::string & what_arg);</span></code>
Constructs an object of class <code><a href="/api/classes/classsystem_1_1system__error.html">system&#95;error</a></code>. 

**Function Parameters**:
* **`ev`** The error value used to create an <code><a href="/api/classes/classsystem_1_1error__code.html">error&#95;code</a></code>. 
* **`ecat`** The <code><a href="/api/classes/classsystem_1_1error__category.html">error&#95;category</a></code> used to create an <code><a href="/api/classes/classsystem_1_1error__code.html">error&#95;code</a></code>. 
* **`what_arg`** A string to include in the result returned by <code><a href="/api/groups/group__system__diagnostics.html#function-what">what()</a></code>. 

**Postconditions**:
* <code><a href="/api/groups/group__system__diagnostics.html#function-code">code()</a> == error&#95;code(ev, ecat)</code>. 
* <code>std::string(what()).find(what&#95;arg) != string::npos</code>. 

<h3 id="function-system_error">
Function <code>system::system&#95;error::system&#95;error</code>
</h3>

<code class="doxybook">
<span><b>system_error</b>(int ev,</span>
<span>&nbsp;&nbsp;const <a href="/api/classes/classsystem_1_1error__category.html">error_category</a> & ecat,</span>
<span>&nbsp;&nbsp;const char * what_arg);</span></code>
Constructs an object of class <code><a href="/api/classes/classsystem_1_1system__error.html">system&#95;error</a></code>. 

**Function Parameters**:
* **`ev`** The error value used to create an <code><a href="/api/classes/classsystem_1_1error__code.html">error&#95;code</a></code>. 
* **`ecat`** The <code><a href="/api/classes/classsystem_1_1error__category.html">error&#95;category</a></code> used to create an <code><a href="/api/classes/classsystem_1_1error__code.html">error&#95;code</a></code>. 
* **`what_arg`** A string to include in the result returned by <code><a href="/api/groups/group__system__diagnostics.html#function-what">what()</a></code>. 

**Postconditions**:
* <code><a href="/api/groups/group__system__diagnostics.html#function-code">code()</a> == error&#95;code(ev, ecat)</code>. 
* <code>std::string(what()).find(what&#95;arg) != string::npos</code>. 

<h3 id="function-system_error">
Function <code>system::system&#95;error::system&#95;error</code>
</h3>

<code class="doxybook">
<span><b>system_error</b>(int ev,</span>
<span>&nbsp;&nbsp;const <a href="/api/classes/classsystem_1_1error__category.html">error_category</a> & ecat);</span></code>
Constructs an object of class <code><a href="/api/classes/classsystem_1_1system__error.html">system&#95;error</a></code>. 

**Function Parameters**:
* **`ev`** The error value used to create an <code><a href="/api/classes/classsystem_1_1error__code.html">error&#95;code</a></code>. 
* **`ecat`** The <code><a href="/api/classes/classsystem_1_1error__category.html">error&#95;category</a></code> used to create an <code><a href="/api/classes/classsystem_1_1error__code.html">error&#95;code</a></code>. 

**Postconditions**:
<code><a href="/api/groups/group__system__diagnostics.html#function-code">code()</a> == error&#95;code(ev, ecat)</code>. 

<h3 id="function-~system_error">
Function <code>system::system&#95;error::~system&#95;error</code>
</h3>

<code class="doxybook">
<span>virtual </span><span><b>~system_error</b>(void);</span></code>
Destructor does not throw. 

<h3 id="function-code">
Function <code>system::system&#95;error::code</code>
</h3>

<code class="doxybook">
<span>const <a href="/api/classes/classsystem_1_1error__code.html">error_code</a> & </span><span><b>code</b>(void) const;</span></code>
Returns an object encoding the error. 

**Returns**:
<code>ec</code> or <code>error&#95;code(ev, ecat)</code>, from the constructor, as appropriate. 

<h3 id="function-what">
Function <code>system::system&#95;error::what</code>
</h3>

<code class="doxybook">
<span>const char * </span><span><b>what</b>(void) const;</span></code>
Returns a human-readable string indicating the nature of the error. 

**Returns**:
a string incorporating <code><a href="/api/groups/group__system__diagnostics.html#function-code">code()</a>.message()</code> and the arguments supplied in the constructor. 


