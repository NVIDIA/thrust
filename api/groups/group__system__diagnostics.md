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
<span>namespace <b><a href="{{ site.baseurl }}/api/namespaces/namespacethrust_1_1system_1_1errc.html">thrust::system::errc</a></b> { <i>…</i> }</span>
<br>
<span>template &lt;typename T&gt;</span>
<span>struct <b><a href="{{ site.baseurl }}/api/classes/structthrust_1_1system_1_1is__error__code__enum.html">thrust::system::is&#95;error&#95;code&#95;enum</a></b>;</span>
<br>
<span>template &lt;typename T&gt;</span>
<span>struct <b><a href="{{ site.baseurl }}/api/classes/structthrust_1_1system_1_1is__error__condition__enum.html">thrust::system::is&#95;error&#95;condition&#95;enum</a></b>;</span>
<br>
<span>struct <b><a href="{{ site.baseurl }}/api/classes/structthrust_1_1system_1_1is__error__condition__enum_3_01errc_1_1errc__t_01_4.html">thrust::system::is&#95;error&#95;condition&#95;enum&lt; errc::errc&#95;t &gt;</a></b>;</span>
<br>
<span class="doxybook-comment">/* The class <code>error&#95;category</code> serves as a base class for types used to identify the source and encoding of a particular category of error code. Classes may be derived from <code>error&#95;category</code> to support categories of errors in addition to those defined in the C++ International Standard.  */</span><span>class <b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1system_1_1error__category.html">thrust::system::error&#95;category</a></b>;</span>
<br>
<span class="doxybook-comment">/* The class <code>error&#95;code</code> describes an object used to hold error code values, such as those originating from the operating system or other low-level application program interfaces.  */</span><span>class <b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1system_1_1error__code.html">thrust::system::error&#95;code</a></b>;</span>
<br>
<span class="doxybook-comment">/* The class <code>error&#95;condition</code> describes an object used to hold values identifying error conditions.  */</span><span>class <b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1system_1_1error__condition.html">thrust::system::error&#95;condition</a></b>;</span>
<br>
<span class="doxybook-comment">/* The class <code>system&#95;error</code> describes an exception object used to report error conditions that have an associated <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1system_1_1error__code.html">error&#95;code</a></code>. Such error conditions typically originate from the operating system or other low-level application program interfaces.  */</span><span>class <b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1system_1_1system__error.html">thrust::system::system&#95;error</a></b>;</span>
<br>
<span>const error_category & </span><span><b><a href="{{ site.baseurl }}/api/groups/group__system__diagnostics.html#function-generic-category">thrust::system::generic&#95;category</a></b>(void);</span>
<br>
<span>const error_category & </span><span><b><a href="{{ site.baseurl }}/api/groups/group__system__diagnostics.html#function-system-category">thrust::system::system&#95;category</a></b>(void);</span>
<br>
<span>error_code </span><span><b><a href="{{ site.baseurl }}/api/groups/group__system__diagnostics.html#function-make-error-code">thrust::system::make&#95;error&#95;code</a></b>(cuda::errc::errc_t e);</span>
<br>
<span>bool </span><span><b><a href="{{ site.baseurl }}/api/groups/group__system__diagnostics.html#function-operator<">thrust::system::operator&lt;</a></b>(const error_code & lhs,</span>
<span>&nbsp;&nbsp;const error_code & rhs);</span>
<br>
<span>template &lt;typename charT,</span>
<span>&nbsp;&nbsp;typename traits&gt;</span>
<span>std::basic_ostream< charT, traits > & </span><span><b><a href="{{ site.baseurl }}/api/groups/group__system__diagnostics.html#function-operator<<">thrust::system::operator&lt;&lt;</a></b>(std::basic_ostream< charT, traits > & os,</span>
<span>&nbsp;&nbsp;const error_code & ec);</span>
<br>
<span>error_condition </span><span><b><a href="{{ site.baseurl }}/api/groups/group__system__diagnostics.html#function-make-error-condition">thrust::system::make&#95;error&#95;condition</a></b>(cuda::errc::errc_t e);</span>
<br>
<span>bool </span><span><b><a href="{{ site.baseurl }}/api/groups/group__system__diagnostics.html#function-operator<">thrust::system::operator&lt;</a></b>(const error_condition & lhs,</span>
<span>&nbsp;&nbsp;const error_condition & rhs);</span>
<br>
<span>bool </span><span><b><a href="{{ site.baseurl }}/api/groups/group__system__diagnostics.html#function-operator==">thrust::system::operator==</a></b>(const error_code & lhs,</span>
<span>&nbsp;&nbsp;const error_code & rhs);</span>
<br>
<span>bool </span><span><b><a href="{{ site.baseurl }}/api/groups/group__system__diagnostics.html#function-operator==">thrust::system::operator==</a></b>(const error_code & lhs,</span>
<span>&nbsp;&nbsp;const error_condition & rhs);</span>
<br>
<span>bool </span><span><b><a href="{{ site.baseurl }}/api/groups/group__system__diagnostics.html#function-operator==">thrust::system::operator==</a></b>(const error_condition & lhs,</span>
<span>&nbsp;&nbsp;const error_code & rhs);</span>
<br>
<span>bool </span><span><b><a href="{{ site.baseurl }}/api/groups/group__system__diagnostics.html#function-operator==">thrust::system::operator==</a></b>(const error_condition & lhs,</span>
<span>&nbsp;&nbsp;const error_condition & rhs);</span>
<br>
<span>bool </span><span><b><a href="{{ site.baseurl }}/api/groups/group__system__diagnostics.html#function-operator!=">thrust::system::operator!=</a></b>(const error_code & lhs,</span>
<span>&nbsp;&nbsp;const error_code & rhs);</span>
<br>
<span>bool </span><span><b><a href="{{ site.baseurl }}/api/groups/group__system__diagnostics.html#function-operator!=">thrust::system::operator!=</a></b>(const error_code & lhs,</span>
<span>&nbsp;&nbsp;const error_condition & rhs);</span>
<br>
<span>bool </span><span><b><a href="{{ site.baseurl }}/api/groups/group__system__diagnostics.html#function-operator!=">thrust::system::operator!=</a></b>(const error_condition & lhs,</span>
<span>&nbsp;&nbsp;const error_code & rhs);</span>
<br>
<span>bool </span><span><b><a href="{{ site.baseurl }}/api/groups/group__system__diagnostics.html#function-operator!=">thrust::system::operator!=</a></b>(const error_condition & lhs,</span>
<span>&nbsp;&nbsp;const error_condition & rhs);</span>
<br>
<span>const error_code & </span><span><b><a href="{{ site.baseurl }}/api/groups/group__system__diagnostics.html#function-code">thrust::system::system&#95;error::code</a></b>(void) const;</span>
<br>
<span>const char * </span><span><b><a href="{{ site.baseurl }}/api/groups/group__system__diagnostics.html#function-what">thrust::system::system&#95;error::what</a></b>(void) const;</span>
</code>

## Member Classes

<h3 id="struct-thrustsystemis-error-code-enum">
<a href="{{ site.baseurl }}/api/classes/structthrust_1_1system_1_1is__error__code__enum.html">Struct <code>thrust::system::is&#95;error&#95;code&#95;enum</code>
</a>
</h3>

**Inherits From**:
`thrust::detail::false_type`

<h3 id="struct-thrustsystemis-error-condition-enum">
<a href="{{ site.baseurl }}/api/classes/structthrust_1_1system_1_1is__error__condition__enum.html">Struct <code>thrust::system::is&#95;error&#95;condition&#95;enum</code>
</a>
</h3>

**Inherits From**:
`thrust::detail::false_type`

<h3 id="struct-thrustsystemis-error-condition-enum<-errcerrc-t->">
<a href="{{ site.baseurl }}/api/classes/structthrust_1_1system_1_1is__error__condition__enum_3_01errc_1_1errc__t_01_4.html">Struct <code>thrust::system::is&#95;error&#95;condition&#95;enum&lt; errc::errc&#95;t &gt;</code>
</a>
</h3>

**Inherits From**:
`thrust::detail::true_type`

<h3 id="class-thrustsystemerror-category">
<a href="{{ site.baseurl }}/api/classes/classthrust_1_1system_1_1error__category.html">Class <code>thrust::system::error&#95;category</code>
</a>
</h3>

The class <code>error&#95;category</code> serves as a base class for types used to identify the source and encoding of a particular category of error code. Classes may be derived from <code>error&#95;category</code> to support categories of errors in addition to those defined in the C++ International Standard. 

<h3 id="class-thrustsystemerror-code">
<a href="{{ site.baseurl }}/api/classes/classthrust_1_1system_1_1error__code.html">Class <code>thrust::system::error&#95;code</code>
</a>
</h3>

The class <code>error&#95;code</code> describes an object used to hold error code values, such as those originating from the operating system or other low-level application program interfaces. 

<h3 id="class-thrustsystemerror-condition">
<a href="{{ site.baseurl }}/api/classes/classthrust_1_1system_1_1error__condition.html">Class <code>thrust::system::error&#95;condition</code>
</a>
</h3>

The class <code>error&#95;condition</code> describes an object used to hold values identifying error conditions. 

<h3 id="class-thrustsystemsystem-error">
<a href="{{ site.baseurl }}/api/classes/classthrust_1_1system_1_1system__error.html">Class <code>thrust::system::system&#95;error</code>
</a>
</h3>

The class <code>system&#95;error</code> describes an exception object used to report error conditions that have an associated <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1system_1_1error__code.html">error&#95;code</a></code>. Such error conditions typically originate from the operating system or other low-level application program interfaces. 

**Inherits From**:
`std::runtime_error`


## Functions

<h3 id="function-generic-category">
Function <code>thrust::system::generic&#95;category</code>
</h3>

<code class="doxybook">
<span>const error_category & </span><span><b>generic_category</b>(void);</span></code>
**Note**:
The object's <code>default&#95;error&#95;condition</code> and <code>equivalent</code> virtual functions shall behave as specified for the class <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1system_1_1error__category.html">error&#95;category</a></code>. The object's <code>name</code> virtual function shall return a pointer to the string <code>"generic"</code>. 

**Returns**:
A reference to an object of a type derived from class <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1system_1_1error__category.html">error&#95;category</a></code>. 

<h3 id="function-system-category">
Function <code>thrust::system::system&#95;category</code>
</h3>

<code class="doxybook">
<span>const error_category & </span><span><b>system_category</b>(void);</span></code>

If the argument <code>ev</code> corresponds to a POSIX <code>errno</code> value <code>posv</code>, the function shall return <code>error&#95;condition(ev,generic&#95;category())</code>. Otherwise, the function shall return <code>error&#95;condition(ev,system&#95;category())</code>. What constitutes correspondence for any given operating system is unspecified. 

**Note**:
The object's <code>equivalent</code> virtual functions shall behave as specified for class <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1system_1_1error__category.html">error&#95;category</a></code>. The object's <code>name</code> virtual function shall return a pointer to the string <code>"system"</code>. The object's <code>default&#95;error&#95;condition</code> virtual function shall behave as follows:

**Returns**:
A reference to an object of a type derived from class <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1system_1_1error__category.html">error&#95;category</a></code>. 

<h3 id="function-make-error-code">
Function <code>thrust::system::make&#95;error&#95;code</code>
</h3>

<code class="doxybook">
<span>error_code </span><span><b>make_error_code</b>(cuda::errc::errc_t e);</span></code>
**Returns**:
* <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1system_1_1error__code.html">error&#95;code</a>(static&#95;cast&lt;int&gt;(e), cuda::error&#95;category())</code>
* <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1system_1_1error__code.html">error&#95;code</a>(static&#95;cast&lt;int&gt;(e), <a href="{{ site.baseurl }}/api/groups/group__system__diagnostics.html#function-generic-category">generic&#95;category()</a>)</code>

<h3 id="function-operator<">
Function <code>thrust::system::operator&lt;</code>
</h3>

<code class="doxybook">
<span>bool </span><span><b>operator<</b>(const error_code & lhs,</span>
<span>&nbsp;&nbsp;const error_code & rhs);</span></code>
**Returns**:
<code>lhs.category() &lt; rhs.category() || lhs.category() == rhs.category() && lhs.value() &lt; rhs.value()</code>. 

<h3 id="function-operator<<">
Function <code>thrust::system::operator&lt;&lt;</code>
</h3>

<code class="doxybook">
<span>template &lt;typename charT,</span>
<span>&nbsp;&nbsp;typename traits&gt;</span>
<span>std::basic_ostream< charT, traits > & </span><span><b>operator<<</b>(std::basic_ostream< charT, traits > & os,</span>
<span>&nbsp;&nbsp;const error_code & ec);</span></code>
Effects: <code>os &lt;&lt; ec.category().name() &lt;&lt; ':' &lt;&lt; ec.value()</code>. 

<h3 id="function-make-error-condition">
Function <code>thrust::system::make&#95;error&#95;condition</code>
</h3>

<code class="doxybook">
<span>error_condition </span><span><b>make_error_condition</b>(cuda::errc::errc_t e);</span></code>
**Returns**:
* <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1system_1_1error__condition.html">error&#95;condition</a>(static&#95;cast&lt;int&gt;(e), cuda::error&#95;category())</code>.
* <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1system_1_1error__condition.html">error&#95;condition</a>(static&#95;cast&lt;int&gt;(e), <a href="{{ site.baseurl }}/api/groups/group__system__diagnostics.html#function-generic-category">generic&#95;category()</a>)</code>. 

<h3 id="function-operator<">
Function <code>thrust::system::operator&lt;</code>
</h3>

<code class="doxybook">
<span>bool </span><span><b>operator<</b>(const error_condition & lhs,</span>
<span>&nbsp;&nbsp;const error_condition & rhs);</span></code>
**Returns**:
<code>lhs.category() &lt; rhs.category() || lhs.category() == rhs.category() && lhs.value() &lt; rhs.value()</code>. 

<h3 id="function-operator==">
Function <code>thrust::system::operator==</code>
</h3>

<code class="doxybook">
<span>bool </span><span><b>operator==</b>(const error_code & lhs,</span>
<span>&nbsp;&nbsp;const error_code & rhs);</span></code>
**Returns**:
<code>lhs.category() == rhs.category() && lhs.value() == rhs.value()</code>. 

<h3 id="function-operator==">
Function <code>thrust::system::operator==</code>
</h3>

<code class="doxybook">
<span>bool </span><span><b>operator==</b>(const error_code & lhs,</span>
<span>&nbsp;&nbsp;const error_condition & rhs);</span></code>
**Returns**:
<code>lhs.category().equivalent(lhs.value(), rhs) || rhs.category().equivalent(lhs,rhs.value())</code>. 

<h3 id="function-operator==">
Function <code>thrust::system::operator==</code>
</h3>

<code class="doxybook">
<span>bool </span><span><b>operator==</b>(const error_condition & lhs,</span>
<span>&nbsp;&nbsp;const error_code & rhs);</span></code>
**Returns**:
<code>rhs.category().equivalent(lhs.value(), lhs) || lhs.category().equivalent(rhs, lhs.value())</code>. 

<h3 id="function-operator==">
Function <code>thrust::system::operator==</code>
</h3>

<code class="doxybook">
<span>bool </span><span><b>operator==</b>(const error_condition & lhs,</span>
<span>&nbsp;&nbsp;const error_condition & rhs);</span></code>
**Returns**:
<code>lhs.category() == rhs.category() && lhs.value() == rhs.value()</code>

<h3 id="function-operator!=">
Function <code>thrust::system::operator!=</code>
</h3>

<code class="doxybook">
<span>bool </span><span><b>operator!=</b>(const error_code & lhs,</span>
<span>&nbsp;&nbsp;const error_code & rhs);</span></code>
**Returns**:
<code>!(lhs == rhs)</code>

<h3 id="function-operator!=">
Function <code>thrust::system::operator!=</code>
</h3>

<code class="doxybook">
<span>bool </span><span><b>operator!=</b>(const error_code & lhs,</span>
<span>&nbsp;&nbsp;const error_condition & rhs);</span></code>
**Returns**:
<code>!(lhs == rhs)</code>

<h3 id="function-operator!=">
Function <code>thrust::system::operator!=</code>
</h3>

<code class="doxybook">
<span>bool </span><span><b>operator!=</b>(const error_condition & lhs,</span>
<span>&nbsp;&nbsp;const error_code & rhs);</span></code>
**Returns**:
<code>!(lhs == rhs)</code>

<h3 id="function-operator!=">
Function <code>thrust::system::operator!=</code>
</h3>

<code class="doxybook">
<span>bool </span><span><b>operator!=</b>(const error_condition & lhs,</span>
<span>&nbsp;&nbsp;const error_condition & rhs);</span></code>
**Returns**:
<code>!(lhs == rhs)</code>

<h3 id="function-system-error">
Function <code>thrust::system::system&#95;error::system&#95;error</code>
</h3>

<code class="doxybook">
<span><b>system_error</b>(error_code ec,</span>
<span>&nbsp;&nbsp;const std::string & what_arg);</span></code>
Constructs an object of class <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1system_1_1system__error.html">system&#95;error</a></code>. 

**Function Parameters**:
* **`ec`** The value returned by <code><a href="{{ site.baseurl }}/api/groups/group__system__diagnostics.html#function-code">code()</a></code>. 
* **`what_arg`** A string to include in the result returned by <code><a href="{{ site.baseurl }}/api/groups/group__system__diagnostics.html#function-what">what()</a></code>. 

**Postconditions**:
* <code><a href="{{ site.baseurl }}/api/groups/group__system__diagnostics.html#function-code">code()</a> == ec</code>. 
* <code>std::string(what()).find(what&#95;arg) != string::npos</code>. 

<h3 id="function-system-error">
Function <code>thrust::system::system&#95;error::system&#95;error</code>
</h3>

<code class="doxybook">
<span><b>system_error</b>(error_code ec,</span>
<span>&nbsp;&nbsp;const char * what_arg);</span></code>
Constructs an object of class <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1system_1_1system__error.html">system&#95;error</a></code>. 

**Function Parameters**:
* **`ec`** The value returned by <code><a href="{{ site.baseurl }}/api/groups/group__system__diagnostics.html#function-code">code()</a></code>. 
* **`what_arg`** A string to include in the result returned by <code><a href="{{ site.baseurl }}/api/groups/group__system__diagnostics.html#function-what">what()</a></code>. 

**Postconditions**:
* <code><a href="{{ site.baseurl }}/api/groups/group__system__diagnostics.html#function-code">code()</a> == ec</code>. 
* <code>std::string(what()).find(what&#95;arg) != string::npos</code>. 

<h3 id="function-system-error">
Function <code>thrust::system::system&#95;error::system&#95;error</code>
</h3>

<code class="doxybook">
<span><b>system_error</b>(error_code ec);</span></code>
Constructs an object of class <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1system_1_1system__error.html">system&#95;error</a></code>. 

**Function Parameters**:
**`ec`**: The value returned by <code><a href="{{ site.baseurl }}/api/groups/group__system__diagnostics.html#function-code">code()</a></code>. 

**Postconditions**:
<code><a href="{{ site.baseurl }}/api/groups/group__system__diagnostics.html#function-code">code()</a> == ec</code>. 

<h3 id="function-system-error">
Function <code>thrust::system::system&#95;error::system&#95;error</code>
</h3>

<code class="doxybook">
<span><b>system_error</b>(int ev,</span>
<span>&nbsp;&nbsp;const error_category & ecat,</span>
<span>&nbsp;&nbsp;const std::string & what_arg);</span></code>
Constructs an object of class <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1system_1_1system__error.html">system&#95;error</a></code>. 

**Function Parameters**:
* **`ev`** The error value used to create an <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1system_1_1error__code.html">error&#95;code</a></code>. 
* **`ecat`** The <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1system_1_1error__category.html">error&#95;category</a></code> used to create an <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1system_1_1error__code.html">error&#95;code</a></code>. 
* **`what_arg`** A string to include in the result returned by <code><a href="{{ site.baseurl }}/api/groups/group__system__diagnostics.html#function-what">what()</a></code>. 

**Postconditions**:
* <code><a href="{{ site.baseurl }}/api/groups/group__system__diagnostics.html#function-code">code()</a> == error&#95;code(ev, ecat)</code>. 
* <code>std::string(what()).find(what&#95;arg) != string::npos</code>. 

<h3 id="function-system-error">
Function <code>thrust::system::system&#95;error::system&#95;error</code>
</h3>

<code class="doxybook">
<span><b>system_error</b>(int ev,</span>
<span>&nbsp;&nbsp;const error_category & ecat,</span>
<span>&nbsp;&nbsp;const char * what_arg);</span></code>
Constructs an object of class <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1system_1_1system__error.html">system&#95;error</a></code>. 

**Function Parameters**:
* **`ev`** The error value used to create an <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1system_1_1error__code.html">error&#95;code</a></code>. 
* **`ecat`** The <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1system_1_1error__category.html">error&#95;category</a></code> used to create an <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1system_1_1error__code.html">error&#95;code</a></code>. 
* **`what_arg`** A string to include in the result returned by <code><a href="{{ site.baseurl }}/api/groups/group__system__diagnostics.html#function-what">what()</a></code>. 

**Postconditions**:
* <code><a href="{{ site.baseurl }}/api/groups/group__system__diagnostics.html#function-code">code()</a> == error&#95;code(ev, ecat)</code>. 
* <code>std::string(what()).find(what&#95;arg) != string::npos</code>. 

<h3 id="function-system-error">
Function <code>thrust::system::system&#95;error::system&#95;error</code>
</h3>

<code class="doxybook">
<span><b>system_error</b>(int ev,</span>
<span>&nbsp;&nbsp;const error_category & ecat);</span></code>
Constructs an object of class <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1system_1_1system__error.html">system&#95;error</a></code>. 

**Function Parameters**:
* **`ev`** The error value used to create an <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1system_1_1error__code.html">error&#95;code</a></code>. 
* **`ecat`** The <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1system_1_1error__category.html">error&#95;category</a></code> used to create an <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1system_1_1error__code.html">error&#95;code</a></code>. 

**Postconditions**:
<code><a href="{{ site.baseurl }}/api/groups/group__system__diagnostics.html#function-code">code()</a> == error&#95;code(ev, ecat)</code>. 

<h3 id="function-~system-error">
Function <code>thrust::system::system&#95;error::~system&#95;error</code>
</h3>

<code class="doxybook">
<span>virtual </span><span><b>~system_error</b>(void);</span></code>
Destructor does not throw. 

<h3 id="function-code">
Function <code>thrust::system::system&#95;error::code</code>
</h3>

<code class="doxybook">
<span>const error_code & </span><span><b>code</b>(void) const;</span></code>
Returns an object encoding the error. 

**Returns**:
<code>ec</code> or <code>error&#95;code(ev, ecat)</code>, from the constructor, as appropriate. 

<h3 id="function-what">
Function <code>thrust::system::system&#95;error::what</code>
</h3>

<code class="doxybook">
<span>const char * </span><span><b>what</b>(void) const;</span></code>
Returns a human-readable string indicating the nature of the error. 

**Returns**:
a string incorporating <code><a href="{{ site.baseurl }}/api/groups/group__system__diagnostics.html#function-code">code()</a>.message()</code> and the arguments supplied in the constructor. 


