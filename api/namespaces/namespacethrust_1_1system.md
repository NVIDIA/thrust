---
title: thrust::system
summary: thrust::system is the namespace which contains specific Thrust backend systems. It also contains functionality for reporting error conditions originating from the operating system or other low-level application program interfaces such as the CUDA runtime. They are provided in a separate namespace for import convenience but are also aliased in the top-level thrust namespace for easy access. 
parent: Systems
grand_parent: System
nav_exclude: true
has_children: true
has_toc: false
---

# Namespace `thrust::system`

<code class="doxybook">
<span>namespace thrust::system {</span>
<br>
<span>namespace <b><a href="{{ site.baseurl }}/api/namespaces/namespacethrust_1_1system_1_1____thrust__device__system__namespace.html">thrust::system::&#95;&#95;THRUST&#95;DEVICE&#95;SYSTEM&#95;NAMESPACE</a></b> { <i>…</i> }</span>
<br>
<span>namespace <b><a href="{{ site.baseurl }}/api/namespaces/namespacethrust_1_1system_1_1____thrust__host__system__namespace.html">thrust::system::&#95;&#95;THRUST&#95;HOST&#95;SYSTEM&#95;NAMESPACE</a></b> { <i>…</i> }</span>
<br>
<span>namespace <b><a href="{{ site.baseurl }}/api/namespaces/namespacethrust_1_1system_1_1cpp.html">thrust::system::cpp</a></b> { <i>…</i> }</span>
<br>
<span class="doxybook-comment">/* <code>thrust::system::cuda</code> is the namespace containing functionality for allocating, manipulating, and deallocating memory available to Thrust's CUDA backend system. The identifiers are provided in a separate namespace underneath <code><a href="{{ site.baseurl }}/api/namespaces/namespacethrust_1_1system.html">thrust::system</a></code> for import convenience but are also aliased in the top-level <code><a href="{{ site.baseurl }}/api/namespaces/namespacethrust_1_1cuda.html">thrust::cuda</a></code> namespace for easy access.  */</span><span>namespace <b><a href="{{ site.baseurl }}/api/namespaces/namespacethrust_1_1system_1_1cuda.html">thrust::system::cuda</a></b> { <i>…</i> }</span>
<br>
<span>namespace <b><a href="{{ site.baseurl }}/api/namespaces/namespacethrust_1_1system_1_1errc.html">thrust::system::errc</a></b> { <i>…</i> }</span>
<br>
<span>namespace <b><a href="{{ site.baseurl }}/api/namespaces/namespacethrust_1_1system_1_1omp.html">thrust::system::omp</a></b> { <i>…</i> }</span>
<br>
<span>namespace <b><a href="{{ site.baseurl }}/api/namespaces/namespacethrust_1_1system_1_1tbb.html">thrust::system::tbb</a></b> { <i>…</i> }</span>
<br>
<span class="doxybook-comment">/* The class <code>error&#95;category</code> serves as a base class for types used to identify the source and encoding of a particular category of error code. Classes may be derived from <code>error&#95;category</code> to support categories of errors in addition to those defined in the C++ International Standard.  */</span><span>class <b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1system_1_1error__category.html">error&#95;category</a></b>;</span>
<br>
<span class="doxybook-comment">/* The class <code>error&#95;code</code> describes an object used to hold error code values, such as those originating from the operating system or other low-level application program interfaces.  */</span><span>class <b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1system_1_1error__code.html">error&#95;code</a></b>;</span>
<br>
<span class="doxybook-comment">/* The class <code>error&#95;condition</code> describes an object used to hold values identifying error conditions.  */</span><span>class <b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1system_1_1error__condition.html">error&#95;condition</a></b>;</span>
<br>
<span>template &lt;typename T&gt;</span>
<span>struct <b><a href="{{ site.baseurl }}/api/classes/structthrust_1_1system_1_1is__error__code__enum.html">is&#95;error&#95;code&#95;enum</a></b>;</span>
<br>
<span>struct <b><a href="{{ site.baseurl }}/api/classes/structthrust_1_1system_1_1is__error__code__enum_3_01cuda_1_1errc_1_1errc__t_01_4.html">is&#95;error&#95;code&#95;enum&lt; cuda::errc::errc&#95;t &gt;</a></b>;</span>
<br>
<span>template &lt;typename T&gt;</span>
<span>struct <b><a href="{{ site.baseurl }}/api/classes/structthrust_1_1system_1_1is__error__condition__enum.html">is&#95;error&#95;condition&#95;enum</a></b>;</span>
<br>
<span>struct <b><a href="{{ site.baseurl }}/api/classes/structthrust_1_1system_1_1is__error__condition__enum_3_01errc_1_1errc__t_01_4.html">is&#95;error&#95;condition&#95;enum&lt; errc::errc&#95;t &gt;</a></b>;</span>
<br>
<span class="doxybook-comment">/* The class <code>system&#95;error</code> describes an exception object used to report error conditions that have an associated <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1system_1_1error__code.html">error&#95;code</a></code>. Such error conditions typically originate from the operating system or other low-level application program interfaces.  */</span><span>class <b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1system_1_1system__error.html">system&#95;error</a></b>;</span>
<br>
<span>const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1system_1_1error__category.html">error_category</a> & </span><span><b><a href="{{ site.baseurl }}/api/namespaces/namespacethrust_1_1system.html#function-cuda-category">cuda&#95;category</a></b>(void);</span>
<br>
<span><a href="{{ site.baseurl }}/api/classes/classthrust_1_1system_1_1error__code.html">error_code</a> </span><span><b><a href="{{ site.baseurl }}/api/groups/group__system__diagnostics.html#function-make-error-code">make&#95;error&#95;code</a></b>(<a href="{{ site.baseurl }}/api/namespaces/namespacethrust_1_1system_1_1cuda_1_1errc.html#enum-errc-t">cuda::errc::errc_t</a> e);</span>
<br>
<span><a href="{{ site.baseurl }}/api/classes/classthrust_1_1system_1_1error__condition.html">error_condition</a> </span><span><b><a href="{{ site.baseurl }}/api/groups/group__system__diagnostics.html#function-make-error-condition">make&#95;error&#95;condition</a></b>(<a href="{{ site.baseurl }}/api/namespaces/namespacethrust_1_1system_1_1cuda_1_1errc.html#enum-errc-t">cuda::errc::errc_t</a> e);</span>
<br>
<span>const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1system_1_1error__category.html">error_category</a> & </span><span><b><a href="{{ site.baseurl }}/api/groups/group__system__diagnostics.html#function-generic-category">generic&#95;category</a></b>(void);</span>
<br>
<span>const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1system_1_1error__category.html">error_category</a> & </span><span><b><a href="{{ site.baseurl }}/api/groups/group__system__diagnostics.html#function-system-category">system&#95;category</a></b>(void);</span>
<br>
<span>bool </span><span><b><a href="{{ site.baseurl }}/api/groups/group__system__diagnostics.html#function-operator<">operator&lt;</a></b>(const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1system_1_1error__code.html">error_code</a> & lhs,</span>
<span>&nbsp;&nbsp;const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1system_1_1error__code.html">error_code</a> & rhs);</span>
<br>
<span>template &lt;typename charT,</span>
<span>&nbsp;&nbsp;typename traits&gt;</span>
<span>std::basic_ostream< charT, traits > & </span><span><b><a href="{{ site.baseurl }}/api/groups/group__system__diagnostics.html#function-operator<<">operator&lt;&lt;</a></b>(std::basic_ostream< charT, traits > & os,</span>
<span>&nbsp;&nbsp;const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1system_1_1error__code.html">error_code</a> & ec);</span>
<br>
<span>bool </span><span><b><a href="{{ site.baseurl }}/api/groups/group__system__diagnostics.html#function-operator<">operator&lt;</a></b>(const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1system_1_1error__condition.html">error_condition</a> & lhs,</span>
<span>&nbsp;&nbsp;const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1system_1_1error__condition.html">error_condition</a> & rhs);</span>
<br>
<span>bool </span><span><b><a href="{{ site.baseurl }}/api/groups/group__system__diagnostics.html#function-operator==">operator==</a></b>(const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1system_1_1error__code.html">error_code</a> & lhs,</span>
<span>&nbsp;&nbsp;const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1system_1_1error__code.html">error_code</a> & rhs);</span>
<br>
<span>bool </span><span><b><a href="{{ site.baseurl }}/api/groups/group__system__diagnostics.html#function-operator==">operator==</a></b>(const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1system_1_1error__code.html">error_code</a> & lhs,</span>
<span>&nbsp;&nbsp;const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1system_1_1error__condition.html">error_condition</a> & rhs);</span>
<br>
<span>bool </span><span><b><a href="{{ site.baseurl }}/api/groups/group__system__diagnostics.html#function-operator==">operator==</a></b>(const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1system_1_1error__condition.html">error_condition</a> & lhs,</span>
<span>&nbsp;&nbsp;const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1system_1_1error__code.html">error_code</a> & rhs);</span>
<br>
<span>bool </span><span><b><a href="{{ site.baseurl }}/api/groups/group__system__diagnostics.html#function-operator==">operator==</a></b>(const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1system_1_1error__condition.html">error_condition</a> & lhs,</span>
<span>&nbsp;&nbsp;const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1system_1_1error__condition.html">error_condition</a> & rhs);</span>
<br>
<span>bool </span><span><b><a href="{{ site.baseurl }}/api/groups/group__system__diagnostics.html#function-operator!=">operator!=</a></b>(const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1system_1_1error__code.html">error_code</a> & lhs,</span>
<span>&nbsp;&nbsp;const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1system_1_1error__code.html">error_code</a> & rhs);</span>
<br>
<span>bool </span><span><b><a href="{{ site.baseurl }}/api/groups/group__system__diagnostics.html#function-operator!=">operator!=</a></b>(const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1system_1_1error__code.html">error_code</a> & lhs,</span>
<span>&nbsp;&nbsp;const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1system_1_1error__condition.html">error_condition</a> & rhs);</span>
<br>
<span>bool </span><span><b><a href="{{ site.baseurl }}/api/groups/group__system__diagnostics.html#function-operator!=">operator!=</a></b>(const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1system_1_1error__condition.html">error_condition</a> & lhs,</span>
<span>&nbsp;&nbsp;const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1system_1_1error__code.html">error_code</a> & rhs);</span>
<br>
<span>bool </span><span><b><a href="{{ site.baseurl }}/api/groups/group__system__diagnostics.html#function-operator!=">operator!=</a></b>(const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1system_1_1error__condition.html">error_condition</a> & lhs,</span>
<span>&nbsp;&nbsp;const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1system_1_1error__condition.html">error_condition</a> & rhs);</span>
<span>} /* namespace thrust::system */</span>
</code>

## Member Classes

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

<h3 id="struct-thrustsystemis-error-code-enum">
<a href="{{ site.baseurl }}/api/classes/structthrust_1_1system_1_1is__error__code__enum.html">Struct <code>thrust::system::is&#95;error&#95;code&#95;enum</code>
</a>
</h3>

**Inherits From**:
`thrust::detail::false_type`

<h3 id="struct-thrustsystemis-error-code-enum<-cudaerrcerrc-t->">
<a href="{{ site.baseurl }}/api/classes/structthrust_1_1system_1_1is__error__code__enum_3_01cuda_1_1errc_1_1errc__t_01_4.html">Struct <code>thrust::system::is&#95;error&#95;code&#95;enum&lt; cuda::errc::errc&#95;t &gt;</code>
</a>
</h3>

**Inherits From**:
`thrust::detail::true_type`

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

<h3 id="class-thrustsystemsystem-error">
<a href="{{ site.baseurl }}/api/classes/classthrust_1_1system_1_1system__error.html">Class <code>thrust::system::system&#95;error</code>
</a>
</h3>

The class <code>system&#95;error</code> describes an exception object used to report error conditions that have an associated <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1system_1_1error__code.html">error&#95;code</a></code>. Such error conditions typically originate from the operating system or other low-level application program interfaces. 

**Inherits From**:
`std::runtime_error`


## Functions

<h3 id="function-cuda-category">
Function <code>thrust::system::cuda&#95;category</code>
</h3>

<code class="doxybook">
<span>const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1system_1_1error__category.html">error_category</a> & </span><span><b>cuda_category</b>(void);</span></code>

If the argument <code>ev</code> corresponds to a CUDA error value, the function shall return <code>error&#95;condition(ev,cuda&#95;category())</code>. Otherwise, the function shall return <code>system&#95;category.default&#95;error&#95;condition(ev)</code>. 

**Note**:
The object's <code>equivalent</code> virtual functions shall behave as specified for the class <code>thrust::error&#95;category</code>. The object's <code>name</code> virtual function shall return a pointer to the string <code>"cuda"</code>. The object's <code>default&#95;error&#95;condition</code> virtual function shall behave as follows:

**Returns**:
A reference to an object of a type derived from class <code>thrust::error&#95;category</code>. 

<h3 id="function-make-error-code">
Function <code>thrust::system::make&#95;error&#95;code</code>
</h3>

<code class="doxybook">
<span><a href="{{ site.baseurl }}/api/classes/classthrust_1_1system_1_1error__code.html">error_code</a> </span><span><b>make_error_code</b>(<a href="{{ site.baseurl }}/api/namespaces/namespacethrust_1_1system_1_1cuda_1_1errc.html#enum-errc-t">cuda::errc::errc_t</a> e);</span></code>
**Returns**:
* <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1system_1_1error__code.html">error&#95;code</a>(static&#95;cast&lt;int&gt;(e), cuda::error&#95;category())</code>
* <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1system_1_1error__code.html">error&#95;code</a>(static&#95;cast&lt;int&gt;(e), <a href="{{ site.baseurl }}/api/groups/group__system__diagnostics.html#function-generic-category">generic&#95;category()</a>)</code>

<h3 id="function-make-error-condition">
Function <code>thrust::system::make&#95;error&#95;condition</code>
</h3>

<code class="doxybook">
<span><a href="{{ site.baseurl }}/api/classes/classthrust_1_1system_1_1error__condition.html">error_condition</a> </span><span><b>make_error_condition</b>(<a href="{{ site.baseurl }}/api/namespaces/namespacethrust_1_1system_1_1cuda_1_1errc.html#enum-errc-t">cuda::errc::errc_t</a> e);</span></code>
**Returns**:
* <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1system_1_1error__condition.html">error&#95;condition</a>(static&#95;cast&lt;int&gt;(e), cuda::error&#95;category())</code>.
* <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1system_1_1error__condition.html">error&#95;condition</a>(static&#95;cast&lt;int&gt;(e), <a href="{{ site.baseurl }}/api/groups/group__system__diagnostics.html#function-generic-category">generic&#95;category()</a>)</code>. 

<h3 id="function-generic-category">
Function <code>thrust::system::generic&#95;category</code>
</h3>

<code class="doxybook">
<span>const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1system_1_1error__category.html">error_category</a> & </span><span><b>generic_category</b>(void);</span></code>
**Note**:
The object's <code>default&#95;error&#95;condition</code> and <code>equivalent</code> virtual functions shall behave as specified for the class <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1system_1_1error__category.html">error&#95;category</a></code>. The object's <code>name</code> virtual function shall return a pointer to the string <code>"generic"</code>. 

**Returns**:
A reference to an object of a type derived from class <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1system_1_1error__category.html">error&#95;category</a></code>. 

<h3 id="function-system-category">
Function <code>thrust::system::system&#95;category</code>
</h3>

<code class="doxybook">
<span>const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1system_1_1error__category.html">error_category</a> & </span><span><b>system_category</b>(void);</span></code>

If the argument <code>ev</code> corresponds to a POSIX <code>errno</code> value <code>posv</code>, the function shall return <code>error&#95;condition(ev,generic&#95;category())</code>. Otherwise, the function shall return <code>error&#95;condition(ev,system&#95;category())</code>. What constitutes correspondence for any given operating system is unspecified. 

**Note**:
The object's <code>equivalent</code> virtual functions shall behave as specified for class <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1system_1_1error__category.html">error&#95;category</a></code>. The object's <code>name</code> virtual function shall return a pointer to the string <code>"system"</code>. The object's <code>default&#95;error&#95;condition</code> virtual function shall behave as follows:

**Returns**:
A reference to an object of a type derived from class <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1system_1_1error__category.html">error&#95;category</a></code>. 

<h3 id="function-operator<">
Function <code>thrust::system::operator&lt;</code>
</h3>

<code class="doxybook">
<span>bool </span><span><b>operator<</b>(const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1system_1_1error__code.html">error_code</a> & lhs,</span>
<span>&nbsp;&nbsp;const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1system_1_1error__code.html">error_code</a> & rhs);</span></code>
**Returns**:
<code>lhs.category() &lt; rhs.category() || lhs.category() == rhs.category() && lhs.value() &lt; rhs.value()</code>. 

<h3 id="function-operator<<">
Function <code>thrust::system::operator&lt;&lt;</code>
</h3>

<code class="doxybook">
<span>template &lt;typename charT,</span>
<span>&nbsp;&nbsp;typename traits&gt;</span>
<span>std::basic_ostream< charT, traits > & </span><span><b>operator<<</b>(std::basic_ostream< charT, traits > & os,</span>
<span>&nbsp;&nbsp;const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1system_1_1error__code.html">error_code</a> & ec);</span></code>
Effects: <code>os &lt;&lt; ec.category().name() &lt;&lt; ':' &lt;&lt; ec.value()</code>. 

<h3 id="function-operator<">
Function <code>thrust::system::operator&lt;</code>
</h3>

<code class="doxybook">
<span>bool </span><span><b>operator<</b>(const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1system_1_1error__condition.html">error_condition</a> & lhs,</span>
<span>&nbsp;&nbsp;const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1system_1_1error__condition.html">error_condition</a> & rhs);</span></code>
**Returns**:
<code>lhs.category() &lt; rhs.category() || lhs.category() == rhs.category() && lhs.value() &lt; rhs.value()</code>. 

<h3 id="function-operator==">
Function <code>thrust::system::operator==</code>
</h3>

<code class="doxybook">
<span>bool </span><span><b>operator==</b>(const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1system_1_1error__code.html">error_code</a> & lhs,</span>
<span>&nbsp;&nbsp;const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1system_1_1error__code.html">error_code</a> & rhs);</span></code>
**Returns**:
<code>lhs.category() == rhs.category() && lhs.value() == rhs.value()</code>. 

<h3 id="function-operator==">
Function <code>thrust::system::operator==</code>
</h3>

<code class="doxybook">
<span>bool </span><span><b>operator==</b>(const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1system_1_1error__code.html">error_code</a> & lhs,</span>
<span>&nbsp;&nbsp;const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1system_1_1error__condition.html">error_condition</a> & rhs);</span></code>
**Returns**:
<code>lhs.category().equivalent(lhs.value(), rhs) || rhs.category().equivalent(lhs,rhs.value())</code>. 

<h3 id="function-operator==">
Function <code>thrust::system::operator==</code>
</h3>

<code class="doxybook">
<span>bool </span><span><b>operator==</b>(const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1system_1_1error__condition.html">error_condition</a> & lhs,</span>
<span>&nbsp;&nbsp;const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1system_1_1error__code.html">error_code</a> & rhs);</span></code>
**Returns**:
<code>rhs.category().equivalent(lhs.value(), lhs) || lhs.category().equivalent(rhs, lhs.value())</code>. 

<h3 id="function-operator==">
Function <code>thrust::system::operator==</code>
</h3>

<code class="doxybook">
<span>bool </span><span><b>operator==</b>(const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1system_1_1error__condition.html">error_condition</a> & lhs,</span>
<span>&nbsp;&nbsp;const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1system_1_1error__condition.html">error_condition</a> & rhs);</span></code>
**Returns**:
<code>lhs.category() == rhs.category() && lhs.value() == rhs.value()</code>

<h3 id="function-operator!=">
Function <code>thrust::system::operator!=</code>
</h3>

<code class="doxybook">
<span>bool </span><span><b>operator!=</b>(const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1system_1_1error__code.html">error_code</a> & lhs,</span>
<span>&nbsp;&nbsp;const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1system_1_1error__code.html">error_code</a> & rhs);</span></code>
**Returns**:
<code>!(lhs == rhs)</code>

<h3 id="function-operator!=">
Function <code>thrust::system::operator!=</code>
</h3>

<code class="doxybook">
<span>bool </span><span><b>operator!=</b>(const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1system_1_1error__code.html">error_code</a> & lhs,</span>
<span>&nbsp;&nbsp;const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1system_1_1error__condition.html">error_condition</a> & rhs);</span></code>
**Returns**:
<code>!(lhs == rhs)</code>

<h3 id="function-operator!=">
Function <code>thrust::system::operator!=</code>
</h3>

<code class="doxybook">
<span>bool </span><span><b>operator!=</b>(const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1system_1_1error__condition.html">error_condition</a> & lhs,</span>
<span>&nbsp;&nbsp;const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1system_1_1error__code.html">error_code</a> & rhs);</span></code>
**Returns**:
<code>!(lhs == rhs)</code>

<h3 id="function-operator!=">
Function <code>thrust::system::operator!=</code>
</h3>

<code class="doxybook">
<span>bool </span><span><b>operator!=</b>(const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1system_1_1error__condition.html">error_condition</a> & lhs,</span>
<span>&nbsp;&nbsp;const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1system_1_1error__condition.html">error_condition</a> & rhs);</span></code>
**Returns**:
<code>!(lhs == rhs)</code>


