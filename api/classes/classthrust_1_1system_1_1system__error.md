---
title: thrust::system::system_error
summary: The class system_error describes an exception object used to report error conditions that have an associated error_code. Such error conditions typically originate from the operating system or other low-level application program interfaces. 
parent: System Diagnostics
grand_parent: System
nav_exclude: true
has_children: true
has_toc: false
---

# Class `thrust::system::system_error`

The class <code>system&#95;error</code> describes an exception object used to report error conditions that have an associated <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1system_1_1error__code.html">error&#95;code</a></code>. Such error conditions typically originate from the operating system or other low-level application program interfaces. 

Thrust uses <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1system_1_1system__error.html">system&#95;error</a></code> to report the error codes returned from device backends such as the CUDA runtime.

The following code listing demonstrates how to catch a <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1system_1_1system__error.html">system&#95;error</a></code> to recover from an error.



```cpp
#include <thrust/device_vector.h>
#include <thrust/system.h>
#include <thrust/sort.h>

void terminate_gracefully(void)
{
  // application-specific termination code here
  ...
}

int main(void)
{
  try
  {
    thrust::device_vector<float> vec;
    thrust::sort(vec.begin(), vec.end());
  }
  catch(thrust::system_error e)
  {
    std::cerr << "Error inside sort: " << e.what() << std::endl;
    terminate_gracefully();
  }

  return 0;
}
```

**Note**:
If an error represents an out-of-memory condition, implementations are encouraged to throw an exception object of type <code>std::bad&#95;alloc</code> rather than <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1system_1_1system__error.html">system&#95;error</a></code>. 

**Inherits From**:
`std::runtime_error`

<code class="doxybook">
<span>#include <thrust/system/system_error.h></span><br>
<span>class thrust::system::system&#95;error {</span>
<span>public:</span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/groups/group__system__diagnostics.html#function-system-error">system&#95;error</a></b>(<a href="{{ site.baseurl }}/api/classes/classthrust_1_1system_1_1error__code.html">error_code</a> ec,</span>
<span>&nbsp;&nbsp;&nbsp;&nbsp;const std::string & what_arg);</span>
<br>
<span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/groups/group__system__diagnostics.html#function-system-error">system&#95;error</a></b>(<a href="{{ site.baseurl }}/api/classes/classthrust_1_1system_1_1error__code.html">error_code</a> ec,</span>
<span>&nbsp;&nbsp;&nbsp;&nbsp;const char * what_arg);</span>
<br>
<span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/groups/group__system__diagnostics.html#function-system-error">system&#95;error</a></b>(<a href="{{ site.baseurl }}/api/classes/classthrust_1_1system_1_1error__code.html">error_code</a> ec);</span>
<br>
<span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/groups/group__system__diagnostics.html#function-system-error">system&#95;error</a></b>(int ev,</span>
<span>&nbsp;&nbsp;&nbsp;&nbsp;const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1system_1_1error__category.html">error_category</a> & ecat,</span>
<span>&nbsp;&nbsp;&nbsp;&nbsp;const std::string & what_arg);</span>
<br>
<span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/groups/group__system__diagnostics.html#function-system-error">system&#95;error</a></b>(int ev,</span>
<span>&nbsp;&nbsp;&nbsp;&nbsp;const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1system_1_1error__category.html">error_category</a> & ecat,</span>
<span>&nbsp;&nbsp;&nbsp;&nbsp;const char * what_arg);</span>
<br>
<span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/groups/group__system__diagnostics.html#function-system-error">system&#95;error</a></b>(int ev,</span>
<span>&nbsp;&nbsp;&nbsp;&nbsp;const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1system_1_1error__category.html">error_category</a> & ecat);</span>
<br>
<span>&nbsp;&nbsp;virtual </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/groups/group__system__diagnostics.html#function-~system-error">~system&#95;error</a></b>(void);</span>
<br>
<span>&nbsp;&nbsp;const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1system_1_1error__code.html">error_code</a> & </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/groups/group__system__diagnostics.html#function-code">code</a></b>(void) const;</span>
<br>
<span>&nbsp;&nbsp;const char * </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/groups/group__system__diagnostics.html#function-what">what</a></b>(void) const;</span>
<span>};</span>
</code>

## Member Functions

<h3 id="function-system-error">
Function <code>thrust::system::system&#95;error::system&#95;error</code>
</h3>

<code class="doxybook">
<span><b>system_error</b>(<a href="{{ site.baseurl }}/api/classes/classthrust_1_1system_1_1error__code.html">error_code</a> ec,</span>
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
<span><b>system_error</b>(<a href="{{ site.baseurl }}/api/classes/classthrust_1_1system_1_1error__code.html">error_code</a> ec,</span>
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
<span><b>system_error</b>(<a href="{{ site.baseurl }}/api/classes/classthrust_1_1system_1_1error__code.html">error_code</a> ec);</span></code>
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
<span>&nbsp;&nbsp;const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1system_1_1error__category.html">error_category</a> & ecat,</span>
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
<span>&nbsp;&nbsp;const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1system_1_1error__category.html">error_category</a> & ecat,</span>
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
<span>&nbsp;&nbsp;const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1system_1_1error__category.html">error_category</a> & ecat);</span></code>
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
<span>const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1system_1_1error__code.html">error_code</a> & </span><span><b>code</b>(void) const;</span></code>
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


