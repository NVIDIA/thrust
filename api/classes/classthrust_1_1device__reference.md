---
title: thrust::device_reference
parent: Memory Management
nav_exclude: true
has_children: true
has_toc: false
---

# Class `thrust::device_reference`

<code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1device__reference.html">device&#95;reference</a></code> acts as a reference-like object to an object stored in device memory. <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1device__reference.html">device&#95;reference</a></code> is not intended to be used directly; rather, this type is the result of deferencing a <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1device__ptr.html">device&#95;ptr</a></code>. Similarly, taking the address of a <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1device__reference.html">device&#95;reference</a></code> yields a <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1device__ptr.html">device&#95;ptr</a></code>.

<code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1device__reference.html">device&#95;reference</a></code> may often be used from host code in place of operations defined on its associated <code>value&#95;type</code>. For example, when <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1device__reference.html">device&#95;reference</a></code> refers to an arithmetic type, arithmetic operations on it are legal:



```cpp
#include <thrust/device_vector.h>

int main(void)
{
  thrust::device_vector<int> vec(1, 13);

  thrust::device_reference<int> ref_to_thirteen = vec[0];

  int x = ref_to_thirteen + 1;

  // x is 14

  return 0;
}
```

Similarly, we can print the value of <code>ref&#95;to&#95;thirteen</code> in the above code by using an <code>iostream:</code>



```cpp
#include <thrust/device_vector.h>
#include <iostream>

int main(void)
{
  thrust::device_vector<int> vec(1, 13);

  thrust::device_reference<int> ref_to_thirteen = vec[0];

  std::cout << ref_to_thirteen << std::endl;

  // 13 is printed

  return 0;
}
```

Of course, we needn't explicitly create a <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1device__reference.html">device&#95;reference</a></code> in the previous example, because one is returned by <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1device__vector.html">device&#95;vector</a>'s</code> bracket operator. A more natural way to print the value of a <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1device__vector.html">device&#95;vector</a></code> element might be:



```cpp
#include <thrust/device_vector.h>
#include <iostream>

int main(void)
{
  thrust::device_vector<int> vec(1, 13);

  std::cout << vec[0] << std::endl;

  // 13 is printed

  return 0;
}
```

These kinds of operations should be used sparingly in performance-critical code, because they imply a potentially expensive copy between host and device space.

Some operations which are possible with regular objects are impossible with their corresponding <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1device__reference.html">device&#95;reference</a></code> objects due to the requirements of the C++ language. For example, because the member access operator cannot be overloaded, member variables and functions of a referent object cannot be directly accessed through its <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1device__reference.html">device&#95;reference</a></code>.

The following code, which generates a compiler error, illustrates:



```cpp
#include <thrust/device_vector.h>

struct foo
{
  int x;
};

int main(void)
{
  thrust::device_vector<foo> foo_vec(1);

  thrust::device_reference<foo> foo_ref = foo_vec[0];

  foo_ref.x = 13; // ERROR: x cannot be accessed through foo_ref

  return 0;
}
```

Instead, a host space copy must be created to access <code>foo's</code><code>x</code> member:



```cpp
#include <thrust/device_vector.h>

struct foo
{
  int x;
};

int main(void)
{
  thrust::device_vector<foo> foo_vec(1);

  // create a local host-side foo object
  foo host_foo;
  host_foo.x = 13;

  thrust::device_reference<foo> foo_ref = foo_vec[0];

  foo_ref = host_foo;

  // foo_ref's x member is 13

  return 0;
}
```

Another common case where a <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1device__reference.html">device&#95;reference</a></code> cannot directly be used in place of its referent object occurs when passing them as parameters to functions like <code>printf</code> which have varargs parameters. Because varargs parameters must be Plain Old Data, a <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1device__reference.html">device&#95;reference</a></code> to a POD type requires a cast when passed to <code>printf:</code>



```cpp
#include <stdio.h>
#include <thrust/device_vector.h>

int main(void)
{
  thrust::device_vector<int> vec(1,13);

  // vec[0] must be cast to int when passing to printf
  printf("%d\n", (int) vec[0]);

  return 0;
}
```

**Inherits From**:
`thrust::reference< T, thrust::device_ptr< T >, thrust::device_reference< T > >`

**See**:
* <a href="{{ site.baseurl }}/api/classes/classthrust_1_1device__ptr.html">device_ptr</a>
* <a href="{{ site.baseurl }}/api/classes/classthrust_1_1device__vector.html">device_vector</a>

<code class="doxybook">
<span>#include <thrust/device_reference.h></span><br>
<span>template &lt;typename T&gt;</span>
<span>class thrust::device&#95;reference {</span>
<span>public:</span><span>&nbsp;&nbsp;typedef <i>see below</i> <b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1device__reference.html#typedef-value-type">value&#95;type</a></b>;</span>
<br>
<span>&nbsp;&nbsp;typedef <i>see below</i> <b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1device__reference.html#typedef-pointer">pointer</a></b>;</span>
<br>
<span>&nbsp;&nbsp;template &lt;typename OtherT&gt;</span>
<span>&nbsp;&nbsp;__host__ __device__ </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1device__reference.html#function-device-reference">device&#95;reference</a></b>(const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1device__reference.html">device_reference</a>< OtherT > & other,</span>
<span>&nbsp;&nbsp;&nbsp;&nbsp;typename thrust::detail::enable_if_convertible< typename <a href="{{ site.baseurl }}/api/classes/classthrust_1_1device__reference.html">device_reference</a>< OtherT ><a href="{{ site.baseurl }}/api/classes/classthrust_1_1device__reference.html#typedef-pointer">::pointer</a>, <a href="{{ site.baseurl }}/api/classes/classthrust_1_1device__reference.html#typedef-pointer">pointer</a> >::type * = 0);</span>
<br>
<span>&nbsp;&nbsp;explicit __host__ __device__ </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1device__reference.html#function-device-reference">device&#95;reference</a></b>(const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1device__reference.html#typedef-pointer">pointer</a> & ptr);</span>
<br>
<span>&nbsp;&nbsp;template &lt;typename OtherT&gt;</span>
<span>&nbsp;&nbsp;__host__ __device__ <a href="{{ site.baseurl }}/api/classes/classthrust_1_1device__reference.html">device_reference</a> & </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1device__reference.html#function-operator=">operator=</a></b>(const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1device__reference.html">device_reference</a>< OtherT > & other);</span>
<br>
<span>&nbsp;&nbsp;__host__ __device__ <a href="{{ site.baseurl }}/api/classes/classthrust_1_1device__reference.html">device_reference</a> & </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1device__reference.html#function-operator=">operator=</a></b>(const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1device__reference.html#typedef-value-type">value_type</a> & x);</span>
<span>};</span>
</code>

## Member Types

<h3 id="typedef-value-type">
Typedef <code>thrust::device&#95;reference::value&#95;type</code>
</h3>

<code class="doxybook">
<span>typedef super_t::value_type<b>value_type</b>;</span></code>
The type of the value referenced by this type of <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1device__reference.html">device&#95;reference</a></code>. 

<h3 id="typedef-pointer">
Typedef <code>thrust::device&#95;reference::pointer</code>
</h3>

<code class="doxybook">
<span>typedef super_t::pointer<b>pointer</b>;</span></code>
The type of the expression <code>&ref</code>, where <code>ref</code> is a <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1device__reference.html">device&#95;reference</a></code>. 


## Member Functions

<h3 id="function-device-reference">
Function <code>thrust::device&#95;reference::device&#95;reference</code>
</h3>

<code class="doxybook">
<span>template &lt;typename OtherT&gt;</span>
<span>__host__ __device__ </span><span><b>device_reference</b>(const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1device__reference.html">device_reference</a>< OtherT > & other,</span>
<span>&nbsp;&nbsp;typename thrust::detail::enable_if_convertible< typename <a href="{{ site.baseurl }}/api/classes/classthrust_1_1device__reference.html">device_reference</a>< OtherT ><a href="{{ site.baseurl }}/api/classes/classthrust_1_1device__reference.html#typedef-pointer">::pointer</a>, <a href="{{ site.baseurl }}/api/classes/classthrust_1_1device__reference.html#typedef-pointer">pointer</a> >::type * = 0);</span></code>
This copy constructor accepts a const reference to another <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1device__reference.html">device&#95;reference</a></code>. After this <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1device__reference.html">device&#95;reference</a></code> is constructed, it shall refer to the same object as <code>other</code>.


The following code snippet demonstrates the semantics of this copy constructor.



```cpp
#include <thrust/device_vector.h>
#include <assert.h>
...
thrust::device_vector<int> v(1,0);
thrust::device_reference<int> ref = v[0];

// ref equals the object at v[0]
assert(ref == v[0]);

// the address of ref equals the address of v[0]
assert(&ref == &v[0]);

// modifying v[0] modifies ref
v[0] = 13;
assert(ref == 13);
```

**Note**:
This constructor is templated primarily to allow initialization of <code>device&#95;reference&lt;const T&gt;</code> from <code>device&#95;reference&lt;T&gt;</code>. 

**Function Parameters**:
**`other`**: A <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1device__reference.html">device&#95;reference</a></code> to copy from.

<h3 id="function-device-reference">
Function <code>thrust::device&#95;reference::device&#95;reference</code>
</h3>

<code class="doxybook">
<span>explicit __host__ __device__ </span><span><b>device_reference</b>(const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1device__reference.html#typedef-pointer">pointer</a> & ptr);</span></code>
This copy constructor initializes this <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1device__reference.html">device&#95;reference</a></code> to refer to an object pointed to by the given <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1device__ptr.html">device&#95;ptr</a></code>. After this <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1device__reference.html">device&#95;reference</a></code> is constructed, it shall refer to the object pointed to by <code>ptr</code>.


The following code snippet demonstrates the semantic of this copy constructor.



```cpp
#include <thrust/device_vector.h>
#include <assert.h>
...
thrust::device_vector<int> v(1,0);
thrust::device_ptr<int> ptr = &v[0];
thrust::device_reference<int> ref(ptr);

// ref equals the object pointed to by ptr
assert(ref == *ptr);

// the address of ref equals ptr
assert(&ref == ptr);

// modifying *ptr modifies ref
*ptr = 13;
assert(ref == 13);
```

**Function Parameters**:
**`ptr`**: A <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1device__ptr.html">device&#95;ptr</a></code> to copy from.

<h3 id="function-operator=">
Function <code>thrust::device&#95;reference::operator=</code>
</h3>

<code class="doxybook">
<span>template &lt;typename OtherT&gt;</span>
<span>__host__ __device__ <a href="{{ site.baseurl }}/api/classes/classthrust_1_1device__reference.html">device_reference</a> & </span><span><b>operator=</b>(const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1device__reference.html">device_reference</a>< OtherT > & other);</span></code>
This assignment operator assigns the value of the object referenced by the given <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1device__reference.html">device&#95;reference</a></code> to the object referenced by this <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1device__reference.html">device&#95;reference</a></code>.

**Function Parameters**:
**`other`**: The <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1device__reference.html">device&#95;reference</a></code> to assign from. 

**Returns**:
<code>&#42;this</code>

<h3 id="function-operator=">
Function <code>thrust::device&#95;reference::operator=</code>
</h3>

<code class="doxybook">
<span>__host__ __device__ <a href="{{ site.baseurl }}/api/classes/classthrust_1_1device__reference.html">device_reference</a> & </span><span><b>operator=</b>(const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1device__reference.html#typedef-value-type">value_type</a> & x);</span></code>
Assignment operator assigns the value of the given value to the value referenced by this <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1device__reference.html">device&#95;reference</a></code>.

**Function Parameters**:
**`x`**: The value to assign from. 

**Returns**:
<code>&#42;this</code>


