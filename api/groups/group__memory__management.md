---
title: Memory Management
parent: API
nav_exclude: false
has_children: true
has_toc: false
---

# Memory Management

All Thrust functionalities related to memory allocation and deallocation. 

## Groups

* **[Allocators]({{ site.baseurl }}/api/groups/group__allocators.html)**
* **[Memory Resources]({{ site.baseurl }}/api/groups/group__memory__resources.html)**

<code class="doxybook">
<span>template &lt;typename T&gt;</span>
<span>class <b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1device__reference.html">thrust::device&#95;reference</a></b>;</span>
<br>
<span class="doxybook-comment">/* <code>device&#95;ptr</code> is a pointer-like object which points to an object that resides in memory associated with the device system.  */</span><span>template &lt;typename T&gt;</span>
<span>class <b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1device__ptr.html">thrust::device&#95;ptr</a></b>;</span>
<br>
<span>template &lt;typename T&gt;</span>
<span>void </span><span><b><a href="{{ site.baseurl }}/api/groups/group__memory__management.html#function-device-delete">thrust::device&#95;delete</a></b>(<a href="{{ site.baseurl }}/api/classes/classthrust_1_1device__ptr.html">thrust::device_ptr</a>< T > ptr,</span>
<span>&nbsp;&nbsp;const size_t n = 1);</span>
<br>
<span>void </span><span><b><a href="{{ site.baseurl }}/api/groups/group__memory__management.html#function-device-free">thrust::device&#95;free</a></b>(<a href="{{ site.baseurl }}/api/classes/classthrust_1_1device__ptr.html">thrust::device_ptr</a>< void > ptr);</span>
<br>
<span><a href="{{ site.baseurl }}/api/classes/classthrust_1_1device__ptr.html">thrust::device_ptr</a>< void > </span><span><b><a href="{{ site.baseurl }}/api/groups/group__memory__management.html#function-device-malloc">thrust::device&#95;malloc</a></b>(const std::size_t n);</span>
<br>
<span>template &lt;typename T&gt;</span>
<span><a href="{{ site.baseurl }}/api/classes/classthrust_1_1device__ptr.html">thrust::device_ptr</a>< T > </span><span><b><a href="{{ site.baseurl }}/api/groups/group__memory__management.html#function-device-malloc">thrust::device&#95;malloc</a></b>(const std::size_t n);</span>
<br>
<span>template &lt;typename T&gt;</span>
<span>device_ptr< T > </span><span><b><a href="{{ site.baseurl }}/api/groups/group__memory__management.html#function-device-new">thrust::device&#95;new</a></b>(device_ptr< void > p,</span>
<span>&nbsp;&nbsp;const size_t n = 1);</span>
<br>
<span>template &lt;typename T&gt;</span>
<span>device_ptr< T > </span><span><b><a href="{{ site.baseurl }}/api/groups/group__memory__management.html#function-device-new">thrust::device&#95;new</a></b>(device_ptr< void > p,</span>
<span>&nbsp;&nbsp;const T & exemplar,</span>
<span>&nbsp;&nbsp;const size_t n = 1);</span>
<br>
<span>template &lt;typename T&gt;</span>
<span>device_ptr< T > </span><span><b><a href="{{ site.baseurl }}/api/groups/group__memory__management.html#function-device-new">thrust::device&#95;new</a></b>(const size_t n = 1);</span>
<br>
<span>template &lt;typename T,</span>
<span>&nbsp;&nbsp;typename CharT,</span>
<span>&nbsp;&nbsp;typename Traits&gt;</span>
<span>__host__ std::basic_ostream< CharT, Traits > & </span><span><b><a href="{{ site.baseurl }}/api/groups/group__memory__management.html#function-operator<<">thrust::operator&lt;&lt;</a></b>(std::basic_ostream< CharT, Traits > & os,</span>
<span>&nbsp;&nbsp;device_ptr< T > const & dp);</span>
<br>
<span class="doxybook-comment">/* Create a <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1device__ptr.html">device&#95;ptr</a></code> from a raw pointer.  */</span><span>template &lt;typename T&gt;</span>
<span>__host__ __device__ device_ptr< T > </span><span><b><a href="{{ site.baseurl }}/api/groups/group__memory__management.html#function-device-pointer-cast">thrust::device&#95;pointer&#95;cast</a></b>(T * ptr);</span>
<br>
<span class="doxybook-comment">/* Create a <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1device__ptr.html">device&#95;ptr</a></code> from another <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1device__ptr.html">device&#95;ptr</a></code>.  */</span><span>template &lt;typename T&gt;</span>
<span>__host__ __device__ device_ptr< T > </span><span><b><a href="{{ site.baseurl }}/api/groups/group__memory__management.html#function-device-pointer-cast">thrust::device&#95;pointer&#95;cast</a></b>(device_ptr< T > const & dptr);</span>
<br>
<span>template &lt;typename T&gt;</span>
<span>__host__ __device__ void </span><span><b><a href="{{ site.baseurl }}/api/groups/group__memory__management.html#function-swap">thrust::swap</a></b>(device_reference< T > & x,</span>
<span>&nbsp;&nbsp;device_reference< T > & y);</span>
<br>
<span>template &lt;typename T,</span>
<span>&nbsp;&nbsp;typename charT,</span>
<span>&nbsp;&nbsp;typename traits&gt;</span>
<span>std::basic_ostream< charT, traits > & </span><span><b><a href="{{ site.baseurl }}/api/groups/group__memory__management.html#function-operator<<">thrust::operator&lt;&lt;</a></b>(std::basic_ostream< charT, traits > & os,</span>
<span>&nbsp;&nbsp;const device_reference< T > & y);</span>
<br>
<span>template &lt;typename DerivedPolicy&gt;</span>
<span>__host__ __device__ pointer< void, DerivedPolicy > </span><span><b><a href="{{ site.baseurl }}/api/groups/group__memory__management.html#function-malloc">thrust::malloc</a></b>(const thrust::detail::execution_policy_base< DerivedPolicy > & system,</span>
<span>&nbsp;&nbsp;std::size_t n);</span>
<br>
<span>template &lt;typename T,</span>
<span>&nbsp;&nbsp;typename DerivedPolicy&gt;</span>
<span>__host__ __device__ pointer< T, DerivedPolicy > </span><span><b><a href="{{ site.baseurl }}/api/groups/group__memory__management.html#function-malloc">thrust::malloc</a></b>(const thrust::detail::execution_policy_base< DerivedPolicy > & system,</span>
<span>&nbsp;&nbsp;std::size_t n);</span>
<br>
<span>template &lt;typename T,</span>
<span>&nbsp;&nbsp;typename DerivedPolicy&gt;</span>
<span>__host__ __device__ <a href="{{ site.baseurl }}/api/classes/structthrust_1_1pair.html">thrust::pair</a>< thrust::pointer< T, DerivedPolicy >, typename thrust::pointer< T, DerivedPolicy >::difference_type > </span><span><b><a href="{{ site.baseurl }}/api/groups/group__memory__management.html#function-get-temporary-buffer">thrust::get&#95;temporary&#95;buffer</a></b>(const thrust::detail::execution_policy_base< DerivedPolicy > & system,</span>
<span>&nbsp;&nbsp;typename thrust::pointer< T, DerivedPolicy >::difference_type n);</span>
<br>
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename Pointer&gt;</span>
<span>__host__ __device__ void </span><span><b><a href="{{ site.baseurl }}/api/groups/group__memory__management.html#function-free">thrust::free</a></b>(const thrust::detail::execution_policy_base< DerivedPolicy > & system,</span>
<span>&nbsp;&nbsp;Pointer ptr);</span>
<br>
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename Pointer&gt;</span>
<span>__host__ __device__ void </span><span><b><a href="{{ site.baseurl }}/api/groups/group__memory__management.html#function-return-temporary-buffer">thrust::return&#95;temporary&#95;buffer</a></b>(const thrust::detail::execution_policy_base< DerivedPolicy > & system,</span>
<span>&nbsp;&nbsp;Pointer p,</span>
<span>&nbsp;&nbsp;std::ptrdiff_t n);</span>
<br>
<span>template &lt;typename Pointer&gt;</span>
<span>__host__ __device__ thrust::detail::pointer_traits< Pointer >::raw_pointer </span><span><b><a href="{{ site.baseurl }}/api/groups/group__memory__management.html#function-raw-pointer-cast">thrust::raw&#95;pointer&#95;cast</a></b>(Pointer ptr);</span>
<br>
<span>template &lt;typename T&gt;</span>
<span>__host__ __device__ detail::raw_reference< T >::type </span><span><b><a href="{{ site.baseurl }}/api/groups/group__memory__management.html#function-raw-reference-cast">thrust::raw&#95;reference&#95;cast</a></b>(T & ref);</span>
<br>
<span>template &lt;typename T&gt;</span>
<span>__host__ __device__ detail::raw_reference< constT >::type </span><span><b><a href="{{ site.baseurl }}/api/groups/group__memory__management.html#function-raw-reference-cast">thrust::raw&#95;reference&#95;cast</a></b>(const T & ref);</span>
</code>

## Member Classes

<h3 id="class-thrustdevice-reference">
<a href="{{ site.baseurl }}/api/classes/classthrust_1_1device__reference.html">Class <code>thrust::device&#95;reference</code>
</a>
</h3>

**Inherits From**:
`thrust::reference< T, thrust::device_ptr< T >, thrust::device_reference< T > >`

<h3 id="class-thrustdevice-ptr">
<a href="{{ site.baseurl }}/api/classes/classthrust_1_1device__ptr.html">Class <code>thrust::device&#95;ptr</code>
</a>
</h3>

<code>device&#95;ptr</code> is a pointer-like object which points to an object that resides in memory associated with the device system. 

**Inherits From**:
`thrust::pointer< T, thrust::device_system_tag, thrust::device_reference< T >, thrust::device_ptr< T > >`


## Functions

<h3 id="function-device-delete">
Function <code>thrust::device&#95;delete</code>
</h3>

<code class="doxybook">
<span>template &lt;typename T&gt;</span>
<span>void </span><span><b>device_delete</b>(<a href="{{ site.baseurl }}/api/classes/classthrust_1_1device__ptr.html">thrust::device_ptr</a>< T > ptr,</span>
<span>&nbsp;&nbsp;const size_t n = 1);</span></code>
<code>device&#95;delete</code> deletes a <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1device__ptr.html">device&#95;ptr</a></code> allocated with <code>device&#95;new</code>.

**Function Parameters**:
* **`ptr`** The <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1device__ptr.html">device&#95;ptr</a></code> to delete, assumed to have been allocated with <code>device&#95;new</code>. 
* **`n`** The number of objects to destroy at <code>ptr</code>. Defaults to <code>1</code> similar to <code>device&#95;new</code>.

**See**:
* <a href="{{ site.baseurl }}/api/classes/classthrust_1_1device__ptr.html">device_ptr</a>
* device_new 

<h3 id="function-device-free">
Function <code>thrust::device&#95;free</code>
</h3>

<code class="doxybook">
<span>void </span><span><b>device_free</b>(<a href="{{ site.baseurl }}/api/classes/classthrust_1_1device__ptr.html">thrust::device_ptr</a>< void > ptr);</span></code>
<code>device&#95;free</code> deallocates memory allocated by the function <code>device&#95;malloc</code>.


The following code snippet demonstrates how to use <code>device&#95;free</code> to deallocate memory allocated by <code>device&#95;malloc</code>.



```cpp
#include <thrust/device_malloc.h>
#include <thrust/device_free.h>
...
// allocate some integers with device_malloc
const int N = 100;
thrust::device_ptr<int> int_array = thrust::device_malloc<int>(N);

// manipulate integers
...

// deallocate with device_free
thrust::device_free(int_array);
```

**Function Parameters**:
**`ptr`**: A <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1device__ptr.html">device&#95;ptr</a></code> pointing to memory to be deallocated.

**See**:
* <a href="{{ site.baseurl }}/api/classes/classthrust_1_1device__ptr.html">device_ptr</a>
* device_malloc 

<h3 id="function-device-malloc">
Function <code>thrust::device&#95;malloc</code>
</h3>

<code class="doxybook">
<span><a href="{{ site.baseurl }}/api/classes/classthrust_1_1device__ptr.html">thrust::device_ptr</a>< void > </span><span><b>device_malloc</b>(const std::size_t n);</span></code>
This version of <code>device&#95;malloc</code> allocates sequential device storage for bytes.


The following code snippet demonstrates how to use <code>device&#95;malloc</code> to allocate a range of device memory.



```cpp
#include <thrust/device_malloc.h>
#include <thrust/device_free.h>
...
// allocate some memory with device_malloc
const int N = 100;
thrust::device_ptr<void> void_ptr = thrust::device_malloc(N);

// manipulate memory
...

// deallocate with device_free
thrust::device_free(void_ptr);
```

**Function Parameters**:
**`n`**: The number of bytes to allocate sequentially in device memory. 

**Returns**:
A <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1device__ptr.html">device&#95;ptr</a></code> to the newly allocated memory.

**See**:
* <a href="{{ site.baseurl }}/api/classes/classthrust_1_1device__ptr.html">device_ptr</a>
* device_free 

<h3 id="function-device-malloc">
Function <code>thrust::device&#95;malloc</code>
</h3>

<code class="doxybook">
<span>template &lt;typename T&gt;</span>
<span><a href="{{ site.baseurl }}/api/classes/classthrust_1_1device__ptr.html">thrust::device_ptr</a>< T > </span><span><b>device_malloc</b>(const std::size_t n);</span></code>
This version of <code>device&#95;malloc</code> allocates sequential device storage for new objects of the given type.


The following code snippet demonstrates how to use <code>device&#95;malloc</code> to allocate a range of device memory.



```cpp
#include <thrust/device_malloc.h>
#include <thrust/device_free.h>
...
// allocate some integers with device_malloc
const int N = 100;
thrust::device_ptr<int> int_array = thrust::device_malloc<int>(N);

// manipulate integers
...

// deallocate with device_free
thrust::device_free(int_array);
```

**Function Parameters**:
**`n`**: The number of objects of type T to allocate sequentially in device memory. 

**Returns**:
A <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1device__ptr.html">device&#95;ptr</a></code> to the newly allocated memory.

**See**:
* <a href="{{ site.baseurl }}/api/classes/classthrust_1_1device__ptr.html">device_ptr</a>
* device_free 

<h3 id="function-device-new">
Function <code>thrust::device&#95;new</code>
</h3>

<code class="doxybook">
<span>template &lt;typename T&gt;</span>
<span>device_ptr< T > </span><span><b>device_new</b>(device_ptr< void > p,</span>
<span>&nbsp;&nbsp;const size_t n = 1);</span></code>
<code>device&#95;new</code> implements the placement <code>new</code> operator for types resident in device memory. <code>device&#95;new</code> calls <code>T</code>'s null constructor on a array of objects in device memory. No memory is allocated by this function.

**Function Parameters**:
* **`p`** A <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1device__ptr.html">device&#95;ptr</a></code> to a region of device memory into which to construct one or many <code>T</code>s. 
* **`n`** The number of objects to construct at <code>p</code>. 

**Returns**:
p, casted to <code>T</code>'s type.

**See**:
<a href="{{ site.baseurl }}/api/classes/classthrust_1_1device__ptr.html">device_ptr</a>

<h3 id="function-device-new">
Function <code>thrust::device&#95;new</code>
</h3>

<code class="doxybook">
<span>template &lt;typename T&gt;</span>
<span>device_ptr< T > </span><span><b>device_new</b>(device_ptr< void > p,</span>
<span>&nbsp;&nbsp;const T & exemplar,</span>
<span>&nbsp;&nbsp;const size_t n = 1);</span></code>
<code>device&#95;new</code> implements the placement new operator for types resident in device memory. <code>device&#95;new</code> calls <code>T</code>'s copy constructor on a array of objects in device memory. No memory is allocated by this function.

**Function Parameters**:
* **`p`** A <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1device__ptr.html">device&#95;ptr</a></code> to a region of device memory into which to construct one or many <code>T</code>s. 
* **`exemplar`** The value from which to copy. 
* **`n`** The number of objects to construct at <code>p</code>. 

**Returns**:
p, casted to <code>T</code>'s type.

**See**:
* <a href="{{ site.baseurl }}/api/classes/classthrust_1_1device__ptr.html">device_ptr</a>
* fill 

<h3 id="function-device-new">
Function <code>thrust::device&#95;new</code>
</h3>

<code class="doxybook">
<span>template &lt;typename T&gt;</span>
<span>device_ptr< T > </span><span><b>device_new</b>(const size_t n = 1);</span></code>
<code>device&#95;new</code> implements the new operator for types resident in device memory. It allocates device memory large enough to hold <code>n</code> new objects of type <code>T</code>.

**Function Parameters**:
**`n`**: The number of objects to allocate. Defaults to <code>1</code>. 

**Returns**:
A <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1device__ptr.html">device&#95;ptr</a></code> to the newly allocated region of device memory. 

<h3 id="function-operator<<">
Function <code>thrust::operator&lt;&lt;</code>
</h3>

<code class="doxybook">
<span>template &lt;typename T,</span>
<span>&nbsp;&nbsp;typename CharT,</span>
<span>&nbsp;&nbsp;typename Traits&gt;</span>
<span>__host__ std::basic_ostream< CharT, Traits > & </span><span><b>operator<<</b>(std::basic_ostream< CharT, Traits > & os,</span>
<span>&nbsp;&nbsp;device_ptr< T > const & dp);</span></code>
Write the address that a <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1device__ptr.html">device&#95;ptr</a></code> points to to an output stream.

**Function Parameters**:
* **`os`** The output stream. 
* **`dp`** The <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1device__ptr.html">device&#95;ptr</a></code> to output.

**Returns**:
<code>os</code>. 

<h3 id="function-device-pointer-cast">
Function <code>thrust::device&#95;pointer&#95;cast</code>
</h3>

<code class="doxybook">
<span>template &lt;typename T&gt;</span>
<span>__host__ __device__ device_ptr< T > </span><span><b>device_pointer_cast</b>(T * ptr);</span></code>
Create a <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1device__ptr.html">device&#95;ptr</a></code> from a raw pointer. 

**Template Parameters**:
**`T`**: Any type. 

**Function Parameters**:
**`ptr`**: A raw pointer to a <code>T</code> in device memory.

**Preconditions**:
<code>ptr</code> points to a location in device memory.

**Returns**:
A <code>device&#95;ptr&lt;T&gt;</code> pointing to <code>ptr</code>. 

<h3 id="function-device-pointer-cast">
Function <code>thrust::device&#95;pointer&#95;cast</code>
</h3>

<code class="doxybook">
<span>template &lt;typename T&gt;</span>
<span>__host__ __device__ device_ptr< T > </span><span><b>device_pointer_cast</b>(device_ptr< T > const & dptr);</span></code>
Create a <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1device__ptr.html">device&#95;ptr</a></code> from another <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1device__ptr.html">device&#95;ptr</a></code>. 

**Template Parameters**:
**`T`**: Any type. 

**Function Parameters**:
**`dptr`**: A <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1device__ptr.html">device&#95;ptr</a></code> to a <code>T</code>. 

<h3 id="function-swap">
Function <code>thrust::swap</code>
</h3>

<code class="doxybook">
<span>template &lt;typename T&gt;</span>
<span>__host__ __device__ void </span><span><b>swap</b>(device_reference< T > & x,</span>
<span>&nbsp;&nbsp;device_reference< T > & y);</span></code>
swaps the value of one <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1device__reference.html">device&#95;reference</a></code> with another. <code>x</code> The first <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1device__reference.html">device&#95;reference</a></code> of interest. <code>y</code> The second <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1device__reference.html">device&#95;reference</a></code> of interest. 

<h3 id="function-operator<<">
Function <code>thrust::operator&lt;&lt;</code>
</h3>

<code class="doxybook">
<span>template &lt;typename T,</span>
<span>&nbsp;&nbsp;typename charT,</span>
<span>&nbsp;&nbsp;typename traits&gt;</span>
<span>std::basic_ostream< charT, traits > & </span><span><b>operator<<</b>(std::basic_ostream< charT, traits > & os,</span>
<span>&nbsp;&nbsp;const device_reference< T > & y);</span></code>
Writes to an output stream the value of a <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1device__reference.html">device&#95;reference</a></code>.

**Function Parameters**:
* **`os`** The output stream. 
* **`y`** The <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1device__reference.html">device&#95;reference</a></code> to output. 

**Returns**:
os. 

<h3 id="function-malloc">
Function <code>thrust::malloc</code>
</h3>

<code class="doxybook">
<span>template &lt;typename DerivedPolicy&gt;</span>
<span>__host__ __device__ pointer< void, DerivedPolicy > </span><span><b>malloc</b>(const thrust::detail::execution_policy_base< DerivedPolicy > & system,</span>
<span>&nbsp;&nbsp;std::size_t n);</span></code>
This version of <code>malloc</code> allocates untyped uninitialized storage associated with a given system.


The following code snippet demonstrates how to use <code>malloc</code> to allocate a range of memory associated with Thrust's device system.



```cpp
#include <thrust/memory.h>
...
// allocate some memory with thrust::malloc
const int N = 100;
thrust::device_system_tag device_sys;
thrust::pointer<void,thrust::device_space_tag> void_ptr = thrust::malloc(device_sys, N);

// manipulate memory
...

// deallocate void_ptr with thrust::free
thrust::free(device_sys, void_ptr);
```

**Template Parameters**:
**`DerivedPolicy`**: The name of the derived execution policy.

**Function Parameters**:
* **`system`** The Thrust system with which to associate the storage. 
* **`n`** The number of bytes of storage to allocate. 

**Preconditions**:
<code>DerivedPolicy</code> must be publically derived from <code>thrust::execution&#95;policy&lt;DerivedPolicy&gt;</code>.

**Returns**:
If allocation succeeds, a pointer to the allocated storage; a null pointer otherwise. The pointer must be deallocated with <code>thrust::free</code>.

**See**:
* free 
* device_malloc 

<h3 id="function-malloc">
Function <code>thrust::malloc</code>
</h3>

<code class="doxybook">
<span>template &lt;typename T,</span>
<span>&nbsp;&nbsp;typename DerivedPolicy&gt;</span>
<span>__host__ __device__ pointer< T, DerivedPolicy > </span><span><b>malloc</b>(const thrust::detail::execution_policy_base< DerivedPolicy > & system,</span>
<span>&nbsp;&nbsp;std::size_t n);</span></code>
This version of <code>malloc</code> allocates typed uninitialized storage associated with a given system.


The following code snippet demonstrates how to use <code>malloc</code> to allocate a range of memory to accomodate integers associated with Thrust's device system.



```cpp
#include <thrust/memory.h>
...
// allocate storage for 100 ints with thrust::malloc
const int N = 100;
thrust::device_system_tag device_sys;
thrust::pointer<int,thrust::device_system_tag> ptr = thrust::malloc<int>(device_sys, N);

// manipulate memory
...

// deallocate ptr with thrust::free
thrust::free(device_sys, ptr);
```

**Template Parameters**:
**`DerivedPolicy`**: The name of the derived execution policy.

**Function Parameters**:
* **`system`** The Thrust system with which to associate the storage. 
* **`n`** The number of elements of type <code>T</code> which the storage should accomodate. 

**Preconditions**:
<code>DerivedPolicy</code> must be publically derived from <code>thrust::execution&#95;policy&lt;DerivedPolicy&gt;</code>.

**Returns**:
If allocation succeeds, a pointer to an allocation large enough to accomodate <code>n</code> elements of type <code>T</code>; a null pointer otherwise. The pointer must be deallocated with <code>thrust::free</code>.

**See**:
* free 
* device_malloc 

<h3 id="function-get-temporary-buffer">
Function <code>thrust::get&#95;temporary&#95;buffer</code>
</h3>

<code class="doxybook">
<span>template &lt;typename T,</span>
<span>&nbsp;&nbsp;typename DerivedPolicy&gt;</span>
<span>__host__ __device__ <a href="{{ site.baseurl }}/api/classes/structthrust_1_1pair.html">thrust::pair</a>< thrust::pointer< T, DerivedPolicy >, typename thrust::pointer< T, DerivedPolicy >::difference_type > </span><span><b>get_temporary_buffer</b>(const thrust::detail::execution_policy_base< DerivedPolicy > & system,</span>
<span>&nbsp;&nbsp;typename thrust::pointer< T, DerivedPolicy >::difference_type n);</span></code>
<code>get&#95;temporary&#95;buffer</code> returns a pointer to storage associated with a given Thrust system sufficient to store up to <code>n</code> objects of type <code>T</code>. If not enough storage is available to accomodate <code>n</code> objects, an implementation may return a smaller buffer. The number of objects the returned buffer can accomodate is also returned.

Thrust uses <code>get&#95;temporary&#95;buffer</code> internally when allocating temporary storage required by algorithm implementations.

The storage allocated with <code>get&#95;temporary&#95;buffer</code> must be returned to the system with <code>return&#95;temporary&#95;buffer</code>.


The following code snippet demonstrates how to use <code>get&#95;temporary&#95;buffer</code> to allocate a range of memory to accomodate integers associated with Thrust's device system.



```cpp
#include <thrust/memory.h>
...
// allocate storage for 100 ints with thrust::get_temporary_buffer
const int N = 100;

typedef thrust::pair<
  thrust::pointer<int,thrust::device_system_tag>,
  std::ptrdiff_t
> ptr_and_size_t;

thrust::device_system_tag device_sys;
ptr_and_size_t ptr_and_size = thrust::get_temporary_buffer<int>(device_sys, N);

// manipulate up to 100 ints
for(int i = 0; i < ptr_and_size.second; ++i)
{
  *ptr_and_size.first = i;
}

// deallocate storage with thrust::return_temporary_buffer
thrust::return_temporary_buffer(device_sys, ptr_and_size.first);
```

**Template Parameters**:
**`DerivedPolicy`**: The name of the derived execution policy.

**Function Parameters**:
* **`system`** The Thrust system with which to associate the storage. 
* **`n`** The requested number of objects of type <code>T</code> the storage should accomodate. 

**Preconditions**:
<code>DerivedPolicy</code> must be publically derived from <code>thrust::execution&#95;policy&lt;DerivedPolicy&gt;</code>.

**Returns**:
A pair <code>p</code> such that <code>p.first</code> is a pointer to the allocated storage and <code>p.second</code> is the number of contiguous objects of type <code>T</code> that the storage can accomodate. If no storage can be allocated, <code>p.first</code> if no storage can be obtained. The storage must be returned to the system using <code>return&#95;temporary&#95;buffer</code>.

**See**:
* malloc 
* return_temporary_buffer 

<h3 id="function-free">
Function <code>thrust::free</code>
</h3>

<code class="doxybook">
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename Pointer&gt;</span>
<span>__host__ __device__ void </span><span><b>free</b>(const thrust::detail::execution_policy_base< DerivedPolicy > & system,</span>
<span>&nbsp;&nbsp;Pointer ptr);</span></code>
<code>free</code> deallocates the storage previously allocated by <code>thrust::malloc</code>.


The following code snippet demonstrates how to use <code>free</code> to deallocate a range of memory previously allocated with <code>thrust::malloc</code>.



```cpp
#include <thrust/memory.h>
...
// allocate storage for 100 ints with thrust::malloc
const int N = 100;
thrust::device_system_tag device_sys;
thrust::pointer<int,thrust::device_system_tag> ptr = thrust::malloc<int>(device_sys, N);

// mainpulate memory
...

// deallocate ptr with thrust::free
thrust::free(device_sys, ptr);
```

**Template Parameters**:
**`DerivedPolicy`**: The name of the derived execution policy.

**Function Parameters**:
* **`system`** The Thrust system with which the storage is associated. 
* **`ptr`** A pointer previously returned by <code>thrust::malloc</code>. If <code>ptr</code> is null, <code>free</code> does nothing.

**Preconditions**:
<code>ptr</code> shall have been returned by a previous call to <code>thrust::malloc(system, n)</code> or <code>thrust::malloc&lt;T&gt;(system, n)</code> for some type <code>T</code>.

<h3 id="function-return-temporary-buffer">
Function <code>thrust::return&#95;temporary&#95;buffer</code>
</h3>

<code class="doxybook">
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename Pointer&gt;</span>
<span>__host__ __device__ void </span><span><b>return_temporary_buffer</b>(const thrust::detail::execution_policy_base< DerivedPolicy > & system,</span>
<span>&nbsp;&nbsp;Pointer p,</span>
<span>&nbsp;&nbsp;std::ptrdiff_t n);</span></code>
<code>return&#95;temporary&#95;buffer</code> deallocates storage associated with a given Thrust system previously allocated by <code>get&#95;temporary&#95;buffer</code>.

Thrust uses <code>return&#95;temporary&#95;buffer</code> internally when deallocating temporary storage required by algorithm implementations.


The following code snippet demonstrates how to use <code>return&#95;temporary&#95;buffer</code> to deallocate a range of memory previously allocated by <code>get&#95;temporary&#95;buffer</code>.



```cpp
#include <thrust/memory.h>
...
// allocate storage for 100 ints with thrust::get_temporary_buffer
const int N = 100;

typedef thrust::pair<
  thrust::pointer<int,thrust::device_system_tag>,
  std::ptrdiff_t
> ptr_and_size_t;

thrust::device_system_tag device_sys;
ptr_and_size_t ptr_and_size = thrust::get_temporary_buffer<int>(device_sys, N);

// manipulate up to 100 ints
for(int i = 0; i < ptr_and_size.second; ++i)
{
  *ptr_and_size.first = i;
}

// deallocate storage with thrust::return_temporary_buffer
thrust::return_temporary_buffer(device_sys, ptr_and_size.first);
```

**Template Parameters**:
**`DerivedPolicy`**: The name of the derived execution policy.

**Function Parameters**:
* **`system`** The Thrust system with which the storage is associated. 
* **`p`** A pointer previously returned by <code>thrust::get&#95;temporary&#95;buffer</code>. If <code>ptr</code> is null, <code>return&#95;temporary&#95;buffer</code> does nothing.

**Preconditions**:
<code>p</code> shall have been previously allocated by <code>thrust::get&#95;temporary&#95;buffer</code>.

**See**:
* free 
* get_temporary_buffer 

<h3 id="function-raw-pointer-cast">
Function <code>thrust::raw&#95;pointer&#95;cast</code>
</h3>

<code class="doxybook">
<span>template &lt;typename Pointer&gt;</span>
<span>__host__ __device__ thrust::detail::pointer_traits< Pointer >::raw_pointer </span><span><b>raw_pointer_cast</b>(Pointer ptr);</span></code>
<code>raw&#95;pointer&#95;cast</code> creates a "raw" pointer from a pointer-like type, simply returning the wrapped pointer, should it exist.

**Function Parameters**:
**`ptr`**: The pointer of interest. 

**Returns**:
<code><a href="{{ site.baseurl }}/api/groups/group__tuple.html#function-get">ptr.get()</a></code>, if the expression is well formed; <code>ptr</code>, otherwise. 

**See**:
raw_reference_cast 

<h3 id="function-raw-reference-cast">
Function <code>thrust::raw&#95;reference&#95;cast</code>
</h3>

<code class="doxybook">
<span>template &lt;typename T&gt;</span>
<span>__host__ __device__ detail::raw_reference< T >::type </span><span><b>raw_reference_cast</b>(T & ref);</span></code>
<code>raw&#95;reference&#95;cast</code> creates a "raw" reference from a wrapped reference type, simply returning the underlying reference, should it exist.

If the argument is not a reference wrapper, the result is a reference to the argument.

**Note**:
There are two versions of <code>raw&#95;reference&#95;cast</code>. One for <code>const</code> references, and one for non-<code>const</code>. 

**Function Parameters**:
**`ref`**: The reference of interest. 

**Returns**:
<code>&#42;thrust::raw&#95;pointer&#95;cast(&ref)</code>. 

**See**:
raw_pointer_cast 

<h3 id="function-raw-reference-cast">
Function <code>thrust::raw&#95;reference&#95;cast</code>
</h3>

<code class="doxybook">
<span>template &lt;typename T&gt;</span>
<span>__host__ __device__ detail::raw_reference< constT >::type </span><span><b>raw_reference_cast</b>(const T & ref);</span></code>
<code>raw&#95;reference&#95;cast</code> creates a "raw" reference from a wrapped reference type, simply returning the underlying reference, should it exist.

If the argument is not a reference wrapper, the result is a reference to the argument.

**Note**:
There are two versions of <code>raw&#95;reference&#95;cast</code>. One for <code>const</code> references, and one for non-<code>const</code>. 

**Function Parameters**:
**`ref`**: The reference of interest. 

**Returns**:
<code>&#42;thrust::raw&#95;pointer&#95;cast(&ref)</code>. 

**See**:
raw_pointer_cast 


