---
title: Set Operations
parent: Algorithms
grand_parent: API
nav_exclude: false
has_children: true
has_toc: false
---

# Set Operations

<code class="doxybook">
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename InputIterator1,</span>
<span>&nbsp;&nbsp;typename InputIterator2,</span>
<span>&nbsp;&nbsp;typename OutputIterator&gt;</span>
<span>__host__ __device__ OutputIterator </span><span><b><a href="{{ site.baseurl }}/api/groups/group__set__operations.html#function-set-difference">thrust::set&#95;difference</a></b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;InputIterator1 first1,</span>
<span>&nbsp;&nbsp;InputIterator1 last1,</span>
<span>&nbsp;&nbsp;InputIterator2 first2,</span>
<span>&nbsp;&nbsp;InputIterator2 last2,</span>
<span>&nbsp;&nbsp;OutputIterator result);</span>
<br>
<span>template &lt;typename InputIterator1,</span>
<span>&nbsp;&nbsp;typename InputIterator2,</span>
<span>&nbsp;&nbsp;typename OutputIterator&gt;</span>
<span>OutputIterator </span><span><b><a href="{{ site.baseurl }}/api/groups/group__set__operations.html#function-set-difference">thrust::set&#95;difference</a></b>(InputIterator1 first1,</span>
<span>&nbsp;&nbsp;InputIterator1 last1,</span>
<span>&nbsp;&nbsp;InputIterator2 first2,</span>
<span>&nbsp;&nbsp;InputIterator2 last2,</span>
<span>&nbsp;&nbsp;OutputIterator result);</span>
<br>
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename InputIterator1,</span>
<span>&nbsp;&nbsp;typename InputIterator2,</span>
<span>&nbsp;&nbsp;typename OutputIterator,</span>
<span>&nbsp;&nbsp;typename StrictWeakCompare&gt;</span>
<span>__host__ __device__ OutputIterator </span><span><b><a href="{{ site.baseurl }}/api/groups/group__set__operations.html#function-set-difference">thrust::set&#95;difference</a></b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;InputIterator1 first1,</span>
<span>&nbsp;&nbsp;InputIterator1 last1,</span>
<span>&nbsp;&nbsp;InputIterator2 first2,</span>
<span>&nbsp;&nbsp;InputIterator2 last2,</span>
<span>&nbsp;&nbsp;OutputIterator result,</span>
<span>&nbsp;&nbsp;StrictWeakCompare comp);</span>
<br>
<span>template &lt;typename InputIterator1,</span>
<span>&nbsp;&nbsp;typename InputIterator2,</span>
<span>&nbsp;&nbsp;typename OutputIterator,</span>
<span>&nbsp;&nbsp;typename StrictWeakCompare&gt;</span>
<span>OutputIterator </span><span><b><a href="{{ site.baseurl }}/api/groups/group__set__operations.html#function-set-difference">thrust::set&#95;difference</a></b>(InputIterator1 first1,</span>
<span>&nbsp;&nbsp;InputIterator1 last1,</span>
<span>&nbsp;&nbsp;InputIterator2 first2,</span>
<span>&nbsp;&nbsp;InputIterator2 last2,</span>
<span>&nbsp;&nbsp;OutputIterator result,</span>
<span>&nbsp;&nbsp;StrictWeakCompare comp);</span>
<br>
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename InputIterator1,</span>
<span>&nbsp;&nbsp;typename InputIterator2,</span>
<span>&nbsp;&nbsp;typename OutputIterator&gt;</span>
<span>__host__ __device__ OutputIterator </span><span><b><a href="{{ site.baseurl }}/api/groups/group__set__operations.html#function-set-intersection">thrust::set&#95;intersection</a></b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;InputIterator1 first1,</span>
<span>&nbsp;&nbsp;InputIterator1 last1,</span>
<span>&nbsp;&nbsp;InputIterator2 first2,</span>
<span>&nbsp;&nbsp;InputIterator2 last2,</span>
<span>&nbsp;&nbsp;OutputIterator result);</span>
<br>
<span>template &lt;typename InputIterator1,</span>
<span>&nbsp;&nbsp;typename InputIterator2,</span>
<span>&nbsp;&nbsp;typename OutputIterator&gt;</span>
<span>OutputIterator </span><span><b><a href="{{ site.baseurl }}/api/groups/group__set__operations.html#function-set-intersection">thrust::set&#95;intersection</a></b>(InputIterator1 first1,</span>
<span>&nbsp;&nbsp;InputIterator1 last1,</span>
<span>&nbsp;&nbsp;InputIterator2 first2,</span>
<span>&nbsp;&nbsp;InputIterator2 last2,</span>
<span>&nbsp;&nbsp;OutputIterator result);</span>
<br>
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename InputIterator1,</span>
<span>&nbsp;&nbsp;typename InputIterator2,</span>
<span>&nbsp;&nbsp;typename OutputIterator,</span>
<span>&nbsp;&nbsp;typename StrictWeakCompare&gt;</span>
<span>__host__ __device__ OutputIterator </span><span><b><a href="{{ site.baseurl }}/api/groups/group__set__operations.html#function-set-intersection">thrust::set&#95;intersection</a></b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;InputIterator1 first1,</span>
<span>&nbsp;&nbsp;InputIterator1 last1,</span>
<span>&nbsp;&nbsp;InputIterator2 first2,</span>
<span>&nbsp;&nbsp;InputIterator2 last2,</span>
<span>&nbsp;&nbsp;OutputIterator result,</span>
<span>&nbsp;&nbsp;StrictWeakCompare comp);</span>
<br>
<span>template &lt;typename InputIterator1,</span>
<span>&nbsp;&nbsp;typename InputIterator2,</span>
<span>&nbsp;&nbsp;typename OutputIterator,</span>
<span>&nbsp;&nbsp;typename StrictWeakCompare&gt;</span>
<span>OutputIterator </span><span><b><a href="{{ site.baseurl }}/api/groups/group__set__operations.html#function-set-intersection">thrust::set&#95;intersection</a></b>(InputIterator1 first1,</span>
<span>&nbsp;&nbsp;InputIterator1 last1,</span>
<span>&nbsp;&nbsp;InputIterator2 first2,</span>
<span>&nbsp;&nbsp;InputIterator2 last2,</span>
<span>&nbsp;&nbsp;OutputIterator result,</span>
<span>&nbsp;&nbsp;StrictWeakCompare comp);</span>
<br>
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename InputIterator1,</span>
<span>&nbsp;&nbsp;typename InputIterator2,</span>
<span>&nbsp;&nbsp;typename OutputIterator&gt;</span>
<span>__host__ __device__ OutputIterator </span><span><b><a href="{{ site.baseurl }}/api/groups/group__set__operations.html#function-set-symmetric-difference">thrust::set&#95;symmetric&#95;difference</a></b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;InputIterator1 first1,</span>
<span>&nbsp;&nbsp;InputIterator1 last1,</span>
<span>&nbsp;&nbsp;InputIterator2 first2,</span>
<span>&nbsp;&nbsp;InputIterator2 last2,</span>
<span>&nbsp;&nbsp;OutputIterator result);</span>
<br>
<span>template &lt;typename InputIterator1,</span>
<span>&nbsp;&nbsp;typename InputIterator2,</span>
<span>&nbsp;&nbsp;typename OutputIterator&gt;</span>
<span>OutputIterator </span><span><b><a href="{{ site.baseurl }}/api/groups/group__set__operations.html#function-set-symmetric-difference">thrust::set&#95;symmetric&#95;difference</a></b>(InputIterator1 first1,</span>
<span>&nbsp;&nbsp;InputIterator1 last1,</span>
<span>&nbsp;&nbsp;InputIterator2 first2,</span>
<span>&nbsp;&nbsp;InputIterator2 last2,</span>
<span>&nbsp;&nbsp;OutputIterator result);</span>
<br>
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename InputIterator1,</span>
<span>&nbsp;&nbsp;typename InputIterator2,</span>
<span>&nbsp;&nbsp;typename OutputIterator,</span>
<span>&nbsp;&nbsp;typename StrictWeakCompare&gt;</span>
<span>__host__ __device__ OutputIterator </span><span><b><a href="{{ site.baseurl }}/api/groups/group__set__operations.html#function-set-symmetric-difference">thrust::set&#95;symmetric&#95;difference</a></b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;InputIterator1 first1,</span>
<span>&nbsp;&nbsp;InputIterator1 last1,</span>
<span>&nbsp;&nbsp;InputIterator2 first2,</span>
<span>&nbsp;&nbsp;InputIterator2 last2,</span>
<span>&nbsp;&nbsp;OutputIterator result,</span>
<span>&nbsp;&nbsp;StrictWeakCompare comp);</span>
<br>
<span>template &lt;typename InputIterator1,</span>
<span>&nbsp;&nbsp;typename InputIterator2,</span>
<span>&nbsp;&nbsp;typename OutputIterator,</span>
<span>&nbsp;&nbsp;typename StrictWeakCompare&gt;</span>
<span>OutputIterator </span><span><b><a href="{{ site.baseurl }}/api/groups/group__set__operations.html#function-set-symmetric-difference">thrust::set&#95;symmetric&#95;difference</a></b>(InputIterator1 first1,</span>
<span>&nbsp;&nbsp;InputIterator1 last1,</span>
<span>&nbsp;&nbsp;InputIterator2 first2,</span>
<span>&nbsp;&nbsp;InputIterator2 last2,</span>
<span>&nbsp;&nbsp;OutputIterator result,</span>
<span>&nbsp;&nbsp;StrictWeakCompare comp);</span>
<br>
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename InputIterator1,</span>
<span>&nbsp;&nbsp;typename InputIterator2,</span>
<span>&nbsp;&nbsp;typename OutputIterator&gt;</span>
<span>__host__ __device__ OutputIterator </span><span><b><a href="{{ site.baseurl }}/api/groups/group__set__operations.html#function-set-union">thrust::set&#95;union</a></b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;InputIterator1 first1,</span>
<span>&nbsp;&nbsp;InputIterator1 last1,</span>
<span>&nbsp;&nbsp;InputIterator2 first2,</span>
<span>&nbsp;&nbsp;InputIterator2 last2,</span>
<span>&nbsp;&nbsp;OutputIterator result);</span>
<br>
<span>template &lt;typename InputIterator1,</span>
<span>&nbsp;&nbsp;typename InputIterator2,</span>
<span>&nbsp;&nbsp;typename OutputIterator&gt;</span>
<span>OutputIterator </span><span><b><a href="{{ site.baseurl }}/api/groups/group__set__operations.html#function-set-union">thrust::set&#95;union</a></b>(InputIterator1 first1,</span>
<span>&nbsp;&nbsp;InputIterator1 last1,</span>
<span>&nbsp;&nbsp;InputIterator2 first2,</span>
<span>&nbsp;&nbsp;InputIterator2 last2,</span>
<span>&nbsp;&nbsp;OutputIterator result);</span>
<br>
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename InputIterator1,</span>
<span>&nbsp;&nbsp;typename InputIterator2,</span>
<span>&nbsp;&nbsp;typename OutputIterator,</span>
<span>&nbsp;&nbsp;typename StrictWeakCompare&gt;</span>
<span>__host__ __device__ OutputIterator </span><span><b><a href="{{ site.baseurl }}/api/groups/group__set__operations.html#function-set-union">thrust::set&#95;union</a></b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;InputIterator1 first1,</span>
<span>&nbsp;&nbsp;InputIterator1 last1,</span>
<span>&nbsp;&nbsp;InputIterator2 first2,</span>
<span>&nbsp;&nbsp;InputIterator2 last2,</span>
<span>&nbsp;&nbsp;OutputIterator result,</span>
<span>&nbsp;&nbsp;StrictWeakCompare comp);</span>
<br>
<span>template &lt;typename InputIterator1,</span>
<span>&nbsp;&nbsp;typename InputIterator2,</span>
<span>&nbsp;&nbsp;typename OutputIterator,</span>
<span>&nbsp;&nbsp;typename StrictWeakCompare&gt;</span>
<span>OutputIterator </span><span><b><a href="{{ site.baseurl }}/api/groups/group__set__operations.html#function-set-union">thrust::set&#95;union</a></b>(InputIterator1 first1,</span>
<span>&nbsp;&nbsp;InputIterator1 last1,</span>
<span>&nbsp;&nbsp;InputIterator2 first2,</span>
<span>&nbsp;&nbsp;InputIterator2 last2,</span>
<span>&nbsp;&nbsp;OutputIterator result,</span>
<span>&nbsp;&nbsp;StrictWeakCompare comp);</span>
<br>
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename InputIterator1,</span>
<span>&nbsp;&nbsp;typename InputIterator2,</span>
<span>&nbsp;&nbsp;typename InputIterator3,</span>
<span>&nbsp;&nbsp;typename InputIterator4,</span>
<span>&nbsp;&nbsp;typename OutputIterator1,</span>
<span>&nbsp;&nbsp;typename OutputIterator2&gt;</span>
<span>__host__ __device__ <a href="{{ site.baseurl }}/api/classes/structthrust_1_1pair.html">thrust::pair</a>< OutputIterator1, OutputIterator2 > </span><span><b><a href="{{ site.baseurl }}/api/groups/group__set__operations.html#function-set-difference-by-key">thrust::set&#95;difference&#95;by&#95;key</a></b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;InputIterator1 keys_first1,</span>
<span>&nbsp;&nbsp;InputIterator1 keys_last1,</span>
<span>&nbsp;&nbsp;InputIterator2 keys_first2,</span>
<span>&nbsp;&nbsp;InputIterator2 keys_last2,</span>
<span>&nbsp;&nbsp;InputIterator3 values_first1,</span>
<span>&nbsp;&nbsp;InputIterator4 values_first2,</span>
<span>&nbsp;&nbsp;OutputIterator1 keys_result,</span>
<span>&nbsp;&nbsp;OutputIterator2 values_result);</span>
<br>
<span>template &lt;typename InputIterator1,</span>
<span>&nbsp;&nbsp;typename InputIterator2,</span>
<span>&nbsp;&nbsp;typename InputIterator3,</span>
<span>&nbsp;&nbsp;typename InputIterator4,</span>
<span>&nbsp;&nbsp;typename OutputIterator1,</span>
<span>&nbsp;&nbsp;typename OutputIterator2&gt;</span>
<span><a href="{{ site.baseurl }}/api/classes/structthrust_1_1pair.html">thrust::pair</a>< OutputIterator1, OutputIterator2 > </span><span><b><a href="{{ site.baseurl }}/api/groups/group__set__operations.html#function-set-difference-by-key">thrust::set&#95;difference&#95;by&#95;key</a></b>(InputIterator1 keys_first1,</span>
<span>&nbsp;&nbsp;InputIterator1 keys_last1,</span>
<span>&nbsp;&nbsp;InputIterator2 keys_first2,</span>
<span>&nbsp;&nbsp;InputIterator2 keys_last2,</span>
<span>&nbsp;&nbsp;InputIterator3 values_first1,</span>
<span>&nbsp;&nbsp;InputIterator4 values_first2,</span>
<span>&nbsp;&nbsp;OutputIterator1 keys_result,</span>
<span>&nbsp;&nbsp;OutputIterator2 values_result);</span>
<br>
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename InputIterator1,</span>
<span>&nbsp;&nbsp;typename InputIterator2,</span>
<span>&nbsp;&nbsp;typename InputIterator3,</span>
<span>&nbsp;&nbsp;typename InputIterator4,</span>
<span>&nbsp;&nbsp;typename OutputIterator1,</span>
<span>&nbsp;&nbsp;typename OutputIterator2,</span>
<span>&nbsp;&nbsp;typename StrictWeakCompare&gt;</span>
<span>__host__ __device__ <a href="{{ site.baseurl }}/api/classes/structthrust_1_1pair.html">thrust::pair</a>< OutputIterator1, OutputIterator2 > </span><span><b><a href="{{ site.baseurl }}/api/groups/group__set__operations.html#function-set-difference-by-key">thrust::set&#95;difference&#95;by&#95;key</a></b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;InputIterator1 keys_first1,</span>
<span>&nbsp;&nbsp;InputIterator1 keys_last1,</span>
<span>&nbsp;&nbsp;InputIterator2 keys_first2,</span>
<span>&nbsp;&nbsp;InputIterator2 keys_last2,</span>
<span>&nbsp;&nbsp;InputIterator3 values_first1,</span>
<span>&nbsp;&nbsp;InputIterator4 values_first2,</span>
<span>&nbsp;&nbsp;OutputIterator1 keys_result,</span>
<span>&nbsp;&nbsp;OutputIterator2 values_result,</span>
<span>&nbsp;&nbsp;StrictWeakCompare comp);</span>
<br>
<span>template &lt;typename InputIterator1,</span>
<span>&nbsp;&nbsp;typename InputIterator2,</span>
<span>&nbsp;&nbsp;typename InputIterator3,</span>
<span>&nbsp;&nbsp;typename InputIterator4,</span>
<span>&nbsp;&nbsp;typename OutputIterator1,</span>
<span>&nbsp;&nbsp;typename OutputIterator2,</span>
<span>&nbsp;&nbsp;typename StrictWeakCompare&gt;</span>
<span><a href="{{ site.baseurl }}/api/classes/structthrust_1_1pair.html">thrust::pair</a>< OutputIterator1, OutputIterator2 > </span><span><b><a href="{{ site.baseurl }}/api/groups/group__set__operations.html#function-set-difference-by-key">thrust::set&#95;difference&#95;by&#95;key</a></b>(InputIterator1 keys_first1,</span>
<span>&nbsp;&nbsp;InputIterator1 keys_last1,</span>
<span>&nbsp;&nbsp;InputIterator2 keys_first2,</span>
<span>&nbsp;&nbsp;InputIterator2 keys_last2,</span>
<span>&nbsp;&nbsp;InputIterator3 values_first1,</span>
<span>&nbsp;&nbsp;InputIterator4 values_first2,</span>
<span>&nbsp;&nbsp;OutputIterator1 keys_result,</span>
<span>&nbsp;&nbsp;OutputIterator2 values_result,</span>
<span>&nbsp;&nbsp;StrictWeakCompare comp);</span>
<br>
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename InputIterator1,</span>
<span>&nbsp;&nbsp;typename InputIterator2,</span>
<span>&nbsp;&nbsp;typename InputIterator3,</span>
<span>&nbsp;&nbsp;typename OutputIterator1,</span>
<span>&nbsp;&nbsp;typename OutputIterator2&gt;</span>
<span>__host__ __device__ <a href="{{ site.baseurl }}/api/classes/structthrust_1_1pair.html">thrust::pair</a>< OutputIterator1, OutputIterator2 > </span><span><b><a href="{{ site.baseurl }}/api/groups/group__set__operations.html#function-set-intersection-by-key">thrust::set&#95;intersection&#95;by&#95;key</a></b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;InputIterator1 keys_first1,</span>
<span>&nbsp;&nbsp;InputIterator1 keys_last1,</span>
<span>&nbsp;&nbsp;InputIterator2 keys_first2,</span>
<span>&nbsp;&nbsp;InputIterator2 keys_last2,</span>
<span>&nbsp;&nbsp;InputIterator3 values_first1,</span>
<span>&nbsp;&nbsp;OutputIterator1 keys_result,</span>
<span>&nbsp;&nbsp;OutputIterator2 values_result);</span>
<br>
<span>template &lt;typename InputIterator1,</span>
<span>&nbsp;&nbsp;typename InputIterator2,</span>
<span>&nbsp;&nbsp;typename InputIterator3,</span>
<span>&nbsp;&nbsp;typename OutputIterator1,</span>
<span>&nbsp;&nbsp;typename OutputIterator2&gt;</span>
<span><a href="{{ site.baseurl }}/api/classes/structthrust_1_1pair.html">thrust::pair</a>< OutputIterator1, OutputIterator2 > </span><span><b><a href="{{ site.baseurl }}/api/groups/group__set__operations.html#function-set-intersection-by-key">thrust::set&#95;intersection&#95;by&#95;key</a></b>(InputIterator1 keys_first1,</span>
<span>&nbsp;&nbsp;InputIterator1 keys_last1,</span>
<span>&nbsp;&nbsp;InputIterator2 keys_first2,</span>
<span>&nbsp;&nbsp;InputIterator2 keys_last2,</span>
<span>&nbsp;&nbsp;InputIterator3 values_first1,</span>
<span>&nbsp;&nbsp;OutputIterator1 keys_result,</span>
<span>&nbsp;&nbsp;OutputIterator2 values_result);</span>
<br>
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename InputIterator1,</span>
<span>&nbsp;&nbsp;typename InputIterator2,</span>
<span>&nbsp;&nbsp;typename InputIterator3,</span>
<span>&nbsp;&nbsp;typename OutputIterator1,</span>
<span>&nbsp;&nbsp;typename OutputIterator2,</span>
<span>&nbsp;&nbsp;typename StrictWeakCompare&gt;</span>
<span>__host__ __device__ <a href="{{ site.baseurl }}/api/classes/structthrust_1_1pair.html">thrust::pair</a>< OutputIterator1, OutputIterator2 > </span><span><b><a href="{{ site.baseurl }}/api/groups/group__set__operations.html#function-set-intersection-by-key">thrust::set&#95;intersection&#95;by&#95;key</a></b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;InputIterator1 keys_first1,</span>
<span>&nbsp;&nbsp;InputIterator1 keys_last1,</span>
<span>&nbsp;&nbsp;InputIterator2 keys_first2,</span>
<span>&nbsp;&nbsp;InputIterator2 keys_last2,</span>
<span>&nbsp;&nbsp;InputIterator3 values_first1,</span>
<span>&nbsp;&nbsp;OutputIterator1 keys_result,</span>
<span>&nbsp;&nbsp;OutputIterator2 values_result,</span>
<span>&nbsp;&nbsp;StrictWeakCompare comp);</span>
<br>
<span>template &lt;typename InputIterator1,</span>
<span>&nbsp;&nbsp;typename InputIterator2,</span>
<span>&nbsp;&nbsp;typename InputIterator3,</span>
<span>&nbsp;&nbsp;typename OutputIterator1,</span>
<span>&nbsp;&nbsp;typename OutputIterator2,</span>
<span>&nbsp;&nbsp;typename StrictWeakCompare&gt;</span>
<span><a href="{{ site.baseurl }}/api/classes/structthrust_1_1pair.html">thrust::pair</a>< OutputIterator1, OutputIterator2 > </span><span><b><a href="{{ site.baseurl }}/api/groups/group__set__operations.html#function-set-intersection-by-key">thrust::set&#95;intersection&#95;by&#95;key</a></b>(InputIterator1 keys_first1,</span>
<span>&nbsp;&nbsp;InputIterator1 keys_last1,</span>
<span>&nbsp;&nbsp;InputIterator2 keys_first2,</span>
<span>&nbsp;&nbsp;InputIterator2 keys_last2,</span>
<span>&nbsp;&nbsp;InputIterator3 values_first1,</span>
<span>&nbsp;&nbsp;OutputIterator1 keys_result,</span>
<span>&nbsp;&nbsp;OutputIterator2 values_result,</span>
<span>&nbsp;&nbsp;StrictWeakCompare comp);</span>
<br>
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename InputIterator1,</span>
<span>&nbsp;&nbsp;typename InputIterator2,</span>
<span>&nbsp;&nbsp;typename InputIterator3,</span>
<span>&nbsp;&nbsp;typename InputIterator4,</span>
<span>&nbsp;&nbsp;typename OutputIterator1,</span>
<span>&nbsp;&nbsp;typename OutputIterator2&gt;</span>
<span>__host__ __device__ <a href="{{ site.baseurl }}/api/classes/structthrust_1_1pair.html">thrust::pair</a>< OutputIterator1, OutputIterator2 > </span><span><b><a href="{{ site.baseurl }}/api/groups/group__set__operations.html#function-set-symmetric-difference-by-key">thrust::set&#95;symmetric&#95;difference&#95;by&#95;key</a></b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;InputIterator1 keys_first1,</span>
<span>&nbsp;&nbsp;InputIterator1 keys_last1,</span>
<span>&nbsp;&nbsp;InputIterator2 keys_first2,</span>
<span>&nbsp;&nbsp;InputIterator2 keys_last2,</span>
<span>&nbsp;&nbsp;InputIterator3 values_first1,</span>
<span>&nbsp;&nbsp;InputIterator4 values_first2,</span>
<span>&nbsp;&nbsp;OutputIterator1 keys_result,</span>
<span>&nbsp;&nbsp;OutputIterator2 values_result);</span>
<br>
<span>template &lt;typename InputIterator1,</span>
<span>&nbsp;&nbsp;typename InputIterator2,</span>
<span>&nbsp;&nbsp;typename InputIterator3,</span>
<span>&nbsp;&nbsp;typename InputIterator4,</span>
<span>&nbsp;&nbsp;typename OutputIterator1,</span>
<span>&nbsp;&nbsp;typename OutputIterator2&gt;</span>
<span><a href="{{ site.baseurl }}/api/classes/structthrust_1_1pair.html">thrust::pair</a>< OutputIterator1, OutputIterator2 > </span><span><b><a href="{{ site.baseurl }}/api/groups/group__set__operations.html#function-set-symmetric-difference-by-key">thrust::set&#95;symmetric&#95;difference&#95;by&#95;key</a></b>(InputIterator1 keys_first1,</span>
<span>&nbsp;&nbsp;InputIterator1 keys_last1,</span>
<span>&nbsp;&nbsp;InputIterator2 keys_first2,</span>
<span>&nbsp;&nbsp;InputIterator2 keys_last2,</span>
<span>&nbsp;&nbsp;InputIterator3 values_first1,</span>
<span>&nbsp;&nbsp;InputIterator4 values_first2,</span>
<span>&nbsp;&nbsp;OutputIterator1 keys_result,</span>
<span>&nbsp;&nbsp;OutputIterator2 values_result);</span>
<br>
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename InputIterator1,</span>
<span>&nbsp;&nbsp;typename InputIterator2,</span>
<span>&nbsp;&nbsp;typename InputIterator3,</span>
<span>&nbsp;&nbsp;typename InputIterator4,</span>
<span>&nbsp;&nbsp;typename OutputIterator1,</span>
<span>&nbsp;&nbsp;typename OutputIterator2,</span>
<span>&nbsp;&nbsp;typename StrictWeakCompare&gt;</span>
<span>__host__ __device__ <a href="{{ site.baseurl }}/api/classes/structthrust_1_1pair.html">thrust::pair</a>< OutputIterator1, OutputIterator2 > </span><span><b><a href="{{ site.baseurl }}/api/groups/group__set__operations.html#function-set-symmetric-difference-by-key">thrust::set&#95;symmetric&#95;difference&#95;by&#95;key</a></b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;InputIterator1 keys_first1,</span>
<span>&nbsp;&nbsp;InputIterator1 keys_last1,</span>
<span>&nbsp;&nbsp;InputIterator2 keys_first2,</span>
<span>&nbsp;&nbsp;InputIterator2 keys_last2,</span>
<span>&nbsp;&nbsp;InputIterator3 values_first1,</span>
<span>&nbsp;&nbsp;InputIterator4 values_first2,</span>
<span>&nbsp;&nbsp;OutputIterator1 keys_result,</span>
<span>&nbsp;&nbsp;OutputIterator2 values_result,</span>
<span>&nbsp;&nbsp;StrictWeakCompare comp);</span>
<br>
<span>template &lt;typename InputIterator1,</span>
<span>&nbsp;&nbsp;typename InputIterator2,</span>
<span>&nbsp;&nbsp;typename InputIterator3,</span>
<span>&nbsp;&nbsp;typename InputIterator4,</span>
<span>&nbsp;&nbsp;typename OutputIterator1,</span>
<span>&nbsp;&nbsp;typename OutputIterator2,</span>
<span>&nbsp;&nbsp;typename StrictWeakCompare&gt;</span>
<span><a href="{{ site.baseurl }}/api/classes/structthrust_1_1pair.html">thrust::pair</a>< OutputIterator1, OutputIterator2 > </span><span><b><a href="{{ site.baseurl }}/api/groups/group__set__operations.html#function-set-symmetric-difference-by-key">thrust::set&#95;symmetric&#95;difference&#95;by&#95;key</a></b>(InputIterator1 keys_first1,</span>
<span>&nbsp;&nbsp;InputIterator1 keys_last1,</span>
<span>&nbsp;&nbsp;InputIterator2 keys_first2,</span>
<span>&nbsp;&nbsp;InputIterator2 keys_last2,</span>
<span>&nbsp;&nbsp;InputIterator3 values_first1,</span>
<span>&nbsp;&nbsp;InputIterator4 values_first2,</span>
<span>&nbsp;&nbsp;OutputIterator1 keys_result,</span>
<span>&nbsp;&nbsp;OutputIterator2 values_result,</span>
<span>&nbsp;&nbsp;StrictWeakCompare comp);</span>
<br>
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename InputIterator1,</span>
<span>&nbsp;&nbsp;typename InputIterator2,</span>
<span>&nbsp;&nbsp;typename InputIterator3,</span>
<span>&nbsp;&nbsp;typename InputIterator4,</span>
<span>&nbsp;&nbsp;typename OutputIterator1,</span>
<span>&nbsp;&nbsp;typename OutputIterator2&gt;</span>
<span>__host__ __device__ <a href="{{ site.baseurl }}/api/classes/structthrust_1_1pair.html">thrust::pair</a>< OutputIterator1, OutputIterator2 > </span><span><b><a href="{{ site.baseurl }}/api/groups/group__set__operations.html#function-set-union-by-key">thrust::set&#95;union&#95;by&#95;key</a></b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;InputIterator1 keys_first1,</span>
<span>&nbsp;&nbsp;InputIterator1 keys_last1,</span>
<span>&nbsp;&nbsp;InputIterator2 keys_first2,</span>
<span>&nbsp;&nbsp;InputIterator2 keys_last2,</span>
<span>&nbsp;&nbsp;InputIterator3 values_first1,</span>
<span>&nbsp;&nbsp;InputIterator4 values_first2,</span>
<span>&nbsp;&nbsp;OutputIterator1 keys_result,</span>
<span>&nbsp;&nbsp;OutputIterator2 values_result);</span>
<br>
<span>template &lt;typename InputIterator1,</span>
<span>&nbsp;&nbsp;typename InputIterator2,</span>
<span>&nbsp;&nbsp;typename InputIterator3,</span>
<span>&nbsp;&nbsp;typename InputIterator4,</span>
<span>&nbsp;&nbsp;typename OutputIterator1,</span>
<span>&nbsp;&nbsp;typename OutputIterator2&gt;</span>
<span><a href="{{ site.baseurl }}/api/classes/structthrust_1_1pair.html">thrust::pair</a>< OutputIterator1, OutputIterator2 > </span><span><b><a href="{{ site.baseurl }}/api/groups/group__set__operations.html#function-set-union-by-key">thrust::set&#95;union&#95;by&#95;key</a></b>(InputIterator1 keys_first1,</span>
<span>&nbsp;&nbsp;InputIterator1 keys_last1,</span>
<span>&nbsp;&nbsp;InputIterator2 keys_first2,</span>
<span>&nbsp;&nbsp;InputIterator2 keys_last2,</span>
<span>&nbsp;&nbsp;InputIterator3 values_first1,</span>
<span>&nbsp;&nbsp;InputIterator4 values_first2,</span>
<span>&nbsp;&nbsp;OutputIterator1 keys_result,</span>
<span>&nbsp;&nbsp;OutputIterator2 values_result);</span>
<br>
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename InputIterator1,</span>
<span>&nbsp;&nbsp;typename InputIterator2,</span>
<span>&nbsp;&nbsp;typename InputIterator3,</span>
<span>&nbsp;&nbsp;typename InputIterator4,</span>
<span>&nbsp;&nbsp;typename OutputIterator1,</span>
<span>&nbsp;&nbsp;typename OutputIterator2,</span>
<span>&nbsp;&nbsp;typename StrictWeakCompare&gt;</span>
<span>__host__ __device__ <a href="{{ site.baseurl }}/api/classes/structthrust_1_1pair.html">thrust::pair</a>< OutputIterator1, OutputIterator2 > </span><span><b><a href="{{ site.baseurl }}/api/groups/group__set__operations.html#function-set-union-by-key">thrust::set&#95;union&#95;by&#95;key</a></b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;InputIterator1 keys_first1,</span>
<span>&nbsp;&nbsp;InputIterator1 keys_last1,</span>
<span>&nbsp;&nbsp;InputIterator2 keys_first2,</span>
<span>&nbsp;&nbsp;InputIterator2 keys_last2,</span>
<span>&nbsp;&nbsp;InputIterator3 values_first1,</span>
<span>&nbsp;&nbsp;InputIterator4 values_first2,</span>
<span>&nbsp;&nbsp;OutputIterator1 keys_result,</span>
<span>&nbsp;&nbsp;OutputIterator2 values_result,</span>
<span>&nbsp;&nbsp;StrictWeakCompare comp);</span>
<br>
<span>template &lt;typename InputIterator1,</span>
<span>&nbsp;&nbsp;typename InputIterator2,</span>
<span>&nbsp;&nbsp;typename InputIterator3,</span>
<span>&nbsp;&nbsp;typename InputIterator4,</span>
<span>&nbsp;&nbsp;typename OutputIterator1,</span>
<span>&nbsp;&nbsp;typename OutputIterator2,</span>
<span>&nbsp;&nbsp;typename StrictWeakCompare&gt;</span>
<span><a href="{{ site.baseurl }}/api/classes/structthrust_1_1pair.html">thrust::pair</a>< OutputIterator1, OutputIterator2 > </span><span><b><a href="{{ site.baseurl }}/api/groups/group__set__operations.html#function-set-union-by-key">thrust::set&#95;union&#95;by&#95;key</a></b>(InputIterator1 keys_first1,</span>
<span>&nbsp;&nbsp;InputIterator1 keys_last1,</span>
<span>&nbsp;&nbsp;InputIterator2 keys_first2,</span>
<span>&nbsp;&nbsp;InputIterator2 keys_last2,</span>
<span>&nbsp;&nbsp;InputIterator3 values_first1,</span>
<span>&nbsp;&nbsp;InputIterator4 values_first2,</span>
<span>&nbsp;&nbsp;OutputIterator1 keys_result,</span>
<span>&nbsp;&nbsp;OutputIterator2 values_result,</span>
<span>&nbsp;&nbsp;StrictWeakCompare comp);</span>
</code>

## Functions

<h3 id="function-set-difference">
Function <code>thrust::set&#95;difference</code>
</h3>

<code class="doxybook">
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename InputIterator1,</span>
<span>&nbsp;&nbsp;typename InputIterator2,</span>
<span>&nbsp;&nbsp;typename OutputIterator&gt;</span>
<span>__host__ __device__ OutputIterator </span><span><b>set_difference</b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;InputIterator1 first1,</span>
<span>&nbsp;&nbsp;InputIterator1 last1,</span>
<span>&nbsp;&nbsp;InputIterator2 first2,</span>
<span>&nbsp;&nbsp;InputIterator2 last2,</span>
<span>&nbsp;&nbsp;OutputIterator result);</span></code>
<code>set&#95;difference</code> constructs a sorted range that is the set difference of the sorted ranges <code>[first1, last1)</code> and <code>[first2, last2)</code>. The return value is the end of the output range.

In the simplest case, <code>set&#95;difference</code> performs the "difference" operation from set theory: the output range contains a copy of every element that is contained in <code>[first1, last1)</code> and not contained in <code>[first2, last1)</code>. The general case is more complicated, because the input ranges may contain duplicate elements. The generalization is that if <code>[first1, last1)</code> contains <code>m</code> elements that are equivalent to each other and if <code>[first2, last2)</code> contains <code>n</code> elements that are equivalent to them, the last <code>max(m-n,0)</code> elements from <code>[first1, last1)</code> range shall be copied to the output range.

This version of <code>set&#95;difference</code> compares elements using <code>operator&lt;</code>.

The algorithm's execution is parallelized as determined by <code>exec</code>.


The following code snippet demonstrates how to use <code>set&#95;difference</code> to compute the set difference of two sets of integers sorted in ascending order using the <code>thrust::host</code> execution policy for parallelization:



```cpp
#include <thrust/set_operations.h>
#include <thrust/execution_policy.h>
...
int A1[7] = {0, 1, 3, 4, 5, 6, 9};
int A2[5] = {1, 3, 5, 7, 9};

int result[3];

int *result_end = thrust::set_difference(thrust::host, A1, A1 + 7, A2, A2 + 5, result);
// result is now {0, 4, 6}
```

**Template Parameters**:
* **`DerivedPolicy`** The name of the derived execution policy. 
* **`InputIterator1`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>, <code>InputIterator1</code> and <code>InputIterator2</code> have the same <code>value&#95;type</code>, <code>InputIterator1's</code><code>value&#95;type</code> is a model of <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a>, the ordering on <code>InputIterator1's</code><code>value&#95;type</code> is a strict weak ordering, as defined in the <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a> requirements, and <code>InputIterator1's</code><code>value&#95;type</code> is convertable to a type in <code>OutputIterator's</code> set of <code>value&#95;types</code>. 
* **`InputIterator2`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>, <code>InputIterator2</code> and <code>InputIterator1</code> have the same <code>value&#95;type</code>, <code>InputIterator2's</code><code>value&#95;type</code> is a model of <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a>, the ordering on <code>InputIterator2's</code><code>value&#95;type</code> is a strict weak ordering, as defined in the <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a> requirements, and <code>InputIterator2's</code><code>value&#95;type</code> is convertable to a type in <code>OutputIterator's</code> set of <code>value&#95;types</code>. 
* **`OutputIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>.

**Function Parameters**:
* **`exec`** The execution policy to use for parallelization. 
* **`first1`** The beginning of the first input range. 
* **`last1`** The end of the first input range. 
* **`first2`** The beginning of the second input range. 
* **`last2`** The end of the second input range. 
* **`result`** The beginning of the output range. 

**Preconditions**:
* The ranges <code>[first1, last1)</code> and <code>[first2, last2)</code> shall be sorted with respect to <code>operator&lt;</code>. 
* The resulting range shall not overlap with either input range.

**Returns**:
The end of the output range.

**See**:
* <a href="https://en.cppreference.com/w/cpp/algorithm/set_difference">https://en.cppreference.com/w/cpp/algorithm/set_difference</a>
* <code>includes</code>
* <code>set&#95;union</code>
* <code>set&#95;intersection</code>
* <code>set&#95;symmetric&#95;difference</code>
* <code>sort</code>
* <code>is&#95;sorted</code>

<h3 id="function-set-difference">
Function <code>thrust::set&#95;difference</code>
</h3>

<code class="doxybook">
<span>template &lt;typename InputIterator1,</span>
<span>&nbsp;&nbsp;typename InputIterator2,</span>
<span>&nbsp;&nbsp;typename OutputIterator&gt;</span>
<span>OutputIterator </span><span><b>set_difference</b>(InputIterator1 first1,</span>
<span>&nbsp;&nbsp;InputIterator1 last1,</span>
<span>&nbsp;&nbsp;InputIterator2 first2,</span>
<span>&nbsp;&nbsp;InputIterator2 last2,</span>
<span>&nbsp;&nbsp;OutputIterator result);</span></code>
<code>set&#95;difference</code> constructs a sorted range that is the set difference of the sorted ranges <code>[first1, last1)</code> and <code>[first2, last2)</code>. The return value is the end of the output range.

In the simplest case, <code>set&#95;difference</code> performs the "difference" operation from set theory: the output range contains a copy of every element that is contained in <code>[first1, last1)</code> and not contained in <code>[first2, last1)</code>. The general case is more complicated, because the input ranges may contain duplicate elements. The generalization is that if <code>[first1, last1)</code> contains <code>m</code> elements that are equivalent to each other and if <code>[first2, last2)</code> contains <code>n</code> elements that are equivalent to them, the last <code>max(m-n,0)</code> elements from <code>[first1, last1)</code> range shall be copied to the output range.

This version of <code>set&#95;difference</code> compares elements using <code>operator&lt;</code>.


The following code snippet demonstrates how to use <code>set&#95;difference</code> to compute the set difference of two sets of integers sorted in ascending order.



```cpp
#include <thrust/set_operations.h>
...
int A1[7] = {0, 1, 3, 4, 5, 6, 9};
int A2[5] = {1, 3, 5, 7, 9};

int result[3];

int *result_end = thrust::set_difference(A1, A1 + 7, A2, A2 + 5, result);
// result is now {0, 4, 6}
```

**Template Parameters**:
* **`InputIterator1`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>, <code>InputIterator1</code> and <code>InputIterator2</code> have the same <code>value&#95;type</code>, <code>InputIterator1's</code><code>value&#95;type</code> is a model of <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a>, the ordering on <code>InputIterator1's</code><code>value&#95;type</code> is a strict weak ordering, as defined in the <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a> requirements, and <code>InputIterator1's</code><code>value&#95;type</code> is convertable to a type in <code>OutputIterator's</code> set of <code>value&#95;types</code>. 
* **`InputIterator2`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>, <code>InputIterator2</code> and <code>InputIterator1</code> have the same <code>value&#95;type</code>, <code>InputIterator2's</code><code>value&#95;type</code> is a model of <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a>, the ordering on <code>InputIterator2's</code><code>value&#95;type</code> is a strict weak ordering, as defined in the <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a> requirements, and <code>InputIterator2's</code><code>value&#95;type</code> is convertable to a type in <code>OutputIterator's</code> set of <code>value&#95;types</code>. 
* **`OutputIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>.

**Function Parameters**:
* **`first1`** The beginning of the first input range. 
* **`last1`** The end of the first input range. 
* **`first2`** The beginning of the second input range. 
* **`last2`** The end of the second input range. 
* **`result`** The beginning of the output range. 

**Preconditions**:
* The ranges <code>[first1, last1)</code> and <code>[first2, last2)</code> shall be sorted with respect to <code>operator&lt;</code>. 
* The resulting range shall not overlap with either input range.

**Returns**:
The end of the output range.

**See**:
* <a href="https://en.cppreference.com/w/cpp/algorithm/set_difference">https://en.cppreference.com/w/cpp/algorithm/set_difference</a>
* <code>includes</code>
* <code>set&#95;union</code>
* <code>set&#95;intersection</code>
* <code>set&#95;symmetric&#95;difference</code>
* <code>sort</code>
* <code>is&#95;sorted</code>

<h3 id="function-set-difference">
Function <code>thrust::set&#95;difference</code>
</h3>

<code class="doxybook">
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename InputIterator1,</span>
<span>&nbsp;&nbsp;typename InputIterator2,</span>
<span>&nbsp;&nbsp;typename OutputIterator,</span>
<span>&nbsp;&nbsp;typename StrictWeakCompare&gt;</span>
<span>__host__ __device__ OutputIterator </span><span><b>set_difference</b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;InputIterator1 first1,</span>
<span>&nbsp;&nbsp;InputIterator1 last1,</span>
<span>&nbsp;&nbsp;InputIterator2 first2,</span>
<span>&nbsp;&nbsp;InputIterator2 last2,</span>
<span>&nbsp;&nbsp;OutputIterator result,</span>
<span>&nbsp;&nbsp;StrictWeakCompare comp);</span></code>
<code>set&#95;difference</code> constructs a sorted range that is the set difference of the sorted ranges <code>[first1, last1)</code> and <code>[first2, last2)</code>. The return value is the end of the output range.

In the simplest case, <code>set&#95;difference</code> performs the "difference" operation from set theory: the output range contains a copy of every element that is contained in <code>[first1, last1)</code> and not contained in <code>[first2, last1)</code>. The general case is more complicated, because the input ranges may contain duplicate elements. The generalization is that if <code>[first1, last1)</code> contains <code>m</code> elements that are equivalent to each other and if <code>[first2, last2)</code> contains <code>n</code> elements that are equivalent to them, the last <code>max(m-n,0)</code> elements from <code>[first1, last1)</code> range shall be copied to the output range.

This version of <code>set&#95;difference</code> compares elements using a function object <code>comp</code>.

The algorithm's execution is parallelized as determined by <code>exec</code>.


The following code snippet demonstrates how to use <code>set&#95;difference</code> to compute the set difference of two sets of integers sorted in descending order using the <code>thrust::host</code> execution policy for parallelization:



```cpp
#include <thrust/set_operations.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>
...
int A1[7] = {9, 6, 5, 4, 3, 1, 0};
int A2[5] = {9, 7, 5, 3, 1};

int result[3];

int *result_end = thrust::set_difference(thrust::host, A1, A1 + 7, A2, A2 + 5, result, thrust::greater<int>());
// result is now {6, 4, 0}
```

**Template Parameters**:
* **`DerivedPolicy`** The name of the derived execution policy. 
* **`InputIterator1`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>, <code>InputIterator1's</code><code>value&#95;type</code> is convertable to <code>StrictWeakCompare's</code><code>first&#95;argument&#95;type</code>. and <code>InputIterator1's</code><code>value&#95;type</code> is convertable to a type in <code>OutputIterator's</code> set of <code>value&#95;types</code>. 
* **`InputIterator2`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>, <code>InputIterator2's</code><code>value&#95;type</code> is convertable to <code>StrictWeakCompare's</code><code>second&#95;argument&#95;type</code>. and <code>InputIterator2's</code><code>value&#95;type</code> is convertable to a type in <code>OutputIterator's</code> set of <code>value&#95;types</code>. 
* **`OutputIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>. 
* **`StrictWeakCompare`** is a model of <a href="https://en.cppreference.com/w/cpp/concepts/strict_weak_order">Strict Weak Ordering</a>.

**Function Parameters**:
* **`exec`** The execution policy to use for parallelization. 
* **`first1`** The beginning of the first input range. 
* **`last1`** The end of the first input range. 
* **`first2`** The beginning of the second input range. 
* **`last2`** The end of the second input range. 
* **`result`** The beginning of the output range. 
* **`comp`** Comparison operator. 

**Preconditions**:
* The ranges <code>[first1, last1)</code> and <code>[first2, last2)</code> shall be sorted with respect to <code>comp</code>. 
* The resulting range shall not overlap with either input range.

**Returns**:
The end of the output range.

**See**:
* <a href="https://en.cppreference.com/w/cpp/algorithm/set_difference">https://en.cppreference.com/w/cpp/algorithm/set_difference</a>
* <code>includes</code>
* <code>set&#95;union</code>
* <code>set&#95;intersection</code>
* <code>set&#95;symmetric&#95;difference</code>
* <code>sort</code>
* <code>is&#95;sorted</code>

<h3 id="function-set-difference">
Function <code>thrust::set&#95;difference</code>
</h3>

<code class="doxybook">
<span>template &lt;typename InputIterator1,</span>
<span>&nbsp;&nbsp;typename InputIterator2,</span>
<span>&nbsp;&nbsp;typename OutputIterator,</span>
<span>&nbsp;&nbsp;typename StrictWeakCompare&gt;</span>
<span>OutputIterator </span><span><b>set_difference</b>(InputIterator1 first1,</span>
<span>&nbsp;&nbsp;InputIterator1 last1,</span>
<span>&nbsp;&nbsp;InputIterator2 first2,</span>
<span>&nbsp;&nbsp;InputIterator2 last2,</span>
<span>&nbsp;&nbsp;OutputIterator result,</span>
<span>&nbsp;&nbsp;StrictWeakCompare comp);</span></code>
<code>set&#95;difference</code> constructs a sorted range that is the set difference of the sorted ranges <code>[first1, last1)</code> and <code>[first2, last2)</code>. The return value is the end of the output range.

In the simplest case, <code>set&#95;difference</code> performs the "difference" operation from set theory: the output range contains a copy of every element that is contained in <code>[first1, last1)</code> and not contained in <code>[first2, last1)</code>. The general case is more complicated, because the input ranges may contain duplicate elements. The generalization is that if <code>[first1, last1)</code> contains <code>m</code> elements that are equivalent to each other and if <code>[first2, last2)</code> contains <code>n</code> elements that are equivalent to them, the last <code>max(m-n,0)</code> elements from <code>[first1, last1)</code> range shall be copied to the output range.

This version of <code>set&#95;difference</code> compares elements using a function object <code>comp</code>.


The following code snippet demonstrates how to use <code>set&#95;difference</code> to compute the set difference of two sets of integers sorted in descending order.



```cpp
#include <thrust/set_operations.h>
#include <thrust/functional.h>
...
int A1[7] = {9, 6, 5, 4, 3, 1, 0};
int A2[5] = {9, 7, 5, 3, 1};

int result[3];

int *result_end = thrust::set_difference(A1, A1 + 7, A2, A2 + 5, result, thrust::greater<int>());
// result is now {6, 4, 0}
```

**Template Parameters**:
* **`InputIterator1`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>, <code>InputIterator1's</code><code>value&#95;type</code> is convertable to <code>StrictWeakCompare's</code><code>first&#95;argument&#95;type</code>. and <code>InputIterator1's</code><code>value&#95;type</code> is convertable to a type in <code>OutputIterator's</code> set of <code>value&#95;types</code>. 
* **`InputIterator2`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>, <code>InputIterator2's</code><code>value&#95;type</code> is convertable to <code>StrictWeakCompare's</code><code>second&#95;argument&#95;type</code>. and <code>InputIterator2's</code><code>value&#95;type</code> is convertable to a type in <code>OutputIterator's</code> set of <code>value&#95;types</code>. 
* **`OutputIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>. 
* **`StrictWeakCompare`** is a model of <a href="https://en.cppreference.com/w/cpp/concepts/strict_weak_order">Strict Weak Ordering</a>.

**Function Parameters**:
* **`first1`** The beginning of the first input range. 
* **`last1`** The end of the first input range. 
* **`first2`** The beginning of the second input range. 
* **`last2`** The end of the second input range. 
* **`result`** The beginning of the output range. 
* **`comp`** Comparison operator. 

**Preconditions**:
* The ranges <code>[first1, last1)</code> and <code>[first2, last2)</code> shall be sorted with respect to <code>comp</code>. 
* The resulting range shall not overlap with either input range.

**Returns**:
The end of the output range.

**See**:
* <a href="https://en.cppreference.com/w/cpp/algorithm/set_difference">https://en.cppreference.com/w/cpp/algorithm/set_difference</a>
* <code>includes</code>
* <code>set&#95;union</code>
* <code>set&#95;intersection</code>
* <code>set&#95;symmetric&#95;difference</code>
* <code>sort</code>
* <code>is&#95;sorted</code>

<h3 id="function-set-intersection">
Function <code>thrust::set&#95;intersection</code>
</h3>

<code class="doxybook">
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename InputIterator1,</span>
<span>&nbsp;&nbsp;typename InputIterator2,</span>
<span>&nbsp;&nbsp;typename OutputIterator&gt;</span>
<span>__host__ __device__ OutputIterator </span><span><b>set_intersection</b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;InputIterator1 first1,</span>
<span>&nbsp;&nbsp;InputIterator1 last1,</span>
<span>&nbsp;&nbsp;InputIterator2 first2,</span>
<span>&nbsp;&nbsp;InputIterator2 last2,</span>
<span>&nbsp;&nbsp;OutputIterator result);</span></code>
<code>set&#95;intersection</code> constructs a sorted range that is the intersection of sorted ranges <code>[first1, last1)</code> and <code>[first2, last2)</code>. The return value is the end of the output range.

In the simplest case, <code>set&#95;intersection</code> performs the "intersection" operation from set theory: the output range contains a copy of every element that is contained in both <code>[first1, last1)</code> and <code>[first2, last2)</code>. The general case is more complicated, because the input ranges may contain duplicate elements. The generalization is that if a value appears <code>m</code> times in <code>[first1, last1)</code> and <code>n</code> times in <code>[first2, last2)</code> (where <code>m</code> may be zero), then it appears <code>min(m,n)</code> times in the output range. <code>set&#95;intersection</code> is stable, meaning that both elements are copied from the first range rather than the second, and that the relative order of elements in the output range is the same as in the first input range.

This version of <code>set&#95;intersection</code> compares objects using <code>operator&lt;</code>.

The algorithm's execution is parallelized as determined by <code>exec</code>.


The following code snippet demonstrates how to use <code>set&#95;intersection</code> to compute the set intersection of two sets of integers sorted in ascending order using the <code>thrust::host</code> execution policy for parallelization:



```cpp
#include <thrust/set_operations.h>
#include <thrust/execution_policy.h>
...
int A1[6] = {1, 3, 5, 7, 9, 11};
int A2[7] = {1, 1, 2, 3, 5,  8, 13};

int result[7];

int *result_end = thrust::set_intersection(thrust::host, A1, A1 + 6, A2, A2 + 7, result);
// result is now {1, 3, 5}
```

**Template Parameters**:
* **`DerivedPolicy`** The name of the derived execution policy. 
* **`InputIterator1`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>, <code>InputIterator1</code> and <code>InputIterator2</code> have the same <code>value&#95;type</code>, <code>InputIterator1's</code><code>value&#95;type</code> is a model of <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a>, the ordering on <code>InputIterator1's</code><code>value&#95;type</code> is a strict weak ordering, as defined in the <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a> requirements, and <code>InputIterator1's</code><code>value&#95;type</code> is convertable to a type in <code>OutputIterator's</code> set of <code>value&#95;types</code>. 
* **`InputIterator2`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>, <code>InputIterator2</code> and <code>InputIterator1</code> have the same <code>value&#95;type</code>, <code>InputIterator2's</code><code>value&#95;type</code> is a model of <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a>, the ordering on <code>InputIterator2's</code><code>value&#95;type</code> is a strict weak ordering, as defined in the <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a> requirements, and <code>InputIterator2's</code><code>value&#95;type</code> is convertable to a type in <code>OutputIterator's</code> set of <code>value&#95;types</code>. 
* **`OutputIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>.

**Function Parameters**:
* **`exec`** The execution policy to use for parallelization. 
* **`first1`** The beginning of the first input range. 
* **`last1`** The end of the first input range. 
* **`first2`** The beginning of the second input range. 
* **`last2`** The end of the second input range. 
* **`result`** The beginning of the output range. 

**Preconditions**:
* The ranges <code>[first1, last1)</code> and <code>[first2, last2)</code> shall be sorted with respect to <code>operator&lt;</code>. 
* The resulting range shall not overlap with either input range.

**Returns**:
The end of the output range.

**See**:
* <a href="https://en.cppreference.com/w/cpp/algorithm/set_intersection">https://en.cppreference.com/w/cpp/algorithm/set_intersection</a>
* <code>includes</code>
* <code>set&#95;union</code>
* <code>set&#95;intersection</code>
* <code>set&#95;symmetric&#95;difference</code>
* <code>sort</code>
* <code>is&#95;sorted</code>

<h3 id="function-set-intersection">
Function <code>thrust::set&#95;intersection</code>
</h3>

<code class="doxybook">
<span>template &lt;typename InputIterator1,</span>
<span>&nbsp;&nbsp;typename InputIterator2,</span>
<span>&nbsp;&nbsp;typename OutputIterator&gt;</span>
<span>OutputIterator </span><span><b>set_intersection</b>(InputIterator1 first1,</span>
<span>&nbsp;&nbsp;InputIterator1 last1,</span>
<span>&nbsp;&nbsp;InputIterator2 first2,</span>
<span>&nbsp;&nbsp;InputIterator2 last2,</span>
<span>&nbsp;&nbsp;OutputIterator result);</span></code>
<code>set&#95;intersection</code> constructs a sorted range that is the intersection of sorted ranges <code>[first1, last1)</code> and <code>[first2, last2)</code>. The return value is the end of the output range.

In the simplest case, <code>set&#95;intersection</code> performs the "intersection" operation from set theory: the output range contains a copy of every element that is contained in both <code>[first1, last1)</code> and <code>[first2, last2)</code>. The general case is more complicated, because the input ranges may contain duplicate elements. The generalization is that if a value appears <code>m</code> times in <code>[first1, last1)</code> and <code>n</code> times in <code>[first2, last2)</code> (where <code>m</code> may be zero), then it appears <code>min(m,n)</code> times in the output range. <code>set&#95;intersection</code> is stable, meaning that both elements are copied from the first range rather than the second, and that the relative order of elements in the output range is the same as in the first input range.

This version of <code>set&#95;intersection</code> compares objects using <code>operator&lt;</code>.


The following code snippet demonstrates how to use <code>set&#95;intersection</code> to compute the set intersection of two sets of integers sorted in ascending order.



```cpp
#include <thrust/set_operations.h>
...
int A1[6] = {1, 3, 5, 7, 9, 11};
int A2[7] = {1, 1, 2, 3, 5,  8, 13};

int result[7];

int *result_end = thrust::set_intersection(A1, A1 + 6, A2, A2 + 7, result);
// result is now {1, 3, 5}
```

**Template Parameters**:
* **`InputIterator1`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>, <code>InputIterator1</code> and <code>InputIterator2</code> have the same <code>value&#95;type</code>, <code>InputIterator1's</code><code>value&#95;type</code> is a model of <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a>, the ordering on <code>InputIterator1's</code><code>value&#95;type</code> is a strict weak ordering, as defined in the <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a> requirements, and <code>InputIterator1's</code><code>value&#95;type</code> is convertable to a type in <code>OutputIterator's</code> set of <code>value&#95;types</code>. 
* **`InputIterator2`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>, <code>InputIterator2</code> and <code>InputIterator1</code> have the same <code>value&#95;type</code>, <code>InputIterator2's</code><code>value&#95;type</code> is a model of <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a>, the ordering on <code>InputIterator2's</code><code>value&#95;type</code> is a strict weak ordering, as defined in the <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a> requirements, and <code>InputIterator2's</code><code>value&#95;type</code> is convertable to a type in <code>OutputIterator's</code> set of <code>value&#95;types</code>. 
* **`OutputIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>.

**Function Parameters**:
* **`first1`** The beginning of the first input range. 
* **`last1`** The end of the first input range. 
* **`first2`** The beginning of the second input range. 
* **`last2`** The end of the second input range. 
* **`result`** The beginning of the output range. 

**Preconditions**:
* The ranges <code>[first1, last1)</code> and <code>[first2, last2)</code> shall be sorted with respect to <code>operator&lt;</code>. 
* The resulting range shall not overlap with either input range.

**Returns**:
The end of the output range.

**See**:
* <a href="https://en.cppreference.com/w/cpp/algorithm/set_intersection">https://en.cppreference.com/w/cpp/algorithm/set_intersection</a>
* <code>includes</code>
* <code>set&#95;union</code>
* <code>set&#95;intersection</code>
* <code>set&#95;symmetric&#95;difference</code>
* <code>sort</code>
* <code>is&#95;sorted</code>

<h3 id="function-set-intersection">
Function <code>thrust::set&#95;intersection</code>
</h3>

<code class="doxybook">
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename InputIterator1,</span>
<span>&nbsp;&nbsp;typename InputIterator2,</span>
<span>&nbsp;&nbsp;typename OutputIterator,</span>
<span>&nbsp;&nbsp;typename StrictWeakCompare&gt;</span>
<span>__host__ __device__ OutputIterator </span><span><b>set_intersection</b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;InputIterator1 first1,</span>
<span>&nbsp;&nbsp;InputIterator1 last1,</span>
<span>&nbsp;&nbsp;InputIterator2 first2,</span>
<span>&nbsp;&nbsp;InputIterator2 last2,</span>
<span>&nbsp;&nbsp;OutputIterator result,</span>
<span>&nbsp;&nbsp;StrictWeakCompare comp);</span></code>
<code>set&#95;intersection</code> constructs a sorted range that is the intersection of sorted ranges <code>[first1, last1)</code> and <code>[first2, last2)</code>. The return value is the end of the output range.

In the simplest case, <code>set&#95;intersection</code> performs the "intersection" operation from set theory: the output range contains a copy of every element that is contained in both <code>[first1, last1)</code> and <code>[first2, last2)</code>. The general case is more complicated, because the input ranges may contain duplicate elements. The generalization is that if a value appears <code>m</code> times in <code>[first1, last1)</code> and <code>n</code> times in <code>[first2, last2)</code> (where <code>m</code> may be zero), then it appears <code>min(m,n)</code> times in the output range. <code>set&#95;intersection</code> is stable, meaning that both elements are copied from the first range rather than the second, and that the relative order of elements in the output range is the same as in the first input range.

This version of <code>set&#95;intersection</code> compares elements using a function object <code>comp</code>.

The algorithm's execution is parallelized as determined by <code>exec</code>.


The following code snippet demonstrates how to use <code>set&#95;intersection</code> to compute the set intersection of sets of integers sorted in descending order using the <code>thrust::host</code> execution policy for parallelization:



```cpp
#include <thrust/set_operations.h>
#include <thrust/execution_policy.h>
...
int A1[6] = {11, 9, 7, 5, 3, 1};
int A2[7] = {13, 8, 5, 3, 2,  1, 1};

int result[3];

int *result_end = thrust::set_intersection(thrust::host, A1, A1 + 6, A2, A2 + 7, result, thrust::greater<int>());
// result is now {5, 3, 1}
```

**Template Parameters**:
* **`DerivedPolicy`** The name of the derived execution policy. 
* **`InputIterator1`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>, <code>InputIterator1</code> and <code>InputIterator2</code> have the same <code>value&#95;type</code>, <code>InputIterator1's</code><code>value&#95;type</code> is a model of <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a>, the ordering on <code>InputIterator1's</code><code>value&#95;type</code> is a strict weak ordering, as defined in the <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a> requirements, and <code>InputIterator1's</code><code>value&#95;type</code> is convertable to a type in <code>OutputIterator's</code> set of <code>value&#95;types</code>. 
* **`InputIterator2`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>, <code>InputIterator2</code> and <code>InputIterator1</code> have the same <code>value&#95;type</code>, <code>InputIterator2's</code><code>value&#95;type</code> is a model of <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a>, the ordering on <code>InputIterator2's</code><code>value&#95;type</code> is a strict weak ordering, as defined in the <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a> requirements, and <code>InputIterator2's</code><code>value&#95;type</code> is convertable to a type in <code>OutputIterator's</code> set of <code>value&#95;types</code>. 
* **`OutputIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>.

**Function Parameters**:
* **`exec`** The execution policy to use for parallelization. 
* **`first1`** The beginning of the first input range. 
* **`last1`** The end of the first input range. 
* **`first2`** The beginning of the second input range. 
* **`last2`** The end of the second input range. 
* **`result`** The beginning of the output range. 
* **`comp`** Comparison operator. 

**Preconditions**:
* The ranges <code>[first1, last1)</code> and <code>[first2, last2)</code> shall be sorted with respect to <code>comp</code>. 
* The resulting range shall not overlap with either input range.

**Returns**:
The end of the output range.

**See**:
* <a href="https://en.cppreference.com/w/cpp/algorithm/set_intersection">https://en.cppreference.com/w/cpp/algorithm/set_intersection</a>
* <code>includes</code>
* <code>set&#95;union</code>
* <code>set&#95;intersection</code>
* <code>set&#95;symmetric&#95;difference</code>
* <code>sort</code>
* <code>is&#95;sorted</code>

<h3 id="function-set-intersection">
Function <code>thrust::set&#95;intersection</code>
</h3>

<code class="doxybook">
<span>template &lt;typename InputIterator1,</span>
<span>&nbsp;&nbsp;typename InputIterator2,</span>
<span>&nbsp;&nbsp;typename OutputIterator,</span>
<span>&nbsp;&nbsp;typename StrictWeakCompare&gt;</span>
<span>OutputIterator </span><span><b>set_intersection</b>(InputIterator1 first1,</span>
<span>&nbsp;&nbsp;InputIterator1 last1,</span>
<span>&nbsp;&nbsp;InputIterator2 first2,</span>
<span>&nbsp;&nbsp;InputIterator2 last2,</span>
<span>&nbsp;&nbsp;OutputIterator result,</span>
<span>&nbsp;&nbsp;StrictWeakCompare comp);</span></code>
<code>set&#95;intersection</code> constructs a sorted range that is the intersection of sorted ranges <code>[first1, last1)</code> and <code>[first2, last2)</code>. The return value is the end of the output range.

In the simplest case, <code>set&#95;intersection</code> performs the "intersection" operation from set theory: the output range contains a copy of every element that is contained in both <code>[first1, last1)</code> and <code>[first2, last2)</code>. The general case is more complicated, because the input ranges may contain duplicate elements. The generalization is that if a value appears <code>m</code> times in <code>[first1, last1)</code> and <code>n</code> times in <code>[first2, last2)</code> (where <code>m</code> may be zero), then it appears <code>min(m,n)</code> times in the output range. <code>set&#95;intersection</code> is stable, meaning that both elements are copied from the first range rather than the second, and that the relative order of elements in the output range is the same as in the first input range.

This version of <code>set&#95;intersection</code> compares elements using a function object <code>comp</code>.


The following code snippet demonstrates how to use <code>set&#95;intersection</code> to compute the set intersection of sets of integers sorted in descending order.



```cpp
#include <thrust/set_operations.h>
...
int A1[6] = {11, 9, 7, 5, 3, 1};
int A2[7] = {13, 8, 5, 3, 2,  1, 1};

int result[3];

int *result_end = thrust::set_intersection(A1, A1 + 6, A2, A2 + 7, result, thrust::greater<int>());
// result is now {5, 3, 1}
```

**Template Parameters**:
* **`InputIterator1`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>, <code>InputIterator1</code> and <code>InputIterator2</code> have the same <code>value&#95;type</code>, <code>InputIterator1's</code><code>value&#95;type</code> is a model of <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a>, the ordering on <code>InputIterator1's</code><code>value&#95;type</code> is a strict weak ordering, as defined in the <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a> requirements, and <code>InputIterator1's</code><code>value&#95;type</code> is convertable to a type in <code>OutputIterator's</code> set of <code>value&#95;types</code>. 
* **`InputIterator2`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>, <code>InputIterator2</code> and <code>InputIterator1</code> have the same <code>value&#95;type</code>, <code>InputIterator2's</code><code>value&#95;type</code> is a model of <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a>, the ordering on <code>InputIterator2's</code><code>value&#95;type</code> is a strict weak ordering, as defined in the <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a> requirements, and <code>InputIterator2's</code><code>value&#95;type</code> is convertable to a type in <code>OutputIterator's</code> set of <code>value&#95;types</code>. 
* **`OutputIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>.

**Function Parameters**:
* **`first1`** The beginning of the first input range. 
* **`last1`** The end of the first input range. 
* **`first2`** The beginning of the second input range. 
* **`last2`** The end of the second input range. 
* **`result`** The beginning of the output range. 
* **`comp`** Comparison operator. 

**Preconditions**:
* The ranges <code>[first1, last1)</code> and <code>[first2, last2)</code> shall be sorted with respect to <code>comp</code>. 
* The resulting range shall not overlap with either input range.

**Returns**:
The end of the output range.

**See**:
* <a href="https://en.cppreference.com/w/cpp/algorithm/set_intersection">https://en.cppreference.com/w/cpp/algorithm/set_intersection</a>
* <code>includes</code>
* <code>set&#95;union</code>
* <code>set&#95;intersection</code>
* <code>set&#95;symmetric&#95;difference</code>
* <code>sort</code>
* <code>is&#95;sorted</code>

<h3 id="function-set-symmetric-difference">
Function <code>thrust::set&#95;symmetric&#95;difference</code>
</h3>

<code class="doxybook">
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename InputIterator1,</span>
<span>&nbsp;&nbsp;typename InputIterator2,</span>
<span>&nbsp;&nbsp;typename OutputIterator&gt;</span>
<span>__host__ __device__ OutputIterator </span><span><b>set_symmetric_difference</b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;InputIterator1 first1,</span>
<span>&nbsp;&nbsp;InputIterator1 last1,</span>
<span>&nbsp;&nbsp;InputIterator2 first2,</span>
<span>&nbsp;&nbsp;InputIterator2 last2,</span>
<span>&nbsp;&nbsp;OutputIterator result);</span></code>
<code>set&#95;symmetric&#95;difference</code> constructs a sorted range that is the set symmetric difference of the sorted ranges <code>[first1, last1)</code> and <code>[first2, last2)</code>. The return value is the end of the output range.

In the simplest case, <code>set&#95;symmetric&#95;difference</code> performs a set theoretic calculation: it constructs the union of the two sets A - B and B - A, where A and B are the two input ranges. That is, the output range contains a copy of every element that is contained in <code>[first1, last1)</code> but not <code>[first2, last1)</code>, and a copy of every element that is contained in <code>[first2, last2)</code> but not <code>[first1, last1)</code>. The general case is more complicated, because the input ranges may contain duplicate elements. The generalization is that if <code>[first1, last1)</code> contains <code>m</code> elements that are equivalent to each other and <code>[first2, last1)</code> contains <code>n</code> elements that are equivalent to them, then <code>|m - n|</code> of those elements shall be copied to the output range: the last <code>m - n</code> elements from <code>[first1, last1)</code> if <code>m &gt; n</code>, and the last <code>n - m</code> of these elements from <code>[first2, last2)</code> if <code>m &lt; n</code>.

This version of <code>set&#95;union</code> compares elements using <code>operator&lt;</code>.

The algorithm's execution is parallelized as determined by <code>exec</code>.


The following code snippet demonstrates how to use <code>set&#95;symmetric&#95;difference</code> to compute the symmetric difference of two sets of integers sorted in ascending order using the <code>thrust::host</code> execution policy for parallelization:



```cpp
#include <thrust/set_operations.h>
#include <thrust/execution_policy.h>
...
int A1[7] = {0, 1, 2, 2, 4, 6, 7};
int A2[5] = {1, 1, 2, 5, 8};

int result[6];

int *result_end = thrust::set_symmetric_difference(thrust::host, A1, A1 + 7, A2, A2 + 5, result);
// result = {0, 4, 5, 6, 7, 8}
```

**Template Parameters**:
* **`DerivedPolicy`** The name of the derived execution policy. 
* **`InputIterator1`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>, <code>InputIterator1</code> and <code>InputIterator2</code> have the same <code>value&#95;type</code>, <code>InputIterator1's</code><code>value&#95;type</code> is a model of <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a>, the ordering on <code>InputIterator1's</code><code>value&#95;type</code> is a strict weak ordering, as defined in the <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a> requirements, and <code>InputIterator1's</code><code>value&#95;type</code> is convertable to a type in <code>OutputIterator's</code> set of <code>value&#95;types</code>. 
* **`InputIterator2`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>, <code>InputIterator2</code> and <code>InputIterator1</code> have the same <code>value&#95;type</code>, <code>InputIterator2's</code><code>value&#95;type</code> is a model of <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a>, the ordering on <code>InputIterator2's</code><code>value&#95;type</code> is a strict weak ordering, as defined in the <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a> requirements, and <code>InputIterator2's</code><code>value&#95;type</code> is convertable to a type in <code>OutputIterator's</code> set of <code>value&#95;types</code>. 
* **`OutputIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>.

**Function Parameters**:
* **`exec`** The execution policy to use for parallelization. 
* **`first1`** The beginning of the first input range. 
* **`last1`** The end of the first input range. 
* **`first2`** The beginning of the second input range. 
* **`last2`** The end of the second input range. 
* **`result`** The beginning of the output range. 

**Preconditions**:
* The ranges <code>[first1, last1)</code> and <code>[first2, last2)</code> shall be sorted with respect to <code>operator&lt;</code>. 
* The resulting range shall not overlap with either input range.

**Returns**:
The end of the output range.

**See**:
* <a href="https://en.cppreference.com/w/cpp/algorithm/set_symmetric_difference">https://en.cppreference.com/w/cpp/algorithm/set_symmetric_difference</a>
* <code>merge</code>
* <code>includes</code>
* <code>set&#95;difference</code>
* <code>set&#95;union</code>
* <code>set&#95;intersection</code>
* <code>sort</code>
* <code>is&#95;sorted</code>

<h3 id="function-set-symmetric-difference">
Function <code>thrust::set&#95;symmetric&#95;difference</code>
</h3>

<code class="doxybook">
<span>template &lt;typename InputIterator1,</span>
<span>&nbsp;&nbsp;typename InputIterator2,</span>
<span>&nbsp;&nbsp;typename OutputIterator&gt;</span>
<span>OutputIterator </span><span><b>set_symmetric_difference</b>(InputIterator1 first1,</span>
<span>&nbsp;&nbsp;InputIterator1 last1,</span>
<span>&nbsp;&nbsp;InputIterator2 first2,</span>
<span>&nbsp;&nbsp;InputIterator2 last2,</span>
<span>&nbsp;&nbsp;OutputIterator result);</span></code>
<code>set&#95;symmetric&#95;difference</code> constructs a sorted range that is the set symmetric difference of the sorted ranges <code>[first1, last1)</code> and <code>[first2, last2)</code>. The return value is the end of the output range.

In the simplest case, <code>set&#95;symmetric&#95;difference</code> performs a set theoretic calculation: it constructs the union of the two sets A - B and B - A, where A and B are the two input ranges. That is, the output range contains a copy of every element that is contained in <code>[first1, last1)</code> but not <code>[first2, last1)</code>, and a copy of every element that is contained in <code>[first2, last2)</code> but not <code>[first1, last1)</code>. The general case is more complicated, because the input ranges may contain duplicate elements. The generalization is that if <code>[first1, last1)</code> contains <code>m</code> elements that are equivalent to each other and <code>[first2, last1)</code> contains <code>n</code> elements that are equivalent to them, then <code>|m - n|</code> of those elements shall be copied to the output range: the last <code>m - n</code> elements from <code>[first1, last1)</code> if <code>m &gt; n</code>, and the last <code>n - m</code> of these elements from <code>[first2, last2)</code> if <code>m &lt; n</code>.

This version of <code>set&#95;union</code> compares elements using <code>operator&lt;</code>.


The following code snippet demonstrates how to use <code>set&#95;symmetric&#95;difference</code> to compute the symmetric difference of two sets of integers sorted in ascending order.



```cpp
#include <thrust/set_operations.h>
...
int A1[7] = {0, 1, 2, 2, 4, 6, 7};
int A2[5] = {1, 1, 2, 5, 8};

int result[6];

int *result_end = thrust::set_symmetric_difference(A1, A1 + 7, A2, A2 + 5, result);
// result = {0, 4, 5, 6, 7, 8}
```

**Template Parameters**:
* **`InputIterator1`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>, <code>InputIterator1</code> and <code>InputIterator2</code> have the same <code>value&#95;type</code>, <code>InputIterator1's</code><code>value&#95;type</code> is a model of <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a>, the ordering on <code>InputIterator1's</code><code>value&#95;type</code> is a strict weak ordering, as defined in the <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a> requirements, and <code>InputIterator1's</code><code>value&#95;type</code> is convertable to a type in <code>OutputIterator's</code> set of <code>value&#95;types</code>. 
* **`InputIterator2`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>, <code>InputIterator2</code> and <code>InputIterator1</code> have the same <code>value&#95;type</code>, <code>InputIterator2's</code><code>value&#95;type</code> is a model of <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a>, the ordering on <code>InputIterator2's</code><code>value&#95;type</code> is a strict weak ordering, as defined in the <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a> requirements, and <code>InputIterator2's</code><code>value&#95;type</code> is convertable to a type in <code>OutputIterator's</code> set of <code>value&#95;types</code>. 
* **`OutputIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>.

**Function Parameters**:
* **`first1`** The beginning of the first input range. 
* **`last1`** The end of the first input range. 
* **`first2`** The beginning of the second input range. 
* **`last2`** The end of the second input range. 
* **`result`** The beginning of the output range. 

**Preconditions**:
* The ranges <code>[first1, last1)</code> and <code>[first2, last2)</code> shall be sorted with respect to <code>operator&lt;</code>. 
* The resulting range shall not overlap with either input range.

**Returns**:
The end of the output range.

**See**:
* <a href="https://en.cppreference.com/w/cpp/algorithm/set_symmetric_difference">https://en.cppreference.com/w/cpp/algorithm/set_symmetric_difference</a>
* <code>merge</code>
* <code>includes</code>
* <code>set&#95;difference</code>
* <code>set&#95;union</code>
* <code>set&#95;intersection</code>
* <code>sort</code>
* <code>is&#95;sorted</code>

<h3 id="function-set-symmetric-difference">
Function <code>thrust::set&#95;symmetric&#95;difference</code>
</h3>

<code class="doxybook">
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename InputIterator1,</span>
<span>&nbsp;&nbsp;typename InputIterator2,</span>
<span>&nbsp;&nbsp;typename OutputIterator,</span>
<span>&nbsp;&nbsp;typename StrictWeakCompare&gt;</span>
<span>__host__ __device__ OutputIterator </span><span><b>set_symmetric_difference</b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;InputIterator1 first1,</span>
<span>&nbsp;&nbsp;InputIterator1 last1,</span>
<span>&nbsp;&nbsp;InputIterator2 first2,</span>
<span>&nbsp;&nbsp;InputIterator2 last2,</span>
<span>&nbsp;&nbsp;OutputIterator result,</span>
<span>&nbsp;&nbsp;StrictWeakCompare comp);</span></code>
<code>set&#95;symmetric&#95;difference</code> constructs a sorted range that is the set symmetric difference of the sorted ranges <code>[first1, last1)</code> and <code>[first2, last2)</code>. The return value is the end of the output range.

In the simplest case, <code>set&#95;symmetric&#95;difference</code> performs a set theoretic calculation: it constructs the union of the two sets A - B and B - A, where A and B are the two input ranges. That is, the output range contains a copy of every element that is contained in <code>[first1, last1)</code> but not <code>[first2, last1)</code>, and a copy of every element that is contained in <code>[first2, last2)</code> but not <code>[first1, last1)</code>. The general case is more complicated, because the input ranges may contain duplicate elements. The generalization is that if <code>[first1, last1)</code> contains <code>m</code> elements that are equivalent to each other and <code>[first2, last1)</code> contains <code>n</code> elements that are equivalent to them, then <code>|m - n|</code> of those elements shall be copied to the output range: the last <code>m - n</code> elements from <code>[first1, last1)</code> if <code>m &gt; n</code>, and the last <code>n - m</code> of these elements from <code>[first2, last2)</code> if <code>m &lt; n</code>.

This version of <code>set&#95;union</code> compares elements using a function object <code>comp</code>.

The algorithm's execution is parallelized as determined by <code>exec</code>.


The following code snippet demonstrates how to use <code>set&#95;symmetric&#95;difference</code> to compute the symmetric difference of two sets of integers sorted in descending order using the <code>thrust::host</code> execution policy for parallelization:



```cpp
#include <thrust/set_operations.h>
#include <thrust/execution_policy.h>
...
int A1[7] = {7, 6, 4, 2, 2, 1, 0};
int A2[5] = {8, 5, 2, 1, 1};

int result[6];

int *result_end = thrust::set_symmetric_difference(thrust::host, A1, A1 + 7, A2, A2 + 5, result);
// result = {8, 7, 6, 5, 4, 0}
```

**Template Parameters**:
* **`DerivedPolicy`** The name of the derived execution policy. 
* **`InputIterator1`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>, <code>InputIterator1</code> and <code>InputIterator2</code> have the same <code>value&#95;type</code>, <code>InputIterator1's</code><code>value&#95;type</code> is a model of <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a>, the ordering on <code>InputIterator1's</code><code>value&#95;type</code> is a strict weak ordering, as defined in the <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a> requirements, and <code>InputIterator1's</code><code>value&#95;type</code> is convertable to a type in <code>OutputIterator's</code> set of <code>value&#95;types</code>. 
* **`InputIterator2`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>, <code>InputIterator2</code> and <code>InputIterator1</code> have the same <code>value&#95;type</code>, <code>InputIterator2's</code><code>value&#95;type</code> is a model of <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a>, the ordering on <code>InputIterator2's</code><code>value&#95;type</code> is a strict weak ordering, as defined in the <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a> requirements, and <code>InputIterator2's</code><code>value&#95;type</code> is convertable to a type in <code>OutputIterator's</code> set of <code>value&#95;types</code>. 
* **`OutputIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>.

**Function Parameters**:
* **`exec`** The execution policy to use for parallelization. 
* **`first1`** The beginning of the first input range. 
* **`last1`** The end of the first input range. 
* **`first2`** The beginning of the second input range. 
* **`last2`** The end of the second input range. 
* **`result`** The beginning of the output range. 
* **`comp`** Comparison operator. 

**Preconditions**:
* The ranges <code>[first1, last1)</code> and <code>[first2, last2)</code> shall be sorted with respect to <code>comp</code>. 
* The resulting range shall not overlap with either input range.

**Returns**:
The end of the output range.

**See**:
* <a href="https://en.cppreference.com/w/cpp/algorithm/set_symmetric_difference">https://en.cppreference.com/w/cpp/algorithm/set_symmetric_difference</a>
* <code>merge</code>
* <code>includes</code>
* <code>set&#95;difference</code>
* <code>set&#95;union</code>
* <code>set&#95;intersection</code>
* <code>sort</code>
* <code>is&#95;sorted</code>

<h3 id="function-set-symmetric-difference">
Function <code>thrust::set&#95;symmetric&#95;difference</code>
</h3>

<code class="doxybook">
<span>template &lt;typename InputIterator1,</span>
<span>&nbsp;&nbsp;typename InputIterator2,</span>
<span>&nbsp;&nbsp;typename OutputIterator,</span>
<span>&nbsp;&nbsp;typename StrictWeakCompare&gt;</span>
<span>OutputIterator </span><span><b>set_symmetric_difference</b>(InputIterator1 first1,</span>
<span>&nbsp;&nbsp;InputIterator1 last1,</span>
<span>&nbsp;&nbsp;InputIterator2 first2,</span>
<span>&nbsp;&nbsp;InputIterator2 last2,</span>
<span>&nbsp;&nbsp;OutputIterator result,</span>
<span>&nbsp;&nbsp;StrictWeakCompare comp);</span></code>
<code>set&#95;symmetric&#95;difference</code> constructs a sorted range that is the set symmetric difference of the sorted ranges <code>[first1, last1)</code> and <code>[first2, last2)</code>. The return value is the end of the output range.

In the simplest case, <code>set&#95;symmetric&#95;difference</code> performs a set theoretic calculation: it constructs the union of the two sets A - B and B - A, where A and B are the two input ranges. That is, the output range contains a copy of every element that is contained in <code>[first1, last1)</code> but not <code>[first2, last1)</code>, and a copy of every element that is contained in <code>[first2, last2)</code> but not <code>[first1, last1)</code>. The general case is more complicated, because the input ranges may contain duplicate elements. The generalization is that if <code>[first1, last1)</code> contains <code>m</code> elements that are equivalent to each other and <code>[first2, last1)</code> contains <code>n</code> elements that are equivalent to them, then <code>|m - n|</code> of those elements shall be copied to the output range: the last <code>m - n</code> elements from <code>[first1, last1)</code> if <code>m &gt; n</code>, and the last <code>n - m</code> of these elements from <code>[first2, last2)</code> if <code>m &lt; n</code>.

This version of <code>set&#95;union</code> compares elements using a function object <code>comp</code>.


The following code snippet demonstrates how to use <code>set&#95;symmetric&#95;difference</code> to compute the symmetric difference of two sets of integers sorted in descending order.



```cpp
#include <thrust/set_operations.h>
...
int A1[7] = {7, 6, 4, 2, 2, 1, 0};
int A2[5] = {8, 5, 2, 1, 1};

int result[6];

int *result_end = thrust::set_symmetric_difference(A1, A1 + 7, A2, A2 + 5, result);
// result = {8, 7, 6, 5, 4, 0}
```

**Template Parameters**:
* **`InputIterator1`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>, <code>InputIterator1</code> and <code>InputIterator2</code> have the same <code>value&#95;type</code>, <code>InputIterator1's</code><code>value&#95;type</code> is a model of <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a>, the ordering on <code>InputIterator1's</code><code>value&#95;type</code> is a strict weak ordering, as defined in the <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a> requirements, and <code>InputIterator1's</code><code>value&#95;type</code> is convertable to a type in <code>OutputIterator's</code> set of <code>value&#95;types</code>. 
* **`InputIterator2`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>, <code>InputIterator2</code> and <code>InputIterator1</code> have the same <code>value&#95;type</code>, <code>InputIterator2's</code><code>value&#95;type</code> is a model of <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a>, the ordering on <code>InputIterator2's</code><code>value&#95;type</code> is a strict weak ordering, as defined in the <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a> requirements, and <code>InputIterator2's</code><code>value&#95;type</code> is convertable to a type in <code>OutputIterator's</code> set of <code>value&#95;types</code>. 
* **`OutputIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>.

**Function Parameters**:
* **`first1`** The beginning of the first input range. 
* **`last1`** The end of the first input range. 
* **`first2`** The beginning of the second input range. 
* **`last2`** The end of the second input range. 
* **`result`** The beginning of the output range. 
* **`comp`** Comparison operator. 

**Preconditions**:
* The ranges <code>[first1, last1)</code> and <code>[first2, last2)</code> shall be sorted with respect to <code>comp</code>. 
* The resulting range shall not overlap with either input range.

**Returns**:
The end of the output range.

**See**:
* <a href="https://en.cppreference.com/w/cpp/algorithm/set_symmetric_difference">https://en.cppreference.com/w/cpp/algorithm/set_symmetric_difference</a>
* <code>merge</code>
* <code>includes</code>
* <code>set&#95;difference</code>
* <code>set&#95;union</code>
* <code>set&#95;intersection</code>
* <code>sort</code>
* <code>is&#95;sorted</code>

<h3 id="function-set-union">
Function <code>thrust::set&#95;union</code>
</h3>

<code class="doxybook">
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename InputIterator1,</span>
<span>&nbsp;&nbsp;typename InputIterator2,</span>
<span>&nbsp;&nbsp;typename OutputIterator&gt;</span>
<span>__host__ __device__ OutputIterator </span><span><b>set_union</b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;InputIterator1 first1,</span>
<span>&nbsp;&nbsp;InputIterator1 last1,</span>
<span>&nbsp;&nbsp;InputIterator2 first2,</span>
<span>&nbsp;&nbsp;InputIterator2 last2,</span>
<span>&nbsp;&nbsp;OutputIterator result);</span></code>
<code>set&#95;union</code> constructs a sorted range that is the union of the sorted ranges <code>[first1, last1)</code> and <code>[first2, last2)</code>. The return value is the end of the output range.

In the simplest case, <code>set&#95;union</code> performs the "union" operation from set theory: the output range contains a copy of every element that is contained in <code>[first1, last1)</code>, <code>[first2, last1)</code>, or both. The general case is more complicated, because the input ranges may contain duplicate elements. The generalization is that if <code>[first1, last1)</code> contains <code>m</code> elements that are equivalent to each other and if <code>[first2, last2)</code> contains <code>n</code> elements that are equivalent to them, then all <code>m</code> elements from the first range shall be copied to the output range, in order, and then <code>max(n - m, 0)</code> elements from the second range shall be copied to the output, in order.

This version of <code>set&#95;union</code> compares elements using <code>operator&lt;</code>.

The algorithm's execution is parallelized as determined by <code>exec</code>.


The following code snippet demonstrates how to use <code>set&#95;union</code> to compute the union of two sets of integers sorted in ascending order using the <code>thrust::host</code> execution policy for parallelization:



```cpp
#include <thrust/set_operations.h>
#include <thrust/execution_policy.h>
...
int A1[7] = {0, 2, 4, 6, 8, 10, 12};
int A2[5] = {1, 3, 5, 7, 9};

int result[11];

int *result_end = thrust::set_union(thrust::host, A1, A1 + 7, A2, A2 + 5, result);
// result = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12}
```

**Template Parameters**:
* **`DerivedPolicy`** The name of the derived execution policy. 
* **`InputIterator1`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>, <code>InputIterator1</code> and <code>InputIterator2</code> have the same <code>value&#95;type</code>, <code>InputIterator1's</code><code>value&#95;type</code> is a model of <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a>, the ordering on <code>InputIterator1's</code><code>value&#95;type</code> is a strict weak ordering, as defined in the <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a> requirements, and <code>InputIterator1's</code><code>value&#95;type</code> is convertable to a type in <code>OutputIterator's</code> set of <code>value&#95;types</code>. 
* **`InputIterator2`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>, <code>InputIterator2</code> and <code>InputIterator1</code> have the same <code>value&#95;type</code>, <code>InputIterator2's</code><code>value&#95;type</code> is a model of <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a>, the ordering on <code>InputIterator2's</code><code>value&#95;type</code> is a strict weak ordering, as defined in the <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a> requirements, and <code>InputIterator2's</code><code>value&#95;type</code> is convertable to a type in <code>OutputIterator's</code> set of <code>value&#95;types</code>. 
* **`OutputIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>.

**Function Parameters**:
* **`exec`** The execution policy to use for parallelization. 
* **`first1`** The beginning of the first input range. 
* **`last1`** The end of the first input range. 
* **`first2`** The beginning of the second input range. 
* **`last2`** The end of the second input range. 
* **`result`** The beginning of the output range. 

**Preconditions**:
* The ranges <code>[first1, last1)</code> and <code>[first2, last2)</code> shall be sorted with respect to <code>operator&lt;</code>. 
* The resulting range shall not overlap with either input range.

**Returns**:
The end of the output range.

**See**:
* <a href="https://en.cppreference.com/w/cpp/algorithm/set_union">https://en.cppreference.com/w/cpp/algorithm/set_union</a>
* <code>merge</code>
* <code>includes</code>
* <code>set&#95;union</code>
* <code>set&#95;intersection</code>
* <code>set&#95;symmetric&#95;difference</code>
* <code>sort</code>
* <code>is&#95;sorted</code>

<h3 id="function-set-union">
Function <code>thrust::set&#95;union</code>
</h3>

<code class="doxybook">
<span>template &lt;typename InputIterator1,</span>
<span>&nbsp;&nbsp;typename InputIterator2,</span>
<span>&nbsp;&nbsp;typename OutputIterator&gt;</span>
<span>OutputIterator </span><span><b>set_union</b>(InputIterator1 first1,</span>
<span>&nbsp;&nbsp;InputIterator1 last1,</span>
<span>&nbsp;&nbsp;InputIterator2 first2,</span>
<span>&nbsp;&nbsp;InputIterator2 last2,</span>
<span>&nbsp;&nbsp;OutputIterator result);</span></code>
<code>set&#95;union</code> constructs a sorted range that is the union of the sorted ranges <code>[first1, last1)</code> and <code>[first2, last2)</code>. The return value is the end of the output range.

In the simplest case, <code>set&#95;union</code> performs the "union" operation from set theory: the output range contains a copy of every element that is contained in <code>[first1, last1)</code>, <code>[first2, last1)</code>, or both. The general case is more complicated, because the input ranges may contain duplicate elements. The generalization is that if <code>[first1, last1)</code> contains <code>m</code> elements that are equivalent to each other and if <code>[first2, last2)</code> contains <code>n</code> elements that are equivalent to them, then all <code>m</code> elements from the first range shall be copied to the output range, in order, and then <code>max(n - m, 0)</code> elements from the second range shall be copied to the output, in order.

This version of <code>set&#95;union</code> compares elements using <code>operator&lt;</code>.


The following code snippet demonstrates how to use <code>set&#95;union</code> to compute the union of two sets of integers sorted in ascending order.



```cpp
#include <thrust/set_operations.h>
...
int A1[7] = {0, 2, 4, 6, 8, 10, 12};
int A2[5] = {1, 3, 5, 7, 9};

int result[11];

int *result_end = thrust::set_union(A1, A1 + 7, A2, A2 + 5, result);
// result = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12}
```

**Template Parameters**:
* **`InputIterator1`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>, <code>InputIterator1</code> and <code>InputIterator2</code> have the same <code>value&#95;type</code>, <code>InputIterator1's</code><code>value&#95;type</code> is a model of <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a>, the ordering on <code>InputIterator1's</code><code>value&#95;type</code> is a strict weak ordering, as defined in the <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a> requirements, and <code>InputIterator1's</code><code>value&#95;type</code> is convertable to a type in <code>OutputIterator's</code> set of <code>value&#95;types</code>. 
* **`InputIterator2`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>, <code>InputIterator2</code> and <code>InputIterator1</code> have the same <code>value&#95;type</code>, <code>InputIterator2's</code><code>value&#95;type</code> is a model of <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a>, the ordering on <code>InputIterator2's</code><code>value&#95;type</code> is a strict weak ordering, as defined in the <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a> requirements, and <code>InputIterator2's</code><code>value&#95;type</code> is convertable to a type in <code>OutputIterator's</code> set of <code>value&#95;types</code>. 
* **`OutputIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>.

**Function Parameters**:
* **`first1`** The beginning of the first input range. 
* **`last1`** The end of the first input range. 
* **`first2`** The beginning of the second input range. 
* **`last2`** The end of the second input range. 
* **`result`** The beginning of the output range. 

**Preconditions**:
* The ranges <code>[first1, last1)</code> and <code>[first2, last2)</code> shall be sorted with respect to <code>operator&lt;</code>. 
* The resulting range shall not overlap with either input range.

**Returns**:
The end of the output range.

**See**:
* <a href="https://en.cppreference.com/w/cpp/algorithm/set_union">https://en.cppreference.com/w/cpp/algorithm/set_union</a>
* <code>merge</code>
* <code>includes</code>
* <code>set&#95;union</code>
* <code>set&#95;intersection</code>
* <code>set&#95;symmetric&#95;difference</code>
* <code>sort</code>
* <code>is&#95;sorted</code>

<h3 id="function-set-union">
Function <code>thrust::set&#95;union</code>
</h3>

<code class="doxybook">
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename InputIterator1,</span>
<span>&nbsp;&nbsp;typename InputIterator2,</span>
<span>&nbsp;&nbsp;typename OutputIterator,</span>
<span>&nbsp;&nbsp;typename StrictWeakCompare&gt;</span>
<span>__host__ __device__ OutputIterator </span><span><b>set_union</b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;InputIterator1 first1,</span>
<span>&nbsp;&nbsp;InputIterator1 last1,</span>
<span>&nbsp;&nbsp;InputIterator2 first2,</span>
<span>&nbsp;&nbsp;InputIterator2 last2,</span>
<span>&nbsp;&nbsp;OutputIterator result,</span>
<span>&nbsp;&nbsp;StrictWeakCompare comp);</span></code>
<code>set&#95;union</code> constructs a sorted range that is the union of the sorted ranges <code>[first1, last1)</code> and <code>[first2, last2)</code>. The return value is the end of the output range.

In the simplest case, <code>set&#95;union</code> performs the "union" operation from set theory: the output range contains a copy of every element that is contained in <code>[first1, last1)</code>, <code>[first2, last1)</code>, or both. The general case is more complicated, because the input ranges may contain duplicate elements. The generalization is that if <code>[first1, last1)</code> contains <code>m</code> elements that are equivalent to each other and if <code>[first2, last2)</code> contains <code>n</code> elements that are equivalent to them, then all <code>m</code> elements from the first range shall be copied to the output range, in order, and then <code>max(n - m, 0)</code> elements from the second range shall be copied to the output, in order.

This version of <code>set&#95;union</code> compares elements using a function object <code>comp</code>.

The algorithm's execution is parallelized as determined by <code>exec</code>.


The following code snippet demonstrates how to use <code>set&#95;union</code> to compute the union of two sets of integers sorted in ascending order using the <code>thrust::host</code> execution policy for parallelization:



```cpp
#include <thrust/set_operations.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>
...
int A1[7] = {12, 10, 8, 6, 4, 2, 0};
int A2[5] = {9, 7, 5, 3, 1};

int result[11];

int *result_end = thrust::set_union(thrust::host, A1, A1 + 7, A2, A2 + 5, result, thrust::greater<int>());
// result = {12, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0}
```

**Template Parameters**:
* **`DerivedPolicy`** The name of the derived execution policy. 
* **`InputIterator1`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>, <code>InputIterator1's</code><code>value&#95;type</code> is convertable to <code>StrictWeakCompare's</code><code>first&#95;argument&#95;type</code>. and <code>InputIterator1's</code><code>value&#95;type</code> is convertable to a type in <code>OutputIterator's</code> set of <code>value&#95;types</code>. 
* **`InputIterator2`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>, <code>InputIterator2's</code><code>value&#95;type</code> is convertable to <code>StrictWeakCompare's</code><code>second&#95;argument&#95;type</code>. and <code>InputIterator2's</code><code>value&#95;type</code> is convertable to a type in <code>OutputIterator's</code> set of <code>value&#95;types</code>. 
* **`OutputIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>. 
* **`StrictWeakCompare`** is a model of <a href="https://en.cppreference.com/w/cpp/concepts/strict_weak_order">Strict Weak Ordering</a>.

**Function Parameters**:
* **`exec`** The execution policy to use for parallelization. 
* **`first1`** The beginning of the first input range. 
* **`last1`** The end of the first input range. 
* **`first2`** The beginning of the second input range. 
* **`last2`** The end of the second input range. 
* **`result`** The beginning of the output range. 
* **`comp`** Comparison operator. 

**Preconditions**:
* The ranges <code>[first1, last1)</code> and <code>[first2, last2)</code> shall be sorted with respect to <code>comp</code>. 
* The resulting range shall not overlap with either input range.

**Returns**:
The end of the output range.

**See**:
* <a href="https://en.cppreference.com/w/cpp/algorithm/set_union">https://en.cppreference.com/w/cpp/algorithm/set_union</a>
* <code>merge</code>
* <code>includes</code>
* <code>set&#95;union</code>
* <code>set&#95;intersection</code>
* <code>set&#95;symmetric&#95;difference</code>
* <code>sort</code>
* <code>is&#95;sorted</code>

<h3 id="function-set-union">
Function <code>thrust::set&#95;union</code>
</h3>

<code class="doxybook">
<span>template &lt;typename InputIterator1,</span>
<span>&nbsp;&nbsp;typename InputIterator2,</span>
<span>&nbsp;&nbsp;typename OutputIterator,</span>
<span>&nbsp;&nbsp;typename StrictWeakCompare&gt;</span>
<span>OutputIterator </span><span><b>set_union</b>(InputIterator1 first1,</span>
<span>&nbsp;&nbsp;InputIterator1 last1,</span>
<span>&nbsp;&nbsp;InputIterator2 first2,</span>
<span>&nbsp;&nbsp;InputIterator2 last2,</span>
<span>&nbsp;&nbsp;OutputIterator result,</span>
<span>&nbsp;&nbsp;StrictWeakCompare comp);</span></code>
<code>set&#95;union</code> constructs a sorted range that is the union of the sorted ranges <code>[first1, last1)</code> and <code>[first2, last2)</code>. The return value is the end of the output range.

In the simplest case, <code>set&#95;union</code> performs the "union" operation from set theory: the output range contains a copy of every element that is contained in <code>[first1, last1)</code>, <code>[first2, last1)</code>, or both. The general case is more complicated, because the input ranges may contain duplicate elements. The generalization is that if <code>[first1, last1)</code> contains <code>m</code> elements that are equivalent to each other and if <code>[first2, last2)</code> contains <code>n</code> elements that are equivalent to them, then all <code>m</code> elements from the first range shall be copied to the output range, in order, and then <code>max(n - m, 0)</code> elements from the second range shall be copied to the output, in order.

This version of <code>set&#95;union</code> compares elements using a function object <code>comp</code>.


The following code snippet demonstrates how to use <code>set&#95;union</code> to compute the union of two sets of integers sorted in ascending order.



```cpp
#include <thrust/set_operations.h>
#include <thrust/functional.h>
...
int A1[7] = {12, 10, 8, 6, 4, 2, 0};
int A2[5] = {9, 7, 5, 3, 1};

int result[11];

int *result_end = thrust::set_union(A1, A1 + 7, A2, A2 + 5, result, thrust::greater<int>());
// result = {12, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0}
```

**Template Parameters**:
* **`InputIterator1`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>, <code>InputIterator1's</code><code>value&#95;type</code> is convertable to <code>StrictWeakCompare's</code><code>first&#95;argument&#95;type</code>. and <code>InputIterator1's</code><code>value&#95;type</code> is convertable to a type in <code>OutputIterator's</code> set of <code>value&#95;types</code>. 
* **`InputIterator2`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>, <code>InputIterator2's</code><code>value&#95;type</code> is convertable to <code>StrictWeakCompare's</code><code>second&#95;argument&#95;type</code>. and <code>InputIterator2's</code><code>value&#95;type</code> is convertable to a type in <code>OutputIterator's</code> set of <code>value&#95;types</code>. 
* **`OutputIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>. 
* **`StrictWeakCompare`** is a model of <a href="https://en.cppreference.com/w/cpp/concepts/strict_weak_order">Strict Weak Ordering</a>.

**Function Parameters**:
* **`first1`** The beginning of the first input range. 
* **`last1`** The end of the first input range. 
* **`first2`** The beginning of the second input range. 
* **`last2`** The end of the second input range. 
* **`result`** The beginning of the output range. 
* **`comp`** Comparison operator. 

**Preconditions**:
* The ranges <code>[first1, last1)</code> and <code>[first2, last2)</code> shall be sorted with respect to <code>comp</code>. 
* The resulting range shall not overlap with either input range.

**Returns**:
The end of the output range.

**See**:
* <a href="https://en.cppreference.com/w/cpp/algorithm/set_union">https://en.cppreference.com/w/cpp/algorithm/set_union</a>
* <code>merge</code>
* <code>includes</code>
* <code>set&#95;union</code>
* <code>set&#95;intersection</code>
* <code>set&#95;symmetric&#95;difference</code>
* <code>sort</code>
* <code>is&#95;sorted</code>

<h3 id="function-set-difference-by-key">
Function <code>thrust::set&#95;difference&#95;by&#95;key</code>
</h3>

<code class="doxybook">
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename InputIterator1,</span>
<span>&nbsp;&nbsp;typename InputIterator2,</span>
<span>&nbsp;&nbsp;typename InputIterator3,</span>
<span>&nbsp;&nbsp;typename InputIterator4,</span>
<span>&nbsp;&nbsp;typename OutputIterator1,</span>
<span>&nbsp;&nbsp;typename OutputIterator2&gt;</span>
<span>__host__ __device__ <a href="{{ site.baseurl }}/api/classes/structthrust_1_1pair.html">thrust::pair</a>< OutputIterator1, OutputIterator2 > </span><span><b>set_difference_by_key</b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;InputIterator1 keys_first1,</span>
<span>&nbsp;&nbsp;InputIterator1 keys_last1,</span>
<span>&nbsp;&nbsp;InputIterator2 keys_first2,</span>
<span>&nbsp;&nbsp;InputIterator2 keys_last2,</span>
<span>&nbsp;&nbsp;InputIterator3 values_first1,</span>
<span>&nbsp;&nbsp;InputIterator4 values_first2,</span>
<span>&nbsp;&nbsp;OutputIterator1 keys_result,</span>
<span>&nbsp;&nbsp;OutputIterator2 values_result);</span></code>
<code>set&#95;difference&#95;by&#95;key</code> performs a key-value difference operation from set theory. <code>set&#95;difference&#95;by&#95;key</code> constructs a sorted range that is the difference of the sorted ranges <code>[keys&#95;first1, keys&#95;last1)</code> and <code>[keys&#95;first2, keys&#95;last2)</code>. Associated with each element from the input and output key ranges is a value element. The associated input value ranges need not be sorted.

In the simplest case, <code>set&#95;difference&#95;by&#95;key</code> performs the "difference" operation from set theory: the keys output range contains a copy of every element that is contained in <code>[keys&#95;first1, keys&#95;last1)</code> and not contained in <code>[keys&#95;first2, keys&#95;last2)</code>. The general case is more complicated, because the input ranges may contain duplicate elements. The generalization is that if <code>[keys&#95;first1, keys&#95;last1)</code> contains <code>m</code> elements that are equivalent to each other and if <code>[keys&#95;first2, keys&#95;last2)</code> contains <code>n</code> elements that are equivalent to them, the last <code>max(m-n,0)</code> elements from <code>[keys&#95;first1, keys&#95;last1)</code> range shall be copied to the output range.

Each time a key element is copied from <code>[keys&#95;first1, keys&#95;last1)</code> or <code>[keys&#95;first2, keys&#95;last2)</code> is copied to the keys output range, the corresponding value element is copied from the corresponding values input range (beginning at <code>values&#95;first1</code> or <code>values&#95;first2</code>) to the values output range.

This version of <code>set&#95;difference&#95;by&#95;key</code> compares key elements using <code>operator&lt;</code>.

The algorithm's execution is parallelized as determined by <code>exec</code>.


The following code snippet demonstrates how to use <code>set&#95;difference&#95;by&#95;key</code> to compute the set difference of two sets of integers sorted in ascending order with their values using the <code>thrust::host</code> execution policy for parallelization:



```cpp
#include <thrust/set_operations.h>
#include <thrust/execution_policy.h>
...
int A_keys[6] = {0, 1, 3, 4, 5, 6, 9};
int A_vals[6] = {0, 0, 0, 0, 0, 0, 0};

int B_keys[5] = {1, 3, 5, 7, 9};
int B_vals[5] = {1, 1, 1, 1, 1};

int keys_result[3];
int vals_result[3];

thrust::pair<int*,int*> end = thrust::set_difference_by_key(thrust::host, A_keys, A_keys + 6, B_keys, B_keys + 5, A_vals, B_vals, keys_result, vals_result);
// keys_result is now {0, 4, 6}
// vals_result is now {0, 0, 0}
```

**Template Parameters**:
* **`DerivedPolicy`** The name of the derived execution policy. 
* **`InputIterator1`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>, <code>InputIterator1</code> and <code>InputIterator2</code> have the same <code>value&#95;type</code>, <code>InputIterator1's</code><code>value&#95;type</code> is a model of <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a>, the ordering on <code>InputIterator1's</code><code>value&#95;type</code> is a strict weak ordering, as defined in the <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a> requirements, and <code>InputIterator1's</code><code>value&#95;type</code> is convertable to a type in <code>OutputIterator's</code> set of <code>value&#95;types</code>. 
* **`InputIterator2`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>, <code>InputIterator2</code> and <code>InputIterator1</code> have the same <code>value&#95;type</code>, <code>InputIterator2's</code><code>value&#95;type</code> is a model of <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a>, the ordering on <code>InputIterator2's</code><code>value&#95;type</code> is a strict weak ordering, as defined in the <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a> requirements, and <code>InputIterator2's</code><code>value&#95;type</code> is convertable to a type in <code>OutputIterator's</code> set of <code>value&#95;types</code>. 
* **`InputIterator3`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>, and <code>InputIterator3's</code><code>value&#95;type</code> is convertible to a type in <code>OutputIterator2's</code> set of <code>value&#95;types</code>. 
* **`InputIterator4`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>, and <code>InputIterator4's</code><code>value&#95;type</code> is convertible to a type in <code>OutputIterator2's</code> set of <code>value&#95;types</code>. 
* **`OutputIterator1`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>. 
* **`OutputIterator2`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>.

**Function Parameters**:
* **`exec`** The execution policy to use for parallelization. 
* **`keys_first1`** The beginning of the first input range of keys. 
* **`keys_last1`** The end of the first input range of keys. 
* **`keys_first2`** The beginning of the second input range of keys. 
* **`keys_last2`** The end of the second input range of keys. 
* **`values_first1`** The beginning of the first input range of values. 
* **`values_first2`** The beginning of the first input range of values. 
* **`keys_result`** The beginning of the output range of keys. 
* **`values_result`** The beginning of the output range of values. 

**Preconditions**:
* The ranges <code>[keys&#95;first1, keys&#95;last1)</code> and <code>[keys&#95;first2, keys&#95;last2)</code> shall be sorted with respect to <code>operator&lt;</code>. 
* The resulting ranges shall not overlap with any input range.

**Returns**:
A <code>pair</code><code>p</code> such that <code>p.first</code> is the end of the output range of keys, and such that <code>p.second</code> is the end of the output range of values.

**See**:
* <code>set&#95;union&#95;by&#95;key</code>
* <code>set&#95;intersection&#95;by&#95;key</code>
* <code>set&#95;symmetric&#95;difference&#95;by&#95;key</code>
* <code>sort&#95;by&#95;key</code>
* <code>is&#95;sorted</code>

<h3 id="function-set-difference-by-key">
Function <code>thrust::set&#95;difference&#95;by&#95;key</code>
</h3>

<code class="doxybook">
<span>template &lt;typename InputIterator1,</span>
<span>&nbsp;&nbsp;typename InputIterator2,</span>
<span>&nbsp;&nbsp;typename InputIterator3,</span>
<span>&nbsp;&nbsp;typename InputIterator4,</span>
<span>&nbsp;&nbsp;typename OutputIterator1,</span>
<span>&nbsp;&nbsp;typename OutputIterator2&gt;</span>
<span><a href="{{ site.baseurl }}/api/classes/structthrust_1_1pair.html">thrust::pair</a>< OutputIterator1, OutputIterator2 > </span><span><b>set_difference_by_key</b>(InputIterator1 keys_first1,</span>
<span>&nbsp;&nbsp;InputIterator1 keys_last1,</span>
<span>&nbsp;&nbsp;InputIterator2 keys_first2,</span>
<span>&nbsp;&nbsp;InputIterator2 keys_last2,</span>
<span>&nbsp;&nbsp;InputIterator3 values_first1,</span>
<span>&nbsp;&nbsp;InputIterator4 values_first2,</span>
<span>&nbsp;&nbsp;OutputIterator1 keys_result,</span>
<span>&nbsp;&nbsp;OutputIterator2 values_result);</span></code>
<code>set&#95;difference&#95;by&#95;key</code> performs a key-value difference operation from set theory. <code>set&#95;difference&#95;by&#95;key</code> constructs a sorted range that is the difference of the sorted ranges <code>[keys&#95;first1, keys&#95;last1)</code> and <code>[keys&#95;first2, keys&#95;last2)</code>. Associated with each element from the input and output key ranges is a value element. The associated input value ranges need not be sorted.

In the simplest case, <code>set&#95;difference&#95;by&#95;key</code> performs the "difference" operation from set theory: the keys output range contains a copy of every element that is contained in <code>[keys&#95;first1, keys&#95;last1)</code> and not contained in <code>[keys&#95;first2, keys&#95;last2)</code>. The general case is more complicated, because the input ranges may contain duplicate elements. The generalization is that if <code>[keys&#95;first1, keys&#95;last1)</code> contains <code>m</code> elements that are equivalent to each other and if <code>[keys&#95;first2, keys&#95;last2)</code> contains <code>n</code> elements that are equivalent to them, the last <code>max(m-n,0)</code> elements from <code>[keys&#95;first1, keys&#95;last1)</code> range shall be copied to the output range.

Each time a key element is copied from <code>[keys&#95;first1, keys&#95;last1)</code> or <code>[keys&#95;first2, keys&#95;last2)</code> is copied to the keys output range, the corresponding value element is copied from the corresponding values input range (beginning at <code>values&#95;first1</code> or <code>values&#95;first2</code>) to the values output range.

This version of <code>set&#95;difference&#95;by&#95;key</code> compares key elements using <code>operator&lt;</code>.


The following code snippet demonstrates how to use <code>set&#95;difference&#95;by&#95;key</code> to compute the set difference of two sets of integers sorted in ascending order with their values.



```cpp
#include <thrust/set_operations.h>
...
int A_keys[6] = {0, 1, 3, 4, 5, 6, 9};
int A_vals[6] = {0, 0, 0, 0, 0, 0, 0};

int B_keys[5] = {1, 3, 5, 7, 9};
int B_vals[5] = {1, 1, 1, 1, 1};

int keys_result[3];
int vals_result[3];

thrust::pair<int*,int*> end = thrust::set_difference_by_key(A_keys, A_keys + 6, B_keys, B_keys + 5, A_vals, B_vals, keys_result, vals_result);
// keys_result is now {0, 4, 6}
// vals_result is now {0, 0, 0}
```

**Template Parameters**:
* **`InputIterator1`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>, <code>InputIterator1</code> and <code>InputIterator2</code> have the same <code>value&#95;type</code>, <code>InputIterator1's</code><code>value&#95;type</code> is a model of <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a>, the ordering on <code>InputIterator1's</code><code>value&#95;type</code> is a strict weak ordering, as defined in the <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a> requirements, and <code>InputIterator1's</code><code>value&#95;type</code> is convertable to a type in <code>OutputIterator's</code> set of <code>value&#95;types</code>. 
* **`InputIterator2`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>, <code>InputIterator2</code> and <code>InputIterator1</code> have the same <code>value&#95;type</code>, <code>InputIterator2's</code><code>value&#95;type</code> is a model of <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a>, the ordering on <code>InputIterator2's</code><code>value&#95;type</code> is a strict weak ordering, as defined in the <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a> requirements, and <code>InputIterator2's</code><code>value&#95;type</code> is convertable to a type in <code>OutputIterator's</code> set of <code>value&#95;types</code>. 
* **`InputIterator3`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>, and <code>InputIterator3's</code><code>value&#95;type</code> is convertible to a type in <code>OutputIterator2's</code> set of <code>value&#95;types</code>. 
* **`InputIterator4`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>, and <code>InputIterator4's</code><code>value&#95;type</code> is convertible to a type in <code>OutputIterator2's</code> set of <code>value&#95;types</code>. 
* **`OutputIterator1`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>. 
* **`OutputIterator2`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>.

**Function Parameters**:
* **`keys_first1`** The beginning of the first input range of keys. 
* **`keys_last1`** The end of the first input range of keys. 
* **`keys_first2`** The beginning of the second input range of keys. 
* **`keys_last2`** The end of the second input range of keys. 
* **`values_first1`** The beginning of the first input range of values. 
* **`values_first2`** The beginning of the first input range of values. 
* **`keys_result`** The beginning of the output range of keys. 
* **`values_result`** The beginning of the output range of values. 

**Preconditions**:
* The ranges <code>[keys&#95;first1, keys&#95;last1)</code> and <code>[keys&#95;first2, keys&#95;last2)</code> shall be sorted with respect to <code>operator&lt;</code>. 
* The resulting ranges shall not overlap with any input range.

**Returns**:
A <code>pair</code><code>p</code> such that <code>p.first</code> is the end of the output range of keys, and such that <code>p.second</code> is the end of the output range of values.

**See**:
* <code>set&#95;union&#95;by&#95;key</code>
* <code>set&#95;intersection&#95;by&#95;key</code>
* <code>set&#95;symmetric&#95;difference&#95;by&#95;key</code>
* <code>sort&#95;by&#95;key</code>
* <code>is&#95;sorted</code>

<h3 id="function-set-difference-by-key">
Function <code>thrust::set&#95;difference&#95;by&#95;key</code>
</h3>

<code class="doxybook">
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename InputIterator1,</span>
<span>&nbsp;&nbsp;typename InputIterator2,</span>
<span>&nbsp;&nbsp;typename InputIterator3,</span>
<span>&nbsp;&nbsp;typename InputIterator4,</span>
<span>&nbsp;&nbsp;typename OutputIterator1,</span>
<span>&nbsp;&nbsp;typename OutputIterator2,</span>
<span>&nbsp;&nbsp;typename StrictWeakCompare&gt;</span>
<span>__host__ __device__ <a href="{{ site.baseurl }}/api/classes/structthrust_1_1pair.html">thrust::pair</a>< OutputIterator1, OutputIterator2 > </span><span><b>set_difference_by_key</b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;InputIterator1 keys_first1,</span>
<span>&nbsp;&nbsp;InputIterator1 keys_last1,</span>
<span>&nbsp;&nbsp;InputIterator2 keys_first2,</span>
<span>&nbsp;&nbsp;InputIterator2 keys_last2,</span>
<span>&nbsp;&nbsp;InputIterator3 values_first1,</span>
<span>&nbsp;&nbsp;InputIterator4 values_first2,</span>
<span>&nbsp;&nbsp;OutputIterator1 keys_result,</span>
<span>&nbsp;&nbsp;OutputIterator2 values_result,</span>
<span>&nbsp;&nbsp;StrictWeakCompare comp);</span></code>
<code>set&#95;difference&#95;by&#95;key</code> performs a key-value difference operation from set theory. <code>set&#95;difference&#95;by&#95;key</code> constructs a sorted range that is the difference of the sorted ranges <code>[keys&#95;first1, keys&#95;last1)</code> and <code>[keys&#95;first2, keys&#95;last2)</code>. Associated with each element from the input and output key ranges is a value element. The associated input value ranges need not be sorted.

In the simplest case, <code>set&#95;difference&#95;by&#95;key</code> performs the "difference" operation from set theory: the keys output range contains a copy of every element that is contained in <code>[keys&#95;first1, keys&#95;last1)</code> and not contained in <code>[keys&#95;first2, keys&#95;last2)</code>. The general case is more complicated, because the input ranges may contain duplicate elements. The generalization is that if <code>[keys&#95;first1, keys&#95;last1)</code> contains <code>m</code> elements that are equivalent to each other and if <code>[keys&#95;first2, keys&#95;last2)</code> contains <code>n</code> elements that are equivalent to them, the last <code>max(m-n,0)</code> elements from <code>[keys&#95;first1, keys&#95;last1)</code> range shall be copied to the output range.

Each time a key element is copied from <code>[keys&#95;first1, keys&#95;last1)</code> or <code>[keys&#95;first2, keys&#95;last2)</code> is copied to the keys output range, the corresponding value element is copied from the corresponding values input range (beginning at <code>values&#95;first1</code> or <code>values&#95;first2</code>) to the values output range.

This version of <code>set&#95;difference&#95;by&#95;key</code> compares key elements using a function object <code>comp</code>.

The algorithm's execution is parallelized as determined by <code>exec</code>.


The following code snippet demonstrates how to use <code>set&#95;difference&#95;by&#95;key</code> to compute the set difference of two sets of integers sorted in descending order with their values using the <code>thrust::host</code> execution policy for parallelization:



```cpp
#include <thrust/set_operations.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>
...
int A_keys[6] = {9, 6, 5, 4, 3, 1, 0};
int A_vals[6] = {0, 0, 0, 0, 0, 0, 0};

int B_keys[5] = {9, 7, 5, 3, 1};
int B_vals[5] = {1, 1, 1, 1, 1};

int keys_result[3];
int vals_result[3];

thrust::pair<int*,int*> end = thrust::set_difference_by_key(thrust::host, A_keys, A_keys + 6, B_keys, B_keys + 5, A_vals, B_vals, keys_result, vals_result, thrust::greater<int>());
// keys_result is now {0, 4, 6}
// vals_result is now {0, 0, 0}
```

**Template Parameters**:
* **`DerivedPolicy`** The name of the derived execution policy. 
* **`InputIterator1`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>, <code>InputIterator1</code> and <code>InputIterator2</code> have the same <code>value&#95;type</code>, <code>InputIterator1's</code><code>value&#95;type</code> is a model of <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a>, the ordering on <code>InputIterator1's</code><code>value&#95;type</code> is a strict weak ordering, as defined in the <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a> requirements, and <code>InputIterator1's</code><code>value&#95;type</code> is convertable to a type in <code>OutputIterator's</code> set of <code>value&#95;types</code>. 
* **`InputIterator2`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>, <code>InputIterator2</code> and <code>InputIterator1</code> have the same <code>value&#95;type</code>, <code>InputIterator2's</code><code>value&#95;type</code> is a model of <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a>, the ordering on <code>InputIterator2's</code><code>value&#95;type</code> is a strict weak ordering, as defined in the <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a> requirements, and <code>InputIterator2's</code><code>value&#95;type</code> is convertable to a type in <code>OutputIterator's</code> set of <code>value&#95;types</code>. 
* **`InputIterator3`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>, and <code>InputIterator3's</code><code>value&#95;type</code> is convertible to a type in <code>OutputIterator2's</code> set of <code>value&#95;types</code>. 
* **`InputIterator4`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>, and <code>InputIterator4's</code><code>value&#95;type</code> is convertible to a type in <code>OutputIterator2's</code> set of <code>value&#95;types</code>. 
* **`OutputIterator1`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>. 
* **`OutputIterator2`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>. 
* **`StrictWeakCompare`** is a model of <a href="https://en.cppreference.com/w/cpp/concepts/strict_weak_order">Strict Weak Ordering</a>.

**Function Parameters**:
* **`exec`** The execution policy to use for parallelization. 
* **`keys_first1`** The beginning of the first input range of keys. 
* **`keys_last1`** The end of the first input range of keys. 
* **`keys_first2`** The beginning of the second input range of keys. 
* **`keys_last2`** The end of the second input range of keys. 
* **`values_first1`** The beginning of the first input range of values. 
* **`values_first2`** The beginning of the first input range of values. 
* **`keys_result`** The beginning of the output range of keys. 
* **`values_result`** The beginning of the output range of values. 
* **`comp`** Comparison operator. 

**Preconditions**:
* The ranges <code>[keys&#95;first1, keys&#95;last1)</code> and <code>[keys&#95;first2, keys&#95;last2)</code> shall be sorted with respect to <code>comp</code>. 
* The resulting ranges shall not overlap with any input range.

**Returns**:
A <code>pair</code><code>p</code> such that <code>p.first</code> is the end of the output range of keys, and such that <code>p.second</code> is the end of the output range of values.

**See**:
* <code>set&#95;union&#95;by&#95;key</code>
* <code>set&#95;intersection&#95;by&#95;key</code>
* <code>set&#95;symmetric&#95;difference&#95;by&#95;key</code>
* <code>sort&#95;by&#95;key</code>
* <code>is&#95;sorted</code>

<h3 id="function-set-difference-by-key">
Function <code>thrust::set&#95;difference&#95;by&#95;key</code>
</h3>

<code class="doxybook">
<span>template &lt;typename InputIterator1,</span>
<span>&nbsp;&nbsp;typename InputIterator2,</span>
<span>&nbsp;&nbsp;typename InputIterator3,</span>
<span>&nbsp;&nbsp;typename InputIterator4,</span>
<span>&nbsp;&nbsp;typename OutputIterator1,</span>
<span>&nbsp;&nbsp;typename OutputIterator2,</span>
<span>&nbsp;&nbsp;typename StrictWeakCompare&gt;</span>
<span><a href="{{ site.baseurl }}/api/classes/structthrust_1_1pair.html">thrust::pair</a>< OutputIterator1, OutputIterator2 > </span><span><b>set_difference_by_key</b>(InputIterator1 keys_first1,</span>
<span>&nbsp;&nbsp;InputIterator1 keys_last1,</span>
<span>&nbsp;&nbsp;InputIterator2 keys_first2,</span>
<span>&nbsp;&nbsp;InputIterator2 keys_last2,</span>
<span>&nbsp;&nbsp;InputIterator3 values_first1,</span>
<span>&nbsp;&nbsp;InputIterator4 values_first2,</span>
<span>&nbsp;&nbsp;OutputIterator1 keys_result,</span>
<span>&nbsp;&nbsp;OutputIterator2 values_result,</span>
<span>&nbsp;&nbsp;StrictWeakCompare comp);</span></code>
<code>set&#95;difference&#95;by&#95;key</code> performs a key-value difference operation from set theory. <code>set&#95;difference&#95;by&#95;key</code> constructs a sorted range that is the difference of the sorted ranges <code>[keys&#95;first1, keys&#95;last1)</code> and <code>[keys&#95;first2, keys&#95;last2)</code>. Associated with each element from the input and output key ranges is a value element. The associated input value ranges need not be sorted.

In the simplest case, <code>set&#95;difference&#95;by&#95;key</code> performs the "difference" operation from set theory: the keys output range contains a copy of every element that is contained in <code>[keys&#95;first1, keys&#95;last1)</code> and not contained in <code>[keys&#95;first2, keys&#95;last2)</code>. The general case is more complicated, because the input ranges may contain duplicate elements. The generalization is that if <code>[keys&#95;first1, keys&#95;last1)</code> contains <code>m</code> elements that are equivalent to each other and if <code>[keys&#95;first2, keys&#95;last2)</code> contains <code>n</code> elements that are equivalent to them, the last <code>max(m-n,0)</code> elements from <code>[keys&#95;first1, keys&#95;last1)</code> range shall be copied to the output range.

Each time a key element is copied from <code>[keys&#95;first1, keys&#95;last1)</code> or <code>[keys&#95;first2, keys&#95;last2)</code> is copied to the keys output range, the corresponding value element is copied from the corresponding values input range (beginning at <code>values&#95;first1</code> or <code>values&#95;first2</code>) to the values output range.

This version of <code>set&#95;difference&#95;by&#95;key</code> compares key elements using a function object <code>comp</code>.


The following code snippet demonstrates how to use <code>set&#95;difference&#95;by&#95;key</code> to compute the set difference of two sets of integers sorted in descending order with their values.



```cpp
#include <thrust/set_operations.h>
#include <thrust/functional.h>
...
int A_keys[6] = {9, 6, 5, 4, 3, 1, 0};
int A_vals[6] = {0, 0, 0, 0, 0, 0, 0};

int B_keys[5] = {9, 7, 5, 3, 1};
int B_vals[5] = {1, 1, 1, 1, 1};

int keys_result[3];
int vals_result[3];

thrust::pair<int*,int*> end = thrust::set_difference_by_key(A_keys, A_keys + 6, B_keys, B_keys + 5, A_vals, B_vals, keys_result, vals_result, thrust::greater<int>());
// keys_result is now {0, 4, 6}
// vals_result is now {0, 0, 0}
```

**Template Parameters**:
* **`InputIterator1`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>, <code>InputIterator1</code> and <code>InputIterator2</code> have the same <code>value&#95;type</code>, <code>InputIterator1's</code><code>value&#95;type</code> is a model of <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a>, the ordering on <code>InputIterator1's</code><code>value&#95;type</code> is a strict weak ordering, as defined in the <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a> requirements, and <code>InputIterator1's</code><code>value&#95;type</code> is convertable to a type in <code>OutputIterator's</code> set of <code>value&#95;types</code>. 
* **`InputIterator2`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>, <code>InputIterator2</code> and <code>InputIterator1</code> have the same <code>value&#95;type</code>, <code>InputIterator2's</code><code>value&#95;type</code> is a model of <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a>, the ordering on <code>InputIterator2's</code><code>value&#95;type</code> is a strict weak ordering, as defined in the <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a> requirements, and <code>InputIterator2's</code><code>value&#95;type</code> is convertable to a type in <code>OutputIterator's</code> set of <code>value&#95;types</code>. 
* **`InputIterator3`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>, and <code>InputIterator3's</code><code>value&#95;type</code> is convertible to a type in <code>OutputIterator2's</code> set of <code>value&#95;types</code>. 
* **`InputIterator4`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>, and <code>InputIterator4's</code><code>value&#95;type</code> is convertible to a type in <code>OutputIterator2's</code> set of <code>value&#95;types</code>. 
* **`OutputIterator1`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>. 
* **`OutputIterator2`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>. 
* **`StrictWeakCompare`** is a model of <a href="https://en.cppreference.com/w/cpp/concepts/strict_weak_order">Strict Weak Ordering</a>.

**Function Parameters**:
* **`keys_first1`** The beginning of the first input range of keys. 
* **`keys_last1`** The end of the first input range of keys. 
* **`keys_first2`** The beginning of the second input range of keys. 
* **`keys_last2`** The end of the second input range of keys. 
* **`values_first1`** The beginning of the first input range of values. 
* **`values_first2`** The beginning of the first input range of values. 
* **`keys_result`** The beginning of the output range of keys. 
* **`values_result`** The beginning of the output range of values. 
* **`comp`** Comparison operator. 

**Preconditions**:
* The ranges <code>[keys&#95;first1, keys&#95;last1)</code> and <code>[keys&#95;first2, keys&#95;last2)</code> shall be sorted with respect to <code>comp</code>. 
* The resulting ranges shall not overlap with any input range.

**Returns**:
A <code>pair</code><code>p</code> such that <code>p.first</code> is the end of the output range of keys, and such that <code>p.second</code> is the end of the output range of values.

**See**:
* <code>set&#95;union&#95;by&#95;key</code>
* <code>set&#95;intersection&#95;by&#95;key</code>
* <code>set&#95;symmetric&#95;difference&#95;by&#95;key</code>
* <code>sort&#95;by&#95;key</code>
* <code>is&#95;sorted</code>

<h3 id="function-set-intersection-by-key">
Function <code>thrust::set&#95;intersection&#95;by&#95;key</code>
</h3>

<code class="doxybook">
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename InputIterator1,</span>
<span>&nbsp;&nbsp;typename InputIterator2,</span>
<span>&nbsp;&nbsp;typename InputIterator3,</span>
<span>&nbsp;&nbsp;typename OutputIterator1,</span>
<span>&nbsp;&nbsp;typename OutputIterator2&gt;</span>
<span>__host__ __device__ <a href="{{ site.baseurl }}/api/classes/structthrust_1_1pair.html">thrust::pair</a>< OutputIterator1, OutputIterator2 > </span><span><b>set_intersection_by_key</b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;InputIterator1 keys_first1,</span>
<span>&nbsp;&nbsp;InputIterator1 keys_last1,</span>
<span>&nbsp;&nbsp;InputIterator2 keys_first2,</span>
<span>&nbsp;&nbsp;InputIterator2 keys_last2,</span>
<span>&nbsp;&nbsp;InputIterator3 values_first1,</span>
<span>&nbsp;&nbsp;OutputIterator1 keys_result,</span>
<span>&nbsp;&nbsp;OutputIterator2 values_result);</span></code>
<code>set&#95;intersection&#95;by&#95;key</code> performs a key-value intersection operation from set theory. <code>set&#95;intersection&#95;by&#95;key</code> constructs a sorted range that is the intersection of the sorted ranges <code>[keys&#95;first1, keys&#95;last1)</code> and <code>[keys&#95;first2, keys&#95;last2)</code>. Associated with each element from the input and output key ranges is a value element. The associated input value ranges need not be sorted.

In the simplest case, <code>set&#95;intersection&#95;by&#95;key</code> performs the "intersection" operation from set theory: the keys output range contains a copy of every element that is contained in both <code>[keys&#95;first1, keys&#95;last1)</code><code>[keys&#95;first2, keys&#95;last2)</code>. The general case is more complicated, because the input ranges may contain duplicate elements. The generalization is that if an element appears <code>m</code> times in <code>[keys&#95;first1, keys&#95;last1)</code> and <code>n</code> times in <code>[keys&#95;first2, keys&#95;last2)</code> (where <code>m</code> may be zero), then it appears <code>min(m,n)</code> times in the keys output range. <code>set&#95;intersection&#95;by&#95;key</code> is stable, meaning both that elements are copied from the first input range rather than the second, and that the relative order of elements in the output range is the same as the first input range.

Each time a key element is copied from <code>[keys&#95;first1, keys&#95;last1)</code> to the keys output range, the corresponding value element is copied from <code>[values&#95;first1, values&#95;last1)</code> to the values output range.

This version of <code>set&#95;intersection&#95;by&#95;key</code> compares objects using <code>operator&lt;</code>.

The algorithm's execution is parallelized as determined by <code>exec</code>.


The following code snippet demonstrates how to use <code>set&#95;intersection&#95;by&#95;key</code> to compute the set intersection of two sets of integers sorted in ascending order with their values using the <code>thrust::host</code> execution policy for parallelization:



```cpp
#include <thrust/set_operations.h>
#include <thrust/execution_policy.h>
...
int A_keys[6] = {1, 3, 5, 7, 9, 11};
int A_vals[6] = {0, 0, 0, 0, 0,  0};

int B_keys[7] = {1, 1, 2, 3, 5,  8, 13};

int keys_result[7];
int vals_result[7];

thrust::pair<int*,int*> end = thrust::set_intersection_by_key(thrust::host, A_keys, A_keys + 6, B_keys, B_keys + 7, A_vals, keys_result, vals_result);

// keys_result is now {1, 3, 5}
// vals_result is now {0, 0, 0}
```

**Note**:
Unlike the other key-value set operations, <code>set&#95;intersection&#95;by&#95;key</code> is unique in that it has no <code>values&#95;first2</code> parameter because elements from the second input range are never copied to the output range.

**Template Parameters**:
* **`DerivedPolicy`** The name of the derived execution policy. 
* **`InputIterator1`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>, <code>InputIterator1</code> and <code>InputIterator2</code> have the same <code>value&#95;type</code>, <code>InputIterator1's</code><code>value&#95;type</code> is a model of <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a>, the ordering on <code>InputIterator1's</code><code>value&#95;type</code> is a strict weak ordering, as defined in the <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a> requirements, and <code>InputIterator1's</code><code>value&#95;type</code> is convertable to a type in <code>OutputIterator's</code> set of <code>value&#95;types</code>. 
* **`InputIterator2`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>, <code>InputIterator2</code> and <code>InputIterator1</code> have the same <code>value&#95;type</code>, <code>InputIterator2's</code><code>value&#95;type</code> is a model of <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a>, the ordering on <code>InputIterator2's</code><code>value&#95;type</code> is a strict weak ordering, as defined in the <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a> requirements, and <code>InputIterator2's</code><code>value&#95;type</code> is convertable to a type in <code>OutputIterator's</code> set of <code>value&#95;types</code>. 
* **`InputIterator3`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>, and <code>InputIterator3's</code><code>value&#95;type</code> is convertible to a type in <code>OutputIterator2's</code> set of <code>value&#95;types</code>. 
* **`OutputIterator1`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>. 
* **`OutputIterator2`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>.

**Function Parameters**:
* **`exec`** The execution policy to use for parallelization. 
* **`keys_first1`** The beginning of the first input range of keys. 
* **`keys_last1`** The end of the first input range of keys. 
* **`keys_first2`** The beginning of the second input range of keys. 
* **`keys_last2`** The end of the second input range of keys. 
* **`values_first1`** The beginning of the first input range of values. 
* **`keys_result`** The beginning of the output range of keys. 
* **`values_result`** The beginning of the output range of values. 

**Preconditions**:
* The ranges <code>[keys&#95;first1, keys&#95;last1)</code> and <code>[keys&#95;first2, keys&#95;last2)</code> shall be sorted with respect to <code>operator&lt;</code>. 
* The resulting ranges shall not overlap with any input range.

**Returns**:
A <code>pair</code><code>p</code> such that <code>p.first</code> is the end of the output range of keys, and such that <code>p.second</code> is the end of the output range of values.

**See**:
* <code>set&#95;union&#95;by&#95;key</code>
* <code>set&#95;difference&#95;by&#95;key</code>
* <code>set&#95;symmetric&#95;difference&#95;by&#95;key</code>
* <code>sort&#95;by&#95;key</code>
* <code>is&#95;sorted</code>

<h3 id="function-set-intersection-by-key">
Function <code>thrust::set&#95;intersection&#95;by&#95;key</code>
</h3>

<code class="doxybook">
<span>template &lt;typename InputIterator1,</span>
<span>&nbsp;&nbsp;typename InputIterator2,</span>
<span>&nbsp;&nbsp;typename InputIterator3,</span>
<span>&nbsp;&nbsp;typename OutputIterator1,</span>
<span>&nbsp;&nbsp;typename OutputIterator2&gt;</span>
<span><a href="{{ site.baseurl }}/api/classes/structthrust_1_1pair.html">thrust::pair</a>< OutputIterator1, OutputIterator2 > </span><span><b>set_intersection_by_key</b>(InputIterator1 keys_first1,</span>
<span>&nbsp;&nbsp;InputIterator1 keys_last1,</span>
<span>&nbsp;&nbsp;InputIterator2 keys_first2,</span>
<span>&nbsp;&nbsp;InputIterator2 keys_last2,</span>
<span>&nbsp;&nbsp;InputIterator3 values_first1,</span>
<span>&nbsp;&nbsp;OutputIterator1 keys_result,</span>
<span>&nbsp;&nbsp;OutputIterator2 values_result);</span></code>
<code>set&#95;intersection&#95;by&#95;key</code> performs a key-value intersection operation from set theory. <code>set&#95;intersection&#95;by&#95;key</code> constructs a sorted range that is the intersection of the sorted ranges <code>[keys&#95;first1, keys&#95;last1)</code> and <code>[keys&#95;first2, keys&#95;last2)</code>. Associated with each element from the input and output key ranges is a value element. The associated input value ranges need not be sorted.

In the simplest case, <code>set&#95;intersection&#95;by&#95;key</code> performs the "intersection" operation from set theory: the keys output range contains a copy of every element that is contained in both <code>[keys&#95;first1, keys&#95;last1)</code><code>[keys&#95;first2, keys&#95;last2)</code>. The general case is more complicated, because the input ranges may contain duplicate elements. The generalization is that if an element appears <code>m</code> times in <code>[keys&#95;first1, keys&#95;last1)</code> and <code>n</code> times in <code>[keys&#95;first2, keys&#95;last2)</code> (where <code>m</code> may be zero), then it appears <code>min(m,n)</code> times in the keys output range. <code>set&#95;intersection&#95;by&#95;key</code> is stable, meaning both that elements are copied from the first input range rather than the second, and that the relative order of elements in the output range is the same as the first input range.

Each time a key element is copied from <code>[keys&#95;first1, keys&#95;last1)</code> to the keys output range, the corresponding value element is copied from <code>[values&#95;first1, values&#95;last1)</code> to the values output range.

This version of <code>set&#95;intersection&#95;by&#95;key</code> compares objects using <code>operator&lt;</code>.


The following code snippet demonstrates how to use <code>set&#95;intersection&#95;by&#95;key</code> to compute the set intersection of two sets of integers sorted in ascending order with their values.



```cpp
#include <thrust/set_operations.h>
...
int A_keys[6] = {1, 3, 5, 7, 9, 11};
int A_vals[6] = {0, 0, 0, 0, 0,  0};

int B_keys[7] = {1, 1, 2, 3, 5,  8, 13};

int keys_result[7];
int vals_result[7];

thrust::pair<int*,int*> end = thrust::set_intersection_by_key(A_keys, A_keys + 6, B_keys, B_keys + 7, A_vals, keys_result, vals_result);

// keys_result is now {1, 3, 5}
// vals_result is now {0, 0, 0}
```

**Note**:
Unlike the other key-value set operations, <code>set&#95;intersection&#95;by&#95;key</code> is unique in that it has no <code>values&#95;first2</code> parameter because elements from the second input range are never copied to the output range.

**Template Parameters**:
* **`InputIterator1`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>, <code>InputIterator1</code> and <code>InputIterator2</code> have the same <code>value&#95;type</code>, <code>InputIterator1's</code><code>value&#95;type</code> is a model of <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a>, the ordering on <code>InputIterator1's</code><code>value&#95;type</code> is a strict weak ordering, as defined in the <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a> requirements, and <code>InputIterator1's</code><code>value&#95;type</code> is convertable to a type in <code>OutputIterator's</code> set of <code>value&#95;types</code>. 
* **`InputIterator2`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>, <code>InputIterator2</code> and <code>InputIterator1</code> have the same <code>value&#95;type</code>, <code>InputIterator2's</code><code>value&#95;type</code> is a model of <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a>, the ordering on <code>InputIterator2's</code><code>value&#95;type</code> is a strict weak ordering, as defined in the <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a> requirements, and <code>InputIterator2's</code><code>value&#95;type</code> is convertable to a type in <code>OutputIterator's</code> set of <code>value&#95;types</code>. 
* **`InputIterator3`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>, and <code>InputIterator3's</code><code>value&#95;type</code> is convertible to a type in <code>OutputIterator2's</code> set of <code>value&#95;types</code>. 
* **`OutputIterator1`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>. 
* **`OutputIterator2`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>.

**Function Parameters**:
* **`keys_first1`** The beginning of the first input range of keys. 
* **`keys_last1`** The end of the first input range of keys. 
* **`keys_first2`** The beginning of the second input range of keys. 
* **`keys_last2`** The end of the second input range of keys. 
* **`values_first1`** The beginning of the first input range of values. 
* **`keys_result`** The beginning of the output range of keys. 
* **`values_result`** The beginning of the output range of values. 

**Preconditions**:
* The ranges <code>[keys&#95;first1, keys&#95;last1)</code> and <code>[keys&#95;first2, keys&#95;last2)</code> shall be sorted with respect to <code>operator&lt;</code>. 
* The resulting ranges shall not overlap with any input range.

**Returns**:
A <code>pair</code><code>p</code> such that <code>p.first</code> is the end of the output range of keys, and such that <code>p.second</code> is the end of the output range of values.

**See**:
* <code>set&#95;union&#95;by&#95;key</code>
* <code>set&#95;difference&#95;by&#95;key</code>
* <code>set&#95;symmetric&#95;difference&#95;by&#95;key</code>
* <code>sort&#95;by&#95;key</code>
* <code>is&#95;sorted</code>

<h3 id="function-set-intersection-by-key">
Function <code>thrust::set&#95;intersection&#95;by&#95;key</code>
</h3>

<code class="doxybook">
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename InputIterator1,</span>
<span>&nbsp;&nbsp;typename InputIterator2,</span>
<span>&nbsp;&nbsp;typename InputIterator3,</span>
<span>&nbsp;&nbsp;typename OutputIterator1,</span>
<span>&nbsp;&nbsp;typename OutputIterator2,</span>
<span>&nbsp;&nbsp;typename StrictWeakCompare&gt;</span>
<span>__host__ __device__ <a href="{{ site.baseurl }}/api/classes/structthrust_1_1pair.html">thrust::pair</a>< OutputIterator1, OutputIterator2 > </span><span><b>set_intersection_by_key</b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;InputIterator1 keys_first1,</span>
<span>&nbsp;&nbsp;InputIterator1 keys_last1,</span>
<span>&nbsp;&nbsp;InputIterator2 keys_first2,</span>
<span>&nbsp;&nbsp;InputIterator2 keys_last2,</span>
<span>&nbsp;&nbsp;InputIterator3 values_first1,</span>
<span>&nbsp;&nbsp;OutputIterator1 keys_result,</span>
<span>&nbsp;&nbsp;OutputIterator2 values_result,</span>
<span>&nbsp;&nbsp;StrictWeakCompare comp);</span></code>
<code>set&#95;intersection&#95;by&#95;key</code> performs a key-value intersection operation from set theory. <code>set&#95;intersection&#95;by&#95;key</code> constructs a sorted range that is the intersection of the sorted ranges <code>[keys&#95;first1, keys&#95;last1)</code> and <code>[keys&#95;first2, keys&#95;last2)</code>. Associated with each element from the input and output key ranges is a value element. The associated input value ranges need not be sorted.

In the simplest case, <code>set&#95;intersection&#95;by&#95;key</code> performs the "intersection" operation from set theory: the keys output range contains a copy of every element that is contained in both <code>[keys&#95;first1, keys&#95;last1)</code><code>[keys&#95;first2, keys&#95;last2)</code>. The general case is more complicated, because the input ranges may contain duplicate elements. The generalization is that if an element appears <code>m</code> times in <code>[keys&#95;first1, keys&#95;last1)</code> and <code>n</code> times in <code>[keys&#95;first2, keys&#95;last2)</code> (where <code>m</code> may be zero), then it appears <code>min(m,n)</code> times in the keys output range. <code>set&#95;intersection&#95;by&#95;key</code> is stable, meaning both that elements are copied from the first input range rather than the second, and that the relative order of elements in the output range is the same as the first input range.

Each time a key element is copied from <code>[keys&#95;first1, keys&#95;last1)</code> to the keys output range, the corresponding value element is copied from <code>[values&#95;first1, values&#95;last1)</code> to the values output range.

This version of <code>set&#95;intersection&#95;by&#95;key</code> compares objects using a function object <code>comp</code>.

The algorithm's execution is parallelized as determined by <code>exec</code>.


The following code snippet demonstrates how to use <code>set&#95;intersection&#95;by&#95;key</code> to compute the set intersection of two sets of integers sorted in descending order with their values using the <code>thrust::host</code> execution policy for parallelization:



```cpp
#include <thrust/set_operations.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>
...
int A_keys[6] = {11, 9, 7, 5, 3, 1};
int A_vals[6] = { 0, 0, 0, 0, 0, 0};

int B_keys[7] = {13, 8, 5, 3, 2, 1, 1};

int keys_result[7];
int vals_result[7];

thrust::pair<int*,int*> end = thrust::set_intersection_by_key(thrust::host, A_keys, A_keys + 6, B_keys, B_keys + 7, A_vals, keys_result, vals_result, thrust::greater<int>());

// keys_result is now {5, 3, 1}
// vals_result is now {0, 0, 0}
```

**Note**:
Unlike the other key-value set operations, <code>set&#95;intersection&#95;by&#95;key</code> is unique in that it has no <code>values&#95;first2</code> parameter because elements from the second input range are never copied to the output range.

**Template Parameters**:
* **`DerivedPolicy`** The name of the derived execution policy. 
* **`InputIterator1`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>, <code>InputIterator1</code> and <code>InputIterator2</code> have the same <code>value&#95;type</code>, <code>InputIterator1's</code><code>value&#95;type</code> is a model of <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a>, the ordering on <code>InputIterator1's</code><code>value&#95;type</code> is a strict weak ordering, as defined in the <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a> requirements, and <code>InputIterator1's</code><code>value&#95;type</code> is convertable to a type in <code>OutputIterator's</code> set of <code>value&#95;types</code>. 
* **`InputIterator2`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>, <code>InputIterator2</code> and <code>InputIterator1</code> have the same <code>value&#95;type</code>, <code>InputIterator2's</code><code>value&#95;type</code> is a model of <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a>, the ordering on <code>InputIterator2's</code><code>value&#95;type</code> is a strict weak ordering, as defined in the <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a> requirements, and <code>InputIterator2's</code><code>value&#95;type</code> is convertable to a type in <code>OutputIterator's</code> set of <code>value&#95;types</code>. 
* **`InputIterator3`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>, and <code>InputIterator3's</code><code>value&#95;type</code> is convertible to a type in <code>OutputIterator2's</code> set of <code>value&#95;types</code>. 
* **`OutputIterator1`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>. 
* **`OutputIterator2`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>. 
* **`StrictWeakCompare`** is a model of <a href="https://en.cppreference.com/w/cpp/concepts/strict_weak_order">Strict Weak Ordering</a>.

**Function Parameters**:
* **`exec`** The execution policy to use for parallelization. 
* **`keys_first1`** The beginning of the first input range of keys. 
* **`keys_last1`** The end of the first input range of keys. 
* **`keys_first2`** The beginning of the second input range of keys. 
* **`keys_last2`** The end of the second input range of keys. 
* **`values_first1`** The beginning of the first input range of values. 
* **`keys_result`** The beginning of the output range of keys. 
* **`values_result`** The beginning of the output range of values. 
* **`comp`** Comparison operator. 

**Preconditions**:
* The ranges <code>[keys&#95;first1, keys&#95;last1)</code> and <code>[keys&#95;first2, keys&#95;last2)</code> shall be sorted with respect to <code>comp</code>. 
* The resulting ranges shall not overlap with any input range.

**Returns**:
A <code>pair</code><code>p</code> such that <code>p.first</code> is the end of the output range of keys, and such that <code>p.second</code> is the end of the output range of values.

**See**:
* <code>set&#95;union&#95;by&#95;key</code>
* <code>set&#95;difference&#95;by&#95;key</code>
* <code>set&#95;symmetric&#95;difference&#95;by&#95;key</code>
* <code>sort&#95;by&#95;key</code>
* <code>is&#95;sorted</code>

<h3 id="function-set-intersection-by-key">
Function <code>thrust::set&#95;intersection&#95;by&#95;key</code>
</h3>

<code class="doxybook">
<span>template &lt;typename InputIterator1,</span>
<span>&nbsp;&nbsp;typename InputIterator2,</span>
<span>&nbsp;&nbsp;typename InputIterator3,</span>
<span>&nbsp;&nbsp;typename OutputIterator1,</span>
<span>&nbsp;&nbsp;typename OutputIterator2,</span>
<span>&nbsp;&nbsp;typename StrictWeakCompare&gt;</span>
<span><a href="{{ site.baseurl }}/api/classes/structthrust_1_1pair.html">thrust::pair</a>< OutputIterator1, OutputIterator2 > </span><span><b>set_intersection_by_key</b>(InputIterator1 keys_first1,</span>
<span>&nbsp;&nbsp;InputIterator1 keys_last1,</span>
<span>&nbsp;&nbsp;InputIterator2 keys_first2,</span>
<span>&nbsp;&nbsp;InputIterator2 keys_last2,</span>
<span>&nbsp;&nbsp;InputIterator3 values_first1,</span>
<span>&nbsp;&nbsp;OutputIterator1 keys_result,</span>
<span>&nbsp;&nbsp;OutputIterator2 values_result,</span>
<span>&nbsp;&nbsp;StrictWeakCompare comp);</span></code>
<code>set&#95;intersection&#95;by&#95;key</code> performs a key-value intersection operation from set theory. <code>set&#95;intersection&#95;by&#95;key</code> constructs a sorted range that is the intersection of the sorted ranges <code>[keys&#95;first1, keys&#95;last1)</code> and <code>[keys&#95;first2, keys&#95;last2)</code>. Associated with each element from the input and output key ranges is a value element. The associated input value ranges need not be sorted.

In the simplest case, <code>set&#95;intersection&#95;by&#95;key</code> performs the "intersection" operation from set theory: the keys output range contains a copy of every element that is contained in both <code>[keys&#95;first1, keys&#95;last1)</code><code>[keys&#95;first2, keys&#95;last2)</code>. The general case is more complicated, because the input ranges may contain duplicate elements. The generalization is that if an element appears <code>m</code> times in <code>[keys&#95;first1, keys&#95;last1)</code> and <code>n</code> times in <code>[keys&#95;first2, keys&#95;last2)</code> (where <code>m</code> may be zero), then it appears <code>min(m,n)</code> times in the keys output range. <code>set&#95;intersection&#95;by&#95;key</code> is stable, meaning both that elements are copied from the first input range rather than the second, and that the relative order of elements in the output range is the same as the first input range.

Each time a key element is copied from <code>[keys&#95;first1, keys&#95;last1)</code> to the keys output range, the corresponding value element is copied from <code>[values&#95;first1, values&#95;last1)</code> to the values output range.

This version of <code>set&#95;intersection&#95;by&#95;key</code> compares objects using a function object <code>comp</code>.


The following code snippet demonstrates how to use <code>set&#95;intersection&#95;by&#95;key</code> to compute the set intersection of two sets of integers sorted in descending order with their values.



```cpp
#include <thrust/set_operations.h>
#include <thrust/functional.h>
...
int A_keys[6] = {11, 9, 7, 5, 3, 1};
int A_vals[6] = { 0, 0, 0, 0, 0, 0};

int B_keys[7] = {13, 8, 5, 3, 2, 1, 1};

int keys_result[7];
int vals_result[7];

thrust::pair<int*,int*> end = thrust::set_intersection_by_key(A_keys, A_keys + 6, B_keys, B_keys + 7, A_vals, keys_result, vals_result, thrust::greater<int>());

// keys_result is now {5, 3, 1}
// vals_result is now {0, 0, 0}
```

**Note**:
Unlike the other key-value set operations, <code>set&#95;intersection&#95;by&#95;key</code> is unique in that it has no <code>values&#95;first2</code> parameter because elements from the second input range are never copied to the output range.

**Template Parameters**:
* **`InputIterator1`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>, <code>InputIterator1</code> and <code>InputIterator2</code> have the same <code>value&#95;type</code>, <code>InputIterator1's</code><code>value&#95;type</code> is a model of <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a>, the ordering on <code>InputIterator1's</code><code>value&#95;type</code> is a strict weak ordering, as defined in the <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a> requirements, and <code>InputIterator1's</code><code>value&#95;type</code> is convertable to a type in <code>OutputIterator's</code> set of <code>value&#95;types</code>. 
* **`InputIterator2`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>, <code>InputIterator2</code> and <code>InputIterator1</code> have the same <code>value&#95;type</code>, <code>InputIterator2's</code><code>value&#95;type</code> is a model of <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a>, the ordering on <code>InputIterator2's</code><code>value&#95;type</code> is a strict weak ordering, as defined in the <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a> requirements, and <code>InputIterator2's</code><code>value&#95;type</code> is convertable to a type in <code>OutputIterator's</code> set of <code>value&#95;types</code>. 
* **`InputIterator3`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>, and <code>InputIterator3's</code><code>value&#95;type</code> is convertible to a type in <code>OutputIterator2's</code> set of <code>value&#95;types</code>. 
* **`OutputIterator1`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>. 
* **`OutputIterator2`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>. 
* **`StrictWeakCompare`** is a model of <a href="https://en.cppreference.com/w/cpp/concepts/strict_weak_order">Strict Weak Ordering</a>.

**Function Parameters**:
* **`keys_first1`** The beginning of the first input range of keys. 
* **`keys_last1`** The end of the first input range of keys. 
* **`keys_first2`** The beginning of the second input range of keys. 
* **`keys_last2`** The end of the second input range of keys. 
* **`values_first1`** The beginning of the first input range of values. 
* **`keys_result`** The beginning of the output range of keys. 
* **`values_result`** The beginning of the output range of values. 
* **`comp`** Comparison operator. 

**Preconditions**:
* The ranges <code>[keys&#95;first1, keys&#95;last1)</code> and <code>[keys&#95;first2, keys&#95;last2)</code> shall be sorted with respect to <code>comp</code>. 
* The resulting ranges shall not overlap with any input range.

**Returns**:
A <code>pair</code><code>p</code> such that <code>p.first</code> is the end of the output range of keys, and such that <code>p.second</code> is the end of the output range of values.

**See**:
* <code>set&#95;union&#95;by&#95;key</code>
* <code>set&#95;difference&#95;by&#95;key</code>
* <code>set&#95;symmetric&#95;difference&#95;by&#95;key</code>
* <code>sort&#95;by&#95;key</code>
* <code>is&#95;sorted</code>

<h3 id="function-set-symmetric-difference-by-key">
Function <code>thrust::set&#95;symmetric&#95;difference&#95;by&#95;key</code>
</h3>

<code class="doxybook">
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename InputIterator1,</span>
<span>&nbsp;&nbsp;typename InputIterator2,</span>
<span>&nbsp;&nbsp;typename InputIterator3,</span>
<span>&nbsp;&nbsp;typename InputIterator4,</span>
<span>&nbsp;&nbsp;typename OutputIterator1,</span>
<span>&nbsp;&nbsp;typename OutputIterator2&gt;</span>
<span>__host__ __device__ <a href="{{ site.baseurl }}/api/classes/structthrust_1_1pair.html">thrust::pair</a>< OutputIterator1, OutputIterator2 > </span><span><b>set_symmetric_difference_by_key</b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;InputIterator1 keys_first1,</span>
<span>&nbsp;&nbsp;InputIterator1 keys_last1,</span>
<span>&nbsp;&nbsp;InputIterator2 keys_first2,</span>
<span>&nbsp;&nbsp;InputIterator2 keys_last2,</span>
<span>&nbsp;&nbsp;InputIterator3 values_first1,</span>
<span>&nbsp;&nbsp;InputIterator4 values_first2,</span>
<span>&nbsp;&nbsp;OutputIterator1 keys_result,</span>
<span>&nbsp;&nbsp;OutputIterator2 values_result);</span></code>
<code>set&#95;symmetric&#95;difference&#95;by&#95;key</code> performs a key-value symmetric difference operation from set theory. <code>set&#95;difference&#95;by&#95;key</code> constructs a sorted range that is the symmetric difference of the sorted ranges <code>[keys&#95;first1, keys&#95;last1)</code> and <code>[keys&#95;first2, keys&#95;last2)</code>. Associated with each element from the input and output key ranges is a value element. The associated input value ranges need not be sorted.

In the simplest case, <code>set&#95;symmetric&#95;difference&#95;by&#95;key</code> performs a set theoretic calculation: it constructs the union of the two sets A - B and B - A, where A and B are the two input ranges. That is, the output range contains a copy of every element that is contained in <code>[keys&#95;first1, keys&#95;last1)</code> but not <code>[keys&#95;first2, keys&#95;last1)</code>, and a copy of every element that is contained in <code>[keys&#95;first2, keys&#95;last2)</code> but not <code>[keys&#95;first1, keys&#95;last1)</code>. The general case is more complicated, because the input ranges may contain duplicate elements. The generalization is that if <code>[keys&#95;first1, keys&#95;last1)</code> contains <code>m</code> elements that are equivalent to each other and <code>[keys&#95;first2, keys&#95;last1)</code> contains <code>n</code> elements that are equivalent to them, then <code>|m - n|</code> of those elements shall be copied to the output range: the last <code>m - n</code> elements from <code>[keys&#95;first1, keys&#95;last1)</code> if <code>m &gt; n</code>, and the last <code>n - m</code> of these elements from <code>[keys&#95;first2, keys&#95;last2)</code> if <code>m &lt; n</code>.

Each time a key element is copied from <code>[keys&#95;first1, keys&#95;last1)</code> or <code>[keys&#95;first2, keys&#95;last2)</code> is copied to the keys output range, the corresponding value element is copied from the corresponding values input range (beginning at <code>values&#95;first1</code> or <code>values&#95;first2</code>) to the values output range.

This version of <code>set&#95;symmetric&#95;difference&#95;by&#95;key</code> compares key elements using <code>operator&lt;</code>.

The algorithm's execution is parallelized as determined by <code>exec</code>.


The following code snippet demonstrates how to use <code>set&#95;symmetric&#95;difference&#95;by&#95;key</code> to compute the symmetric difference of two sets of integers sorted in ascending order with their values using the <code>thrust::host</code> execution policy for parallelization:



```cpp
#include <thrust/set_operations.h>
#include <thrust/execution_policy.h>
...
int A_keys[6] = {0, 1, 2, 2, 4, 6, 7};
int A_vals[6] = {0, 0, 0, 0, 0, 0, 0};

int B_keys[5] = {1, 1, 2, 5, 8};
int B_vals[5] = {1, 1, 1, 1, 1};

int keys_result[6];
int vals_result[6];

thrust::pair<int*,int*> end = thrust::set_symmetric_difference_by_key(thrust::host, A_keys, A_keys + 6, B_keys, B_keys + 5, A_vals, B_vals, keys_result, vals_result);
// keys_result is now {0, 4, 5, 6, 7, 8}
// vals_result is now {0, 0, 1, 0, 0, 1}
```

**Template Parameters**:
* **`DerivedPolicy`** The name of the derived execution policy. 
* **`InputIterator1`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>, <code>InputIterator1</code> and <code>InputIterator2</code> have the same <code>value&#95;type</code>, <code>InputIterator1's</code><code>value&#95;type</code> is a model of <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a>, the ordering on <code>InputIterator1's</code><code>value&#95;type</code> is a strict weak ordering, as defined in the <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a> requirements, and <code>InputIterator1's</code><code>value&#95;type</code> is convertable to a type in <code>OutputIterator's</code> set of <code>value&#95;types</code>. 
* **`InputIterator2`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>, <code>InputIterator2</code> and <code>InputIterator1</code> have the same <code>value&#95;type</code>, <code>InputIterator2's</code><code>value&#95;type</code> is a model of <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a>, the ordering on <code>InputIterator2's</code><code>value&#95;type</code> is a strict weak ordering, as defined in the <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a> requirements, and <code>InputIterator2's</code><code>value&#95;type</code> is convertable to a type in <code>OutputIterator's</code> set of <code>value&#95;types</code>. 
* **`InputIterator3`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>, and <code>InputIterator3's</code><code>value&#95;type</code> is convertible to a type in <code>OutputIterator2's</code> set of <code>value&#95;types</code>. 
* **`InputIterator4`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>, and <code>InputIterator4's</code><code>value&#95;type</code> is convertible to a type in <code>OutputIterator2's</code> set of <code>value&#95;types</code>. 
* **`OutputIterator1`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>. 
* **`OutputIterator2`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>.

**Function Parameters**:
* **`exec`** The execution policy to use for parallelization. 
* **`keys_first1`** The beginning of the first input range of keys. 
* **`keys_last1`** The end of the first input range of keys. 
* **`keys_first2`** The beginning of the second input range of keys. 
* **`keys_last2`** The end of the second input range of keys. 
* **`values_first1`** The beginning of the first input range of values. 
* **`values_first2`** The beginning of the first input range of values. 
* **`keys_result`** The beginning of the output range of keys. 
* **`values_result`** The beginning of the output range of values. 

**Preconditions**:
* The ranges <code>[keys&#95;first1, keys&#95;last1)</code> and <code>[keys&#95;first2, keys&#95;last2)</code> shall be sorted with respect to <code>operator&lt;</code>. 
* The resulting ranges shall not overlap with any input range.

**Returns**:
A <code>pair</code><code>p</code> such that <code>p.first</code> is the end of the output range of keys, and such that <code>p.second</code> is the end of the output range of values.

**See**:
* <code>set&#95;union&#95;by&#95;key</code>
* <code>set&#95;intersection&#95;by&#95;key</code>
* <code>set&#95;difference&#95;by&#95;key</code>
* <code>sort&#95;by&#95;key</code>
* <code>is&#95;sorted</code>

<h3 id="function-set-symmetric-difference-by-key">
Function <code>thrust::set&#95;symmetric&#95;difference&#95;by&#95;key</code>
</h3>

<code class="doxybook">
<span>template &lt;typename InputIterator1,</span>
<span>&nbsp;&nbsp;typename InputIterator2,</span>
<span>&nbsp;&nbsp;typename InputIterator3,</span>
<span>&nbsp;&nbsp;typename InputIterator4,</span>
<span>&nbsp;&nbsp;typename OutputIterator1,</span>
<span>&nbsp;&nbsp;typename OutputIterator2&gt;</span>
<span><a href="{{ site.baseurl }}/api/classes/structthrust_1_1pair.html">thrust::pair</a>< OutputIterator1, OutputIterator2 > </span><span><b>set_symmetric_difference_by_key</b>(InputIterator1 keys_first1,</span>
<span>&nbsp;&nbsp;InputIterator1 keys_last1,</span>
<span>&nbsp;&nbsp;InputIterator2 keys_first2,</span>
<span>&nbsp;&nbsp;InputIterator2 keys_last2,</span>
<span>&nbsp;&nbsp;InputIterator3 values_first1,</span>
<span>&nbsp;&nbsp;InputIterator4 values_first2,</span>
<span>&nbsp;&nbsp;OutputIterator1 keys_result,</span>
<span>&nbsp;&nbsp;OutputIterator2 values_result);</span></code>
<code>set&#95;symmetric&#95;difference&#95;by&#95;key</code> performs a key-value symmetric difference operation from set theory. <code>set&#95;difference&#95;by&#95;key</code> constructs a sorted range that is the symmetric difference of the sorted ranges <code>[keys&#95;first1, keys&#95;last1)</code> and <code>[keys&#95;first2, keys&#95;last2)</code>. Associated with each element from the input and output key ranges is a value element. The associated input value ranges need not be sorted.

In the simplest case, <code>set&#95;symmetric&#95;difference&#95;by&#95;key</code> performs a set theoretic calculation: it constructs the union of the two sets A - B and B - A, where A and B are the two input ranges. That is, the output range contains a copy of every element that is contained in <code>[keys&#95;first1, keys&#95;last1)</code> but not <code>[keys&#95;first2, keys&#95;last1)</code>, and a copy of every element that is contained in <code>[keys&#95;first2, keys&#95;last2)</code> but not <code>[keys&#95;first1, keys&#95;last1)</code>. The general case is more complicated, because the input ranges may contain duplicate elements. The generalization is that if <code>[keys&#95;first1, keys&#95;last1)</code> contains <code>m</code> elements that are equivalent to each other and <code>[keys&#95;first2, keys&#95;last1)</code> contains <code>n</code> elements that are equivalent to them, then <code>|m - n|</code> of those elements shall be copied to the output range: the last <code>m - n</code> elements from <code>[keys&#95;first1, keys&#95;last1)</code> if <code>m &gt; n</code>, and the last <code>n - m</code> of these elements from <code>[keys&#95;first2, keys&#95;last2)</code> if <code>m &lt; n</code>.

Each time a key element is copied from <code>[keys&#95;first1, keys&#95;last1)</code> or <code>[keys&#95;first2, keys&#95;last2)</code> is copied to the keys output range, the corresponding value element is copied from the corresponding values input range (beginning at <code>values&#95;first1</code> or <code>values&#95;first2</code>) to the values output range.

This version of <code>set&#95;symmetric&#95;difference&#95;by&#95;key</code> compares key elements using <code>operator&lt;</code>.


The following code snippet demonstrates how to use <code>set&#95;symmetric&#95;difference&#95;by&#95;key</code> to compute the symmetric difference of two sets of integers sorted in ascending order with their values.



```cpp
#include <thrust/set_operations.h>
...
int A_keys[6] = {0, 1, 2, 2, 4, 6, 7};
int A_vals[6] = {0, 0, 0, 0, 0, 0, 0};

int B_keys[5] = {1, 1, 2, 5, 8};
int B_vals[5] = {1, 1, 1, 1, 1};

int keys_result[6];
int vals_result[6];

thrust::pair<int*,int*> end = thrust::set_symmetric_difference_by_key(A_keys, A_keys + 6, B_keys, B_keys + 5, A_vals, B_vals, keys_result, vals_result);
// keys_result is now {0, 4, 5, 6, 7, 8}
// vals_result is now {0, 0, 1, 0, 0, 1}
```

**Template Parameters**:
* **`InputIterator1`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>, <code>InputIterator1</code> and <code>InputIterator2</code> have the same <code>value&#95;type</code>, <code>InputIterator1's</code><code>value&#95;type</code> is a model of <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a>, the ordering on <code>InputIterator1's</code><code>value&#95;type</code> is a strict weak ordering, as defined in the <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a> requirements, and <code>InputIterator1's</code><code>value&#95;type</code> is convertable to a type in <code>OutputIterator's</code> set of <code>value&#95;types</code>. 
* **`InputIterator2`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>, <code>InputIterator2</code> and <code>InputIterator1</code> have the same <code>value&#95;type</code>, <code>InputIterator2's</code><code>value&#95;type</code> is a model of <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a>, the ordering on <code>InputIterator2's</code><code>value&#95;type</code> is a strict weak ordering, as defined in the <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a> requirements, and <code>InputIterator2's</code><code>value&#95;type</code> is convertable to a type in <code>OutputIterator's</code> set of <code>value&#95;types</code>. 
* **`InputIterator3`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>, and <code>InputIterator3's</code><code>value&#95;type</code> is convertible to a type in <code>OutputIterator2's</code> set of <code>value&#95;types</code>. 
* **`InputIterator4`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>, and <code>InputIterator4's</code><code>value&#95;type</code> is convertible to a type in <code>OutputIterator2's</code> set of <code>value&#95;types</code>. 
* **`OutputIterator1`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>. 
* **`OutputIterator2`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>.

**Function Parameters**:
* **`keys_first1`** The beginning of the first input range of keys. 
* **`keys_last1`** The end of the first input range of keys. 
* **`keys_first2`** The beginning of the second input range of keys. 
* **`keys_last2`** The end of the second input range of keys. 
* **`values_first1`** The beginning of the first input range of values. 
* **`values_first2`** The beginning of the first input range of values. 
* **`keys_result`** The beginning of the output range of keys. 
* **`values_result`** The beginning of the output range of values. 

**Preconditions**:
* The ranges <code>[keys&#95;first1, keys&#95;last1)</code> and <code>[keys&#95;first2, keys&#95;last2)</code> shall be sorted with respect to <code>operator&lt;</code>. 
* The resulting ranges shall not overlap with any input range.

**Returns**:
A <code>pair</code><code>p</code> such that <code>p.first</code> is the end of the output range of keys, and such that <code>p.second</code> is the end of the output range of values.

**See**:
* <code>set&#95;union&#95;by&#95;key</code>
* <code>set&#95;intersection&#95;by&#95;key</code>
* <code>set&#95;difference&#95;by&#95;key</code>
* <code>sort&#95;by&#95;key</code>
* <code>is&#95;sorted</code>

<h3 id="function-set-symmetric-difference-by-key">
Function <code>thrust::set&#95;symmetric&#95;difference&#95;by&#95;key</code>
</h3>

<code class="doxybook">
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename InputIterator1,</span>
<span>&nbsp;&nbsp;typename InputIterator2,</span>
<span>&nbsp;&nbsp;typename InputIterator3,</span>
<span>&nbsp;&nbsp;typename InputIterator4,</span>
<span>&nbsp;&nbsp;typename OutputIterator1,</span>
<span>&nbsp;&nbsp;typename OutputIterator2,</span>
<span>&nbsp;&nbsp;typename StrictWeakCompare&gt;</span>
<span>__host__ __device__ <a href="{{ site.baseurl }}/api/classes/structthrust_1_1pair.html">thrust::pair</a>< OutputIterator1, OutputIterator2 > </span><span><b>set_symmetric_difference_by_key</b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;InputIterator1 keys_first1,</span>
<span>&nbsp;&nbsp;InputIterator1 keys_last1,</span>
<span>&nbsp;&nbsp;InputIterator2 keys_first2,</span>
<span>&nbsp;&nbsp;InputIterator2 keys_last2,</span>
<span>&nbsp;&nbsp;InputIterator3 values_first1,</span>
<span>&nbsp;&nbsp;InputIterator4 values_first2,</span>
<span>&nbsp;&nbsp;OutputIterator1 keys_result,</span>
<span>&nbsp;&nbsp;OutputIterator2 values_result,</span>
<span>&nbsp;&nbsp;StrictWeakCompare comp);</span></code>
<code>set&#95;symmetric&#95;difference&#95;by&#95;key</code> performs a key-value symmetric difference operation from set theory. <code>set&#95;difference&#95;by&#95;key</code> constructs a sorted range that is the symmetric difference of the sorted ranges <code>[keys&#95;first1, keys&#95;last1)</code> and <code>[keys&#95;first2, keys&#95;last2)</code>. Associated with each element from the input and output key ranges is a value element. The associated input value ranges need not be sorted.

In the simplest case, <code>set&#95;symmetric&#95;difference&#95;by&#95;key</code> performs a set theoretic calculation: it constructs the union of the two sets A - B and B - A, where A and B are the two input ranges. That is, the output range contains a copy of every element that is contained in <code>[keys&#95;first1, keys&#95;last1)</code> but not <code>[keys&#95;first2, keys&#95;last1)</code>, and a copy of every element that is contained in <code>[keys&#95;first2, keys&#95;last2)</code> but not <code>[keys&#95;first1, keys&#95;last1)</code>. The general case is more complicated, because the input ranges may contain duplicate elements. The generalization is that if <code>[keys&#95;first1, keys&#95;last1)</code> contains <code>m</code> elements that are equivalent to each other and <code>[keys&#95;first2, keys&#95;last1)</code> contains <code>n</code> elements that are equivalent to them, then <code>|m - n|</code> of those elements shall be copied to the output range: the last <code>m - n</code> elements from <code>[keys&#95;first1, keys&#95;last1)</code> if <code>m &gt; n</code>, and the last <code>n - m</code> of these elements from <code>[keys&#95;first2, keys&#95;last2)</code> if <code>m &lt; n</code>.

Each time a key element is copied from <code>[keys&#95;first1, keys&#95;last1)</code> or <code>[keys&#95;first2, keys&#95;last2)</code> is copied to the keys output range, the corresponding value element is copied from the corresponding values input range (beginning at <code>values&#95;first1</code> or <code>values&#95;first2</code>) to the values output range.

This version of <code>set&#95;symmetric&#95;difference&#95;by&#95;key</code> compares key elements using a function object <code>comp</code>.

The algorithm's execution is parallelized as determined by <code>exec</code>.


The following code snippet demonstrates how to use <code>set&#95;symmetric&#95;difference&#95;by&#95;key</code> to compute the symmetric difference of two sets of integers sorted in descending order with their values using the <code>thrust::host</code> execution policy for parallelization:



```cpp
#include <thrust/set_operations.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>
...
int A_keys[6] = {7, 6, 4, 2, 2, 1, 0};
int A_vals[6] = {0, 0, 0, 0, 0, 0, 0};

int B_keys[5] = {8, 5, 2, 1, 1};
int B_vals[5] = {1, 1, 1, 1, 1};

int keys_result[6];
int vals_result[6];

thrust::pair<int*,int*> end = thrust::set_symmetric_difference_by_key(thrust::host, A_keys, A_keys + 6, B_keys, B_keys + 5, A_vals, B_vals, keys_result, vals_result);
// keys_result is now {8, 7, 6, 5, 4, 0}
// vals_result is now {1, 0, 0, 1, 0, 0}
```

**Template Parameters**:
* **`DerivedPolicy`** The name of the derived execution policy. 
* **`InputIterator1`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>, <code>InputIterator1</code> and <code>InputIterator2</code> have the same <code>value&#95;type</code>, <code>InputIterator1's</code><code>value&#95;type</code> is a model of <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a>, the ordering on <code>InputIterator1's</code><code>value&#95;type</code> is a strict weak ordering, as defined in the <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a> requirements, and <code>InputIterator1's</code><code>value&#95;type</code> is convertable to a type in <code>OutputIterator's</code> set of <code>value&#95;types</code>. 
* **`InputIterator2`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>, <code>InputIterator2</code> and <code>InputIterator1</code> have the same <code>value&#95;type</code>, <code>InputIterator2's</code><code>value&#95;type</code> is a model of <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a>, the ordering on <code>InputIterator2's</code><code>value&#95;type</code> is a strict weak ordering, as defined in the <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a> requirements, and <code>InputIterator2's</code><code>value&#95;type</code> is convertable to a type in <code>OutputIterator's</code> set of <code>value&#95;types</code>. 
* **`InputIterator3`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>, and <code>InputIterator3's</code><code>value&#95;type</code> is convertible to a type in <code>OutputIterator2's</code> set of <code>value&#95;types</code>. 
* **`InputIterator4`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>, and <code>InputIterator4's</code><code>value&#95;type</code> is convertible to a type in <code>OutputIterator2's</code> set of <code>value&#95;types</code>. 
* **`OutputIterator1`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>. 
* **`OutputIterator2`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>. 
* **`StrictWeakCompare`** is a model of <a href="https://en.cppreference.com/w/cpp/concepts/strict_weak_order">Strict Weak Ordering</a>.

**Function Parameters**:
* **`exec`** The execution policy to use for parallelization. 
* **`keys_first1`** The beginning of the first input range of keys. 
* **`keys_last1`** The end of the first input range of keys. 
* **`keys_first2`** The beginning of the second input range of keys. 
* **`keys_last2`** The end of the second input range of keys. 
* **`values_first1`** The beginning of the first input range of values. 
* **`values_first2`** The beginning of the first input range of values. 
* **`keys_result`** The beginning of the output range of keys. 
* **`values_result`** The beginning of the output range of values. 
* **`comp`** Comparison operator. 

**Preconditions**:
* The ranges <code>[keys&#95;first1, keys&#95;last1)</code> and <code>[keys&#95;first2, keys&#95;last2)</code> shall be sorted with respect to <code>comp</code>. 
* The resulting ranges shall not overlap with any input range.

**Returns**:
A <code>pair</code><code>p</code> such that <code>p.first</code> is the end of the output range of keys, and such that <code>p.second</code> is the end of the output range of values.

**See**:
* <code>set&#95;union&#95;by&#95;key</code>
* <code>set&#95;intersection&#95;by&#95;key</code>
* <code>set&#95;difference&#95;by&#95;key</code>
* <code>sort&#95;by&#95;key</code>
* <code>is&#95;sorted</code>

<h3 id="function-set-symmetric-difference-by-key">
Function <code>thrust::set&#95;symmetric&#95;difference&#95;by&#95;key</code>
</h3>

<code class="doxybook">
<span>template &lt;typename InputIterator1,</span>
<span>&nbsp;&nbsp;typename InputIterator2,</span>
<span>&nbsp;&nbsp;typename InputIterator3,</span>
<span>&nbsp;&nbsp;typename InputIterator4,</span>
<span>&nbsp;&nbsp;typename OutputIterator1,</span>
<span>&nbsp;&nbsp;typename OutputIterator2,</span>
<span>&nbsp;&nbsp;typename StrictWeakCompare&gt;</span>
<span><a href="{{ site.baseurl }}/api/classes/structthrust_1_1pair.html">thrust::pair</a>< OutputIterator1, OutputIterator2 > </span><span><b>set_symmetric_difference_by_key</b>(InputIterator1 keys_first1,</span>
<span>&nbsp;&nbsp;InputIterator1 keys_last1,</span>
<span>&nbsp;&nbsp;InputIterator2 keys_first2,</span>
<span>&nbsp;&nbsp;InputIterator2 keys_last2,</span>
<span>&nbsp;&nbsp;InputIterator3 values_first1,</span>
<span>&nbsp;&nbsp;InputIterator4 values_first2,</span>
<span>&nbsp;&nbsp;OutputIterator1 keys_result,</span>
<span>&nbsp;&nbsp;OutputIterator2 values_result,</span>
<span>&nbsp;&nbsp;StrictWeakCompare comp);</span></code>
<code>set&#95;symmetric&#95;difference&#95;by&#95;key</code> performs a key-value symmetric difference operation from set theory. <code>set&#95;difference&#95;by&#95;key</code> constructs a sorted range that is the symmetric difference of the sorted ranges <code>[keys&#95;first1, keys&#95;last1)</code> and <code>[keys&#95;first2, keys&#95;last2)</code>. Associated with each element from the input and output key ranges is a value element. The associated input value ranges need not be sorted.

In the simplest case, <code>set&#95;symmetric&#95;difference&#95;by&#95;key</code> performs a set theoretic calculation: it constructs the union of the two sets A - B and B - A, where A and B are the two input ranges. That is, the output range contains a copy of every element that is contained in <code>[keys&#95;first1, keys&#95;last1)</code> but not <code>[keys&#95;first2, keys&#95;last1)</code>, and a copy of every element that is contained in <code>[keys&#95;first2, keys&#95;last2)</code> but not <code>[keys&#95;first1, keys&#95;last1)</code>. The general case is more complicated, because the input ranges may contain duplicate elements. The generalization is that if <code>[keys&#95;first1, keys&#95;last1)</code> contains <code>m</code> elements that are equivalent to each other and <code>[keys&#95;first2, keys&#95;last1)</code> contains <code>n</code> elements that are equivalent to them, then <code>|m - n|</code> of those elements shall be copied to the output range: the last <code>m - n</code> elements from <code>[keys&#95;first1, keys&#95;last1)</code> if <code>m &gt; n</code>, and the last <code>n - m</code> of these elements from <code>[keys&#95;first2, keys&#95;last2)</code> if <code>m &lt; n</code>.

Each time a key element is copied from <code>[keys&#95;first1, keys&#95;last1)</code> or <code>[keys&#95;first2, keys&#95;last2)</code> is copied to the keys output range, the corresponding value element is copied from the corresponding values input range (beginning at <code>values&#95;first1</code> or <code>values&#95;first2</code>) to the values output range.

This version of <code>set&#95;symmetric&#95;difference&#95;by&#95;key</code> compares key elements using a function object <code>comp</code>.


The following code snippet demonstrates how to use <code>set&#95;symmetric&#95;difference&#95;by&#95;key</code> to compute the symmetric difference of two sets of integers sorted in descending order with their values.



```cpp
#include <thrust/set_operations.h>
#include <thrust/functional.h>
...
int A_keys[6] = {7, 6, 4, 2, 2, 1, 0};
int A_vals[6] = {0, 0, 0, 0, 0, 0, 0};

int B_keys[5] = {8, 5, 2, 1, 1};
int B_vals[5] = {1, 1, 1, 1, 1};

int keys_result[6];
int vals_result[6];

thrust::pair<int*,int*> end = thrust::set_symmetric_difference_by_key(A_keys, A_keys + 6, B_keys, B_keys + 5, A_vals, B_vals, keys_result, vals_result);
// keys_result is now {8, 7, 6, 5, 4, 0}
// vals_result is now {1, 0, 0, 1, 0, 0}
```

**Template Parameters**:
* **`InputIterator1`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>, <code>InputIterator1</code> and <code>InputIterator2</code> have the same <code>value&#95;type</code>, <code>InputIterator1's</code><code>value&#95;type</code> is a model of <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a>, the ordering on <code>InputIterator1's</code><code>value&#95;type</code> is a strict weak ordering, as defined in the <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a> requirements, and <code>InputIterator1's</code><code>value&#95;type</code> is convertable to a type in <code>OutputIterator's</code> set of <code>value&#95;types</code>. 
* **`InputIterator2`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>, <code>InputIterator2</code> and <code>InputIterator1</code> have the same <code>value&#95;type</code>, <code>InputIterator2's</code><code>value&#95;type</code> is a model of <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a>, the ordering on <code>InputIterator2's</code><code>value&#95;type</code> is a strict weak ordering, as defined in the <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a> requirements, and <code>InputIterator2's</code><code>value&#95;type</code> is convertable to a type in <code>OutputIterator's</code> set of <code>value&#95;types</code>. 
* **`InputIterator3`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>, and <code>InputIterator3's</code><code>value&#95;type</code> is convertible to a type in <code>OutputIterator2's</code> set of <code>value&#95;types</code>. 
* **`InputIterator4`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>, and <code>InputIterator4's</code><code>value&#95;type</code> is convertible to a type in <code>OutputIterator2's</code> set of <code>value&#95;types</code>. 
* **`OutputIterator1`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>. 
* **`OutputIterator2`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>. 
* **`StrictWeakCompare`** is a model of <a href="https://en.cppreference.com/w/cpp/concepts/strict_weak_order">Strict Weak Ordering</a>.

**Function Parameters**:
* **`keys_first1`** The beginning of the first input range of keys. 
* **`keys_last1`** The end of the first input range of keys. 
* **`keys_first2`** The beginning of the second input range of keys. 
* **`keys_last2`** The end of the second input range of keys. 
* **`values_first1`** The beginning of the first input range of values. 
* **`values_first2`** The beginning of the first input range of values. 
* **`keys_result`** The beginning of the output range of keys. 
* **`values_result`** The beginning of the output range of values. 
* **`comp`** Comparison operator. 

**Preconditions**:
* The ranges <code>[keys&#95;first1, keys&#95;last1)</code> and <code>[keys&#95;first2, keys&#95;last2)</code> shall be sorted with respect to <code>comp</code>. 
* The resulting ranges shall not overlap with any input range.

**Returns**:
A <code>pair</code><code>p</code> such that <code>p.first</code> is the end of the output range of keys, and such that <code>p.second</code> is the end of the output range of values.

**See**:
* <code>set&#95;union&#95;by&#95;key</code>
* <code>set&#95;intersection&#95;by&#95;key</code>
* <code>set&#95;difference&#95;by&#95;key</code>
* <code>sort&#95;by&#95;key</code>
* <code>is&#95;sorted</code>

<h3 id="function-set-union-by-key">
Function <code>thrust::set&#95;union&#95;by&#95;key</code>
</h3>

<code class="doxybook">
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename InputIterator1,</span>
<span>&nbsp;&nbsp;typename InputIterator2,</span>
<span>&nbsp;&nbsp;typename InputIterator3,</span>
<span>&nbsp;&nbsp;typename InputIterator4,</span>
<span>&nbsp;&nbsp;typename OutputIterator1,</span>
<span>&nbsp;&nbsp;typename OutputIterator2&gt;</span>
<span>__host__ __device__ <a href="{{ site.baseurl }}/api/classes/structthrust_1_1pair.html">thrust::pair</a>< OutputIterator1, OutputIterator2 > </span><span><b>set_union_by_key</b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;InputIterator1 keys_first1,</span>
<span>&nbsp;&nbsp;InputIterator1 keys_last1,</span>
<span>&nbsp;&nbsp;InputIterator2 keys_first2,</span>
<span>&nbsp;&nbsp;InputIterator2 keys_last2,</span>
<span>&nbsp;&nbsp;InputIterator3 values_first1,</span>
<span>&nbsp;&nbsp;InputIterator4 values_first2,</span>
<span>&nbsp;&nbsp;OutputIterator1 keys_result,</span>
<span>&nbsp;&nbsp;OutputIterator2 values_result);</span></code>
<code>set&#95;union&#95;by&#95;key</code> performs a key-value union operation from set theory. <code>set&#95;union&#95;by&#95;key</code> constructs a sorted range that is the union of the sorted ranges <code>[keys&#95;first1, keys&#95;last1)</code> and <code>[keys&#95;first2, keys&#95;last2)</code>. Associated with each element from the input and output key ranges is a value element. The associated input value ranges need not be sorted.

In the simplest case, <code>set&#95;union&#95;by&#95;key</code> performs the "union" operation from set theory: the output range contains a copy of every element that is contained in <code>[keys&#95;first1, keys&#95;last1)</code>, <code>[keys&#95;first2, keys&#95;last1)</code>, or both. The general case is more complicated, because the input ranges may contain duplicate elements. The generalization is that if <code>[keys&#95;first1, keys&#95;last1)</code> contains <code>m</code> elements that are equivalent to each other and if <code>[keys&#95;first2, keys&#95;last2)</code> contains <code>n</code> elements that are equivalent to them, then all <code>m</code> elements from the first range shall be copied to the output range, in order, and then <code>max(n - m, 0)</code> elements from the second range shall be copied to the output, in order.

Each time a key element is copied from <code>[keys&#95;first1, keys&#95;last1)</code> or <code>[keys&#95;first2, keys&#95;last2)</code> is copied to the keys output range, the corresponding value element is copied from the corresponding values input range (beginning at <code>values&#95;first1</code> or <code>values&#95;first2</code>) to the values output range.

This version of <code>set&#95;union&#95;by&#95;key</code> compares key elements using <code>operator&lt;</code>.

The algorithm's execution is parallelized as determined by <code>exec</code>.


The following code snippet demonstrates how to use <code>set&#95;symmetric&#95;difference&#95;by&#95;key</code> to compute the symmetric difference of two sets of integers sorted in ascending order with their values using the <code>thrust::host</code> execution policy for parallelization:



```cpp
#include <thrust/set_operations.h>
#include <thrust/execution_policy.h>
...
int A_keys[6] = {0, 2, 4, 6, 8, 10, 12};
int A_vals[6] = {0, 0, 0, 0, 0,  0,  0};

int B_keys[5] = {1, 3, 5, 7, 9};
int B_vals[5] = {1, 1, 1, 1, 1};

int keys_result[11];
int vals_result[11];

thrust::pair<int*,int*> end = thrust::set_symmetric_difference_by_key(thrust::host, A_keys, A_keys + 6, B_keys, B_keys + 5, A_vals, B_vals, keys_result, vals_result);
// keys_result is now {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12}
// vals_result is now {0, 1, 0, 1, 0, 1, 0, 1, 0, 1,  0,  0}
```

**Template Parameters**:
* **`DerivedPolicy`** The name of the derived execution policy. 
* **`InputIterator1`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>, <code>InputIterator1</code> and <code>InputIterator2</code> have the same <code>value&#95;type</code>, <code>InputIterator1's</code><code>value&#95;type</code> is a model of <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a>, the ordering on <code>InputIterator1's</code><code>value&#95;type</code> is a strict weak ordering, as defined in the <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a> requirements, and <code>InputIterator1's</code><code>value&#95;type</code> is convertable to a type in <code>OutputIterator's</code> set of <code>value&#95;types</code>. 
* **`InputIterator2`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>, <code>InputIterator2</code> and <code>InputIterator1</code> have the same <code>value&#95;type</code>, <code>InputIterator2's</code><code>value&#95;type</code> is a model of <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a>, the ordering on <code>InputIterator2's</code><code>value&#95;type</code> is a strict weak ordering, as defined in the <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a> requirements, and <code>InputIterator2's</code><code>value&#95;type</code> is convertable to a type in <code>OutputIterator's</code> set of <code>value&#95;types</code>. 
* **`InputIterator3`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>, and <code>InputIterator3's</code><code>value&#95;type</code> is convertible to a type in <code>OutputIterator2's</code> set of <code>value&#95;types</code>. 
* **`InputIterator4`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>, and <code>InputIterator4's</code><code>value&#95;type</code> is convertible to a type in <code>OutputIterator2's</code> set of <code>value&#95;types</code>. 
* **`OutputIterator1`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>. 
* **`OutputIterator2`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>.

**Function Parameters**:
* **`exec`** The execution policy to use for parallelization. 
* **`keys_first1`** The beginning of the first input range of keys. 
* **`keys_last1`** The end of the first input range of keys. 
* **`keys_first2`** The beginning of the second input range of keys. 
* **`keys_last2`** The end of the second input range of keys. 
* **`values_first1`** The beginning of the first input range of values. 
* **`values_first2`** The beginning of the first input range of values. 
* **`keys_result`** The beginning of the output range of keys. 
* **`values_result`** The beginning of the output range of values. 

**Preconditions**:
* The ranges <code>[keys&#95;first1, keys&#95;last1)</code> and <code>[keys&#95;first2, keys&#95;last2)</code> shall be sorted with respect to <code>operator&lt;</code>. 
* The resulting ranges shall not overlap with any input range.

**Returns**:
A <code>pair</code><code>p</code> such that <code>p.first</code> is the end of the output range of keys, and such that <code>p.second</code> is the end of the output range of values.

**See**:
* <code>set&#95;symmetric&#95;difference&#95;by&#95;key</code>
* <code>set&#95;intersection&#95;by&#95;key</code>
* <code>set&#95;difference&#95;by&#95;key</code>
* <code>sort&#95;by&#95;key</code>
* <code>is&#95;sorted</code>

<h3 id="function-set-union-by-key">
Function <code>thrust::set&#95;union&#95;by&#95;key</code>
</h3>

<code class="doxybook">
<span>template &lt;typename InputIterator1,</span>
<span>&nbsp;&nbsp;typename InputIterator2,</span>
<span>&nbsp;&nbsp;typename InputIterator3,</span>
<span>&nbsp;&nbsp;typename InputIterator4,</span>
<span>&nbsp;&nbsp;typename OutputIterator1,</span>
<span>&nbsp;&nbsp;typename OutputIterator2&gt;</span>
<span><a href="{{ site.baseurl }}/api/classes/structthrust_1_1pair.html">thrust::pair</a>< OutputIterator1, OutputIterator2 > </span><span><b>set_union_by_key</b>(InputIterator1 keys_first1,</span>
<span>&nbsp;&nbsp;InputIterator1 keys_last1,</span>
<span>&nbsp;&nbsp;InputIterator2 keys_first2,</span>
<span>&nbsp;&nbsp;InputIterator2 keys_last2,</span>
<span>&nbsp;&nbsp;InputIterator3 values_first1,</span>
<span>&nbsp;&nbsp;InputIterator4 values_first2,</span>
<span>&nbsp;&nbsp;OutputIterator1 keys_result,</span>
<span>&nbsp;&nbsp;OutputIterator2 values_result);</span></code>
<code>set&#95;union&#95;by&#95;key</code> performs a key-value union operation from set theory. <code>set&#95;union&#95;by&#95;key</code> constructs a sorted range that is the union of the sorted ranges <code>[keys&#95;first1, keys&#95;last1)</code> and <code>[keys&#95;first2, keys&#95;last2)</code>. Associated with each element from the input and output key ranges is a value element. The associated input value ranges need not be sorted.

In the simplest case, <code>set&#95;union&#95;by&#95;key</code> performs the "union" operation from set theory: the output range contains a copy of every element that is contained in <code>[keys&#95;first1, keys&#95;last1)</code>, <code>[keys&#95;first2, keys&#95;last1)</code>, or both. The general case is more complicated, because the input ranges may contain duplicate elements. The generalization is that if <code>[keys&#95;first1, keys&#95;last1)</code> contains <code>m</code> elements that are equivalent to each other and if <code>[keys&#95;first2, keys&#95;last2)</code> contains <code>n</code> elements that are equivalent to them, then all <code>m</code> elements from the first range shall be copied to the output range, in order, and then <code>max(n - m, 0)</code> elements from the second range shall be copied to the output, in order.

Each time a key element is copied from <code>[keys&#95;first1, keys&#95;last1)</code> or <code>[keys&#95;first2, keys&#95;last2)</code> is copied to the keys output range, the corresponding value element is copied from the corresponding values input range (beginning at <code>values&#95;first1</code> or <code>values&#95;first2</code>) to the values output range.

This version of <code>set&#95;union&#95;by&#95;key</code> compares key elements using <code>operator&lt;</code>.


The following code snippet demonstrates how to use <code>set&#95;symmetric&#95;difference&#95;by&#95;key</code> to compute the symmetric difference of two sets of integers sorted in ascending order with their values.



```cpp
#include <thrust/set_operations.h>
...
int A_keys[6] = {0, 2, 4, 6, 8, 10, 12};
int A_vals[6] = {0, 0, 0, 0, 0,  0,  0};

int B_keys[5] = {1, 3, 5, 7, 9};
int B_vals[5] = {1, 1, 1, 1, 1};

int keys_result[11];
int vals_result[11];

thrust::pair<int*,int*> end = thrust::set_symmetric_difference_by_key(A_keys, A_keys + 6, B_keys, B_keys + 5, A_vals, B_vals, keys_result, vals_result);
// keys_result is now {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12}
// vals_result is now {0, 1, 0, 1, 0, 1, 0, 1, 0, 1,  0,  0}
```

**Template Parameters**:
* **`InputIterator1`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>, <code>InputIterator1</code> and <code>InputIterator2</code> have the same <code>value&#95;type</code>, <code>InputIterator1's</code><code>value&#95;type</code> is a model of <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a>, the ordering on <code>InputIterator1's</code><code>value&#95;type</code> is a strict weak ordering, as defined in the <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a> requirements, and <code>InputIterator1's</code><code>value&#95;type</code> is convertable to a type in <code>OutputIterator's</code> set of <code>value&#95;types</code>. 
* **`InputIterator2`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>, <code>InputIterator2</code> and <code>InputIterator1</code> have the same <code>value&#95;type</code>, <code>InputIterator2's</code><code>value&#95;type</code> is a model of <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a>, the ordering on <code>InputIterator2's</code><code>value&#95;type</code> is a strict weak ordering, as defined in the <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a> requirements, and <code>InputIterator2's</code><code>value&#95;type</code> is convertable to a type in <code>OutputIterator's</code> set of <code>value&#95;types</code>. 
* **`InputIterator3`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>, and <code>InputIterator3's</code><code>value&#95;type</code> is convertible to a type in <code>OutputIterator2's</code> set of <code>value&#95;types</code>. 
* **`InputIterator4`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>, and <code>InputIterator4's</code><code>value&#95;type</code> is convertible to a type in <code>OutputIterator2's</code> set of <code>value&#95;types</code>. 
* **`OutputIterator1`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>. 
* **`OutputIterator2`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>.

**Function Parameters**:
* **`keys_first1`** The beginning of the first input range of keys. 
* **`keys_last1`** The end of the first input range of keys. 
* **`keys_first2`** The beginning of the second input range of keys. 
* **`keys_last2`** The end of the second input range of keys. 
* **`values_first1`** The beginning of the first input range of values. 
* **`values_first2`** The beginning of the first input range of values. 
* **`keys_result`** The beginning of the output range of keys. 
* **`values_result`** The beginning of the output range of values. 

**Preconditions**:
* The ranges <code>[keys&#95;first1, keys&#95;last1)</code> and <code>[keys&#95;first2, keys&#95;last2)</code> shall be sorted with respect to <code>operator&lt;</code>. 
* The resulting ranges shall not overlap with any input range.

**Returns**:
A <code>pair</code><code>p</code> such that <code>p.first</code> is the end of the output range of keys, and such that <code>p.second</code> is the end of the output range of values.

**See**:
* <code>set&#95;symmetric&#95;difference&#95;by&#95;key</code>
* <code>set&#95;intersection&#95;by&#95;key</code>
* <code>set&#95;difference&#95;by&#95;key</code>
* <code>sort&#95;by&#95;key</code>
* <code>is&#95;sorted</code>

<h3 id="function-set-union-by-key">
Function <code>thrust::set&#95;union&#95;by&#95;key</code>
</h3>

<code class="doxybook">
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename InputIterator1,</span>
<span>&nbsp;&nbsp;typename InputIterator2,</span>
<span>&nbsp;&nbsp;typename InputIterator3,</span>
<span>&nbsp;&nbsp;typename InputIterator4,</span>
<span>&nbsp;&nbsp;typename OutputIterator1,</span>
<span>&nbsp;&nbsp;typename OutputIterator2,</span>
<span>&nbsp;&nbsp;typename StrictWeakCompare&gt;</span>
<span>__host__ __device__ <a href="{{ site.baseurl }}/api/classes/structthrust_1_1pair.html">thrust::pair</a>< OutputIterator1, OutputIterator2 > </span><span><b>set_union_by_key</b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;InputIterator1 keys_first1,</span>
<span>&nbsp;&nbsp;InputIterator1 keys_last1,</span>
<span>&nbsp;&nbsp;InputIterator2 keys_first2,</span>
<span>&nbsp;&nbsp;InputIterator2 keys_last2,</span>
<span>&nbsp;&nbsp;InputIterator3 values_first1,</span>
<span>&nbsp;&nbsp;InputIterator4 values_first2,</span>
<span>&nbsp;&nbsp;OutputIterator1 keys_result,</span>
<span>&nbsp;&nbsp;OutputIterator2 values_result,</span>
<span>&nbsp;&nbsp;StrictWeakCompare comp);</span></code>
<code>set&#95;union&#95;by&#95;key</code> performs a key-value union operation from set theory. <code>set&#95;union&#95;by&#95;key</code> constructs a sorted range that is the union of the sorted ranges <code>[keys&#95;first1, keys&#95;last1)</code> and <code>[keys&#95;first2, keys&#95;last2)</code>. Associated with each element from the input and output key ranges is a value element. The associated input value ranges need not be sorted.

In the simplest case, <code>set&#95;union&#95;by&#95;key</code> performs the "union" operation from set theory: the output range contains a copy of every element that is contained in <code>[keys&#95;first1, keys&#95;last1)</code>, <code>[keys&#95;first2, keys&#95;last1)</code>, or both. The general case is more complicated, because the input ranges may contain duplicate elements. The generalization is that if <code>[keys&#95;first1, keys&#95;last1)</code> contains <code>m</code> elements that are equivalent to each other and if <code>[keys&#95;first2, keys&#95;last2)</code> contains <code>n</code> elements that are equivalent to them, then all <code>m</code> elements from the first range shall be copied to the output range, in order, and then <code>max(n - m, 0)</code> elements from the second range shall be copied to the output, in order.

Each time a key element is copied from <code>[keys&#95;first1, keys&#95;last1)</code> or <code>[keys&#95;first2, keys&#95;last2)</code> is copied to the keys output range, the corresponding value element is copied from the corresponding values input range (beginning at <code>values&#95;first1</code> or <code>values&#95;first2</code>) to the values output range.

This version of <code>set&#95;union&#95;by&#95;key</code> compares key elements using a function object <code>comp</code>.

The algorithm's execution is parallelized as determined by <code>exec</code>.


The following code snippet demonstrates how to use <code>set&#95;symmetric&#95;difference&#95;by&#95;key</code> to compute the symmetric difference of two sets of integers sorted in descending order with their values using the <code>thrust::host</code> execution policy for parallelization:



```cpp
#include <thrust/set_operations.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>
...
int A_keys[6] = {12, 10, 8, 6, 4, 2, 0};
int A_vals[6] = { 0,  0, 0, 0, 0, 0, 0};

int B_keys[5] = {9, 7, 5, 3, 1};
int B_vals[5] = {1, 1, 1, 1, 1};

int keys_result[11];
int vals_result[11];

thrust::pair<int*,int*> end = thrust::set_symmetric_difference_by_key(thrust::host, A_keys, A_keys + 6, B_keys, B_keys + 5, A_vals, B_vals, keys_result, vals_result, thrust::greater<int>());
// keys_result is now {12, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0}
// vals_result is now { 0,  1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0}
```

**Template Parameters**:
* **`DerivedPolicy`** The name of the derived execution policy. 
* **`InputIterator1`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>, <code>InputIterator1</code> and <code>InputIterator2</code> have the same <code>value&#95;type</code>, <code>InputIterator1's</code><code>value&#95;type</code> is a model of <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a>, the ordering on <code>InputIterator1's</code><code>value&#95;type</code> is a strict weak ordering, as defined in the <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a> requirements, and <code>InputIterator1's</code><code>value&#95;type</code> is convertable to a type in <code>OutputIterator's</code> set of <code>value&#95;types</code>. 
* **`InputIterator2`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>, <code>InputIterator2</code> and <code>InputIterator1</code> have the same <code>value&#95;type</code>, <code>InputIterator2's</code><code>value&#95;type</code> is a model of <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a>, the ordering on <code>InputIterator2's</code><code>value&#95;type</code> is a strict weak ordering, as defined in the <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a> requirements, and <code>InputIterator2's</code><code>value&#95;type</code> is convertable to a type in <code>OutputIterator's</code> set of <code>value&#95;types</code>. 
* **`InputIterator3`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>, and <code>InputIterator3's</code><code>value&#95;type</code> is convertible to a type in <code>OutputIterator2's</code> set of <code>value&#95;types</code>. 
* **`InputIterator4`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>, and <code>InputIterator4's</code><code>value&#95;type</code> is convertible to a type in <code>OutputIterator2's</code> set of <code>value&#95;types</code>. 
* **`OutputIterator1`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>. 
* **`OutputIterator2`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>. 
* **`StrictWeakCompare`** is a model of <a href="https://en.cppreference.com/w/cpp/concepts/strict_weak_order">Strict Weak Ordering</a>.

**Function Parameters**:
* **`exec`** The execution policy to use for parallelization. 
* **`keys_first1`** The beginning of the first input range of keys. 
* **`keys_last1`** The end of the first input range of keys. 
* **`keys_first2`** The beginning of the second input range of keys. 
* **`keys_last2`** The end of the second input range of keys. 
* **`values_first1`** The beginning of the first input range of values. 
* **`values_first2`** The beginning of the first input range of values. 
* **`keys_result`** The beginning of the output range of keys. 
* **`values_result`** The beginning of the output range of values. 
* **`comp`** Comparison operator. 

**Preconditions**:
* The ranges <code>[keys&#95;first1, keys&#95;last1)</code> and <code>[keys&#95;first2, keys&#95;last2)</code> shall be sorted with respect to <code>comp</code>. 
* The resulting ranges shall not overlap with any input range.

**Returns**:
A <code>pair</code><code>p</code> such that <code>p.first</code> is the end of the output range of keys, and such that <code>p.second</code> is the end of the output range of values.

**See**:
* <code>set&#95;symmetric&#95;difference&#95;by&#95;key</code>
* <code>set&#95;intersection&#95;by&#95;key</code>
* <code>set&#95;difference&#95;by&#95;key</code>
* <code>sort&#95;by&#95;key</code>
* <code>is&#95;sorted</code>

<h3 id="function-set-union-by-key">
Function <code>thrust::set&#95;union&#95;by&#95;key</code>
</h3>

<code class="doxybook">
<span>template &lt;typename InputIterator1,</span>
<span>&nbsp;&nbsp;typename InputIterator2,</span>
<span>&nbsp;&nbsp;typename InputIterator3,</span>
<span>&nbsp;&nbsp;typename InputIterator4,</span>
<span>&nbsp;&nbsp;typename OutputIterator1,</span>
<span>&nbsp;&nbsp;typename OutputIterator2,</span>
<span>&nbsp;&nbsp;typename StrictWeakCompare&gt;</span>
<span><a href="{{ site.baseurl }}/api/classes/structthrust_1_1pair.html">thrust::pair</a>< OutputIterator1, OutputIterator2 > </span><span><b>set_union_by_key</b>(InputIterator1 keys_first1,</span>
<span>&nbsp;&nbsp;InputIterator1 keys_last1,</span>
<span>&nbsp;&nbsp;InputIterator2 keys_first2,</span>
<span>&nbsp;&nbsp;InputIterator2 keys_last2,</span>
<span>&nbsp;&nbsp;InputIterator3 values_first1,</span>
<span>&nbsp;&nbsp;InputIterator4 values_first2,</span>
<span>&nbsp;&nbsp;OutputIterator1 keys_result,</span>
<span>&nbsp;&nbsp;OutputIterator2 values_result,</span>
<span>&nbsp;&nbsp;StrictWeakCompare comp);</span></code>
<code>set&#95;union&#95;by&#95;key</code> performs a key-value union operation from set theory. <code>set&#95;union&#95;by&#95;key</code> constructs a sorted range that is the union of the sorted ranges <code>[keys&#95;first1, keys&#95;last1)</code> and <code>[keys&#95;first2, keys&#95;last2)</code>. Associated with each element from the input and output key ranges is a value element. The associated input value ranges need not be sorted.

In the simplest case, <code>set&#95;union&#95;by&#95;key</code> performs the "union" operation from set theory: the output range contains a copy of every element that is contained in <code>[keys&#95;first1, keys&#95;last1)</code>, <code>[keys&#95;first2, keys&#95;last1)</code>, or both. The general case is more complicated, because the input ranges may contain duplicate elements. The generalization is that if <code>[keys&#95;first1, keys&#95;last1)</code> contains <code>m</code> elements that are equivalent to each other and if <code>[keys&#95;first2, keys&#95;last2)</code> contains <code>n</code> elements that are equivalent to them, then all <code>m</code> elements from the first range shall be copied to the output range, in order, and then <code>max(n - m, 0)</code> elements from the second range shall be copied to the output, in order.

Each time a key element is copied from <code>[keys&#95;first1, keys&#95;last1)</code> or <code>[keys&#95;first2, keys&#95;last2)</code> is copied to the keys output range, the corresponding value element is copied from the corresponding values input range (beginning at <code>values&#95;first1</code> or <code>values&#95;first2</code>) to the values output range.

This version of <code>set&#95;union&#95;by&#95;key</code> compares key elements using a function object <code>comp</code>.


The following code snippet demonstrates how to use <code>set&#95;symmetric&#95;difference&#95;by&#95;key</code> to compute the symmetric difference of two sets of integers sorted in descending order with their values.



```cpp
#include <thrust/set_operations.h>
#include <thrust/functional.h>
...
int A_keys[6] = {12, 10, 8, 6, 4, 2, 0};
int A_vals[6] = { 0,  0, 0, 0, 0, 0, 0};

int B_keys[5] = {9, 7, 5, 3, 1};
int B_vals[5] = {1, 1, 1, 1, 1};

int keys_result[11];
int vals_result[11];

thrust::pair<int*,int*> end = thrust::set_symmetric_difference_by_key(A_keys, A_keys + 6, B_keys, B_keys + 5, A_vals, B_vals, keys_result, vals_result, thrust::greater<int>());
// keys_result is now {12, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0}
// vals_result is now { 0,  1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0}
```

**Template Parameters**:
* **`InputIterator1`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>, <code>InputIterator1</code> and <code>InputIterator2</code> have the same <code>value&#95;type</code>, <code>InputIterator1's</code><code>value&#95;type</code> is a model of <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a>, the ordering on <code>InputIterator1's</code><code>value&#95;type</code> is a strict weak ordering, as defined in the <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a> requirements, and <code>InputIterator1's</code><code>value&#95;type</code> is convertable to a type in <code>OutputIterator's</code> set of <code>value&#95;types</code>. 
* **`InputIterator2`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>, <code>InputIterator2</code> and <code>InputIterator1</code> have the same <code>value&#95;type</code>, <code>InputIterator2's</code><code>value&#95;type</code> is a model of <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a>, the ordering on <code>InputIterator2's</code><code>value&#95;type</code> is a strict weak ordering, as defined in the <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a> requirements, and <code>InputIterator2's</code><code>value&#95;type</code> is convertable to a type in <code>OutputIterator's</code> set of <code>value&#95;types</code>. 
* **`InputIterator3`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>, and <code>InputIterator3's</code><code>value&#95;type</code> is convertible to a type in <code>OutputIterator2's</code> set of <code>value&#95;types</code>. 
* **`InputIterator4`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>, and <code>InputIterator4's</code><code>value&#95;type</code> is convertible to a type in <code>OutputIterator2's</code> set of <code>value&#95;types</code>. 
* **`OutputIterator1`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>. 
* **`OutputIterator2`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>. 
* **`StrictWeakCompare`** is a model of <a href="https://en.cppreference.com/w/cpp/concepts/strict_weak_order">Strict Weak Ordering</a>.

**Function Parameters**:
* **`keys_first1`** The beginning of the first input range of keys. 
* **`keys_last1`** The end of the first input range of keys. 
* **`keys_first2`** The beginning of the second input range of keys. 
* **`keys_last2`** The end of the second input range of keys. 
* **`values_first1`** The beginning of the first input range of values. 
* **`values_first2`** The beginning of the first input range of values. 
* **`keys_result`** The beginning of the output range of keys. 
* **`values_result`** The beginning of the output range of values. 
* **`comp`** Comparison operator. 

**Preconditions**:
* The ranges <code>[keys&#95;first1, keys&#95;last1)</code> and <code>[keys&#95;first2, keys&#95;last2)</code> shall be sorted with respect to <code>comp</code>. 
* The resulting ranges shall not overlap with any input range.

**Returns**:
A <code>pair</code><code>p</code> such that <code>p.first</code> is the end of the output range of keys, and such that <code>p.second</code> is the end of the output range of values.

**See**:
* <code>set&#95;symmetric&#95;difference&#95;by&#95;key</code>
* <code>set&#95;intersection&#95;by&#95;key</code>
* <code>set&#95;difference&#95;by&#95;key</code>
* <code>sort&#95;by&#95;key</code>
* <code>is&#95;sorted</code>


