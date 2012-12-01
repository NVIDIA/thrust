from build import plot_results, print_results

#valid formats are png, pdf, ps, eps and svg
#if format=None the plot will be displayed
format = 'png'
output = print_results
#output = plot_results

for function in ['fill', 'reduce', 'inner_product', 'gather', 'merge']:
    output(function + '.xml', 'InputType', 'InputSize', 'Bandwidth', format=format)

for function in ['inclusive_scan', 'inclusive_segmented_scan', 'unique']:
    output(function + '.xml', 'InputType', 'InputSize', 'Throughput', format=format)

for method in ['indirect_sort']:
    output(method + '.xml',    'Sort', 'VectorLength', 'Time', plot='semilogx', title='Indirect Sorting', format=format)

for method in ['sort', 'comparison_sort', 'radix_sort']:
    output(method + '.xml',    'KeyType', 'InputSize', 'Sorting', title='thrust::' + method, format=format)
    output(method + '_by_key.xml', 'KeyType', 'InputSize', 'Sorting', title='thrust::' + method + '_by_key', format=format)

for method in ['set_difference', 'set_intersection', 'set_symmetric_difference', 'set_union']:
  output(method + '.xml', 'InputType', 'InputSize', 'Sorting', title='thrust::' + method, format=format)
    
output('stl_sort.xml', 'KeyType', 'InputSize', 'Sorting', title='std::sort', format=format)

for method in ['radix_sort']:
    output(method + '_bits.xml', 'KeyType', 'KeyBits', 'Sorting', title='thrust::' + method, plot='plot', dpi=72, format=format)

for format in ['png', 'pdf']:
    output('reduce_float.xml', 'InputType', 'InputSize', 'Bandwidth', dpi=120, plot='semilogx', title='thrust::reduce<float>()', format=format)
    output('sort_large.xml',  'KeyType', 'InputSize', 'Sorting', dpi=120, plot='semilogx', title='thrust::sort<T>()', format=format)

