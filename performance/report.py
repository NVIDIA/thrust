from build import plot_results

#valid formats are png, pdf, ps, eps and svg
#if format=None the plot will be displayed
format = 'png'

for function in ['fill', 'reduce', 'inner_product']:
    plot_results(function + '.xml', 'InputType', 'InputSize', 'Bandwidth', format=format)

for function in ['inclusive_scan', 'unique']:
    plot_results(function + '.xml', 'InputType', 'InputSize', 'Throughput', format=format)

for method in ['sort', 'merge_sort', 'radix_sort']:
    plot_results(method + '.xml',    'KeyType', 'InputSize', 'Sorting', format=format)
    plot_results(method + '_by_key.xml', 'KeyType', 'InputSize', 'Sorting', format=format)

for format in ['png', 'pdf']:
    plot_results('reduce_float.xml', 'InputType', 'InputSize', 'Bandwidth', dpi=120, plot='semilogx', title='thrust::reduce<float>()', format=format)
    plot_results('sort_large.xml',  'KeyType', 'InputSize', 'Sorting', dpi=120, plot='semilogx', title='thrust::sort<T>()', format=format)

