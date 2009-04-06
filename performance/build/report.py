from build import parse_testsuite_xml

__all__ = ['plot_results']

#TODO add print_results which outputs a CSV file

def full_label(name):
    known_labels = {'Throughput' : 'Throughput (GOp/s)',
                    'Sorting'    : 'Sorting Rate (MKey/s)',
                    'Bandwidth'  : 'Memory Bandwidth (GByte/s)',
                    'InputSize'  : 'Input Size',
                    'KeyType'    : 'Key Type' }

    if name in known_labels:
        return known_labels[name]
    else:
        return name


def plot_results(input_file, series_key, x_axis, y_axis, plot='loglog', dpi=72, title=None, format=None):
    """Plot performance data stored in an XML file

    if format is None then the figure is shown, otherwise it is 
    written to a file with the specified extension

    Example
    -------
    input_file = 'reduce.xml'
    series_key = 'InputType'
    x_axis = 'InputSize'
    y_axis = 'Throughput'
    format = 'pdf'
    """

    try:
        fid = open(input_file)
    except IOError:
        print "unable to open file '%s'" % input_file
        return

    TS = parse_testsuite_xml(fid)
    
    series_titles = set([test.variables[series_key] for (testname,test) in TS.tests.items()])
    series = dict( zip(series_titles, [list() for s_title in series_titles]) )
    
    for testname,test in TS.tests.items():
        if x_axis in test.variables and y_axis in test.results:
            series[test.variables[series_key]].append( (test.variables[x_axis], test.results[y_axis]) )
    

    if title is None:
        title = TS.name

    import pylab
    
    pylab.figure()
    pylab.title(title)
    pylab.xlabel(full_label(x_axis))
    pylab.ylabel(full_label(y_axis))

    plotter = getattr(pylab, plot) 
    for series_title,series_data in series.items():
        series_data.sort()
        x_values = [val[0] for val in series_data]
        y_values = [val[1] for val in series_data]
   
        plotter(x_values, y_values, label=series_title)

    if len(series) >= 2:
        pylab.legend(loc=0)
   
    if format is None:
        pylab.show()    
    else:
        import os
        fname = os.path.splitext(input_file)[0] + '.' + format
        pylab.savefig(fname, dpi=dpi)
