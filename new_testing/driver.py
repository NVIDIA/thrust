import glob
import os

def generate_tests(modulename):
    print "Loading module",modulename
     
    module = __import__(modulename)

    names = [name for name in dir(module) if name.startswith('Test')]
        
    generators = [getattr(module, name)() for name in names]

    for (name,generator) in zip(names,generators):
        print "Generating tests for %s" % (name,)
            
        fid = open(name + '.cu', 'w')
        
        fid.write('#include "%s.h"\n\n' % (modulename,))
    
        for n,test in enumerate(generator):
            fid.write('void %s%d(void)\n{\n' % (name,n))
            fid.write(test)
            fid.write('\n}\n\n')

if __name__ == '__main__':
    folders    = ['']

    for folder in folders:
        for filename in glob.glob(os.path.join(folder, "*.py")):
            if filename == __file__: continue # ignore this file

            #TODO handle subdirectories
            modulename = os.path.splitext(filename)[0]

            generate_tests(modulename)

