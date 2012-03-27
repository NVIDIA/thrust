import os
import re
Import('env')

# clone the environment so as not to pollute the parent
my_env = env.Clone()

# divine the version number from thrust/version.h
version = int(re.search('THRUST_VERSION ([0-9]+)', File('#thrust/version.h').get_contents()).group(1))
major   = int(version / 100000)
minor   = int(version / 100) % 1000
subminor = version % 100

# create the Thrust zip
for item in my_env.RecursiveGlob('*', '#thrust'):
  my_env.InstallAs(os.path.join('thrust', Dir('#thrust').rel_path(item)), item)
# grab the CHANGELOG as well
my_env.Install('thrust', '#CHANGELOG')

# make sure to change directory into the variant dir to ensure the paths are correct in the zipfile
# note Zip uses the special site_scons/site_tools/zip.py to WAR an issue with the chdir parameter
thrust_zipfile = my_env.Zip('thrust-{0}.{1}.{2}.zip'.format(major,minor,subminor), 'thrust', chdir = 1)
my_env.Alias('dist', thrust_zipfile)


# create the examples zip
# do not recurse into the 'targets' directory, should it exist
for item in my_env.RecursiveGlob('*', '#examples', 'targets'):
  # avoid included SCons-related files in the distribution
  # XXX would be nice if we could ignore all dotfiles and anything in .gitignore
  if item.get_path(item.get_dir()) not in ['SConscript','.sconsign.dblite']:
    my_env.InstallAs(os.path.join('examples', Dir('#examples').rel_path(item)), item)
# make sure to change directory into the variant dir to ensure the paths are correct in the zipfile
# note Zip uses the special site_scons/site_tools/zip.py to WAR an issue with the chdir parameter
examples_zipfile = my_env.Zip('examples-{0}.{1}.zip'.format(major,minor), 'examples', chdir = 1)
my_env.Alias('dist', examples_zipfile)

# generate documentation
# note that thrust.dox instructs doxygen to output to the targets directory
public_headers = my_env.RecursiveGlob('*.h', '#thrust', exclude='detail')
thrust_docs = my_env.Command('doc/html', public_headers, 'doxygen doc/thrust.dox')
my_env.Alias('doc', thrust_docs)

