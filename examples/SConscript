import os

Import('env')

# create a clone of the environment so that we don't alter the parent
my_env = env.Clone()

# find all .cus & .cpps in the current directory
sources = []
directories = ['.']

# find all .cus & .cpps in the current directory
sources = []
directories = ['.', my_env['device_backend']]
extensions = ['.cu','.cpp']

for dir in directories:
  for ext in extensions:
    regex = os.path.join(dir, '*' + ext)
    sources.extend(my_env.Glob(regex))

# compile examples
for src in sources:
  program = my_env.Program(src)
  # add the program to the 'run_examples' alias
  program_alias = my_env.Alias('run_examples', [program], program[0].abspath)
  # always build the 'run_examples' target whether or not it needs it
  my_env.AlwaysBuild(program_alias)

