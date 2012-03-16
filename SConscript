Import('env')

# create the thrust zip
# note that we package the CHANGELOG with the headers
installation_nodes = env.Install('thrust', 'CHANGELOG')
env.Zip('thrust.zip', 'thrust')

# create the examples zip
env.Zip('examples.zip', 'examples')

