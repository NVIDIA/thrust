#!/usr/bin/env python

"""In order to run performance tests in Eris, we create this script to
1) Run the benchmark app multiple times and report the average score
2) Print Eris style banner '&&&& PERF' so it can be parsed by Eris."""

import argparse
import os
import sys
import csv
import subprocess

TEST_NAME = "bench"
OUTPUT_FILE_NAME = lambda i: TEST_NAME + "_" + str(i) + ".csv"
COMBINED_OUTPUT_FILE_NAME = TEST_NAME + "_combined.csv"
POSTPROCESS_NAME = "combine_benchmark_results.py"

parser = argparse.ArgumentParser(description='ERIS wrapper script for Thrust benchmarks')
parser.add_argument(
  '-n', '--numloops', default=5, type=int,
  metavar='N', help='Run the benchmark N times.'
)
args = parser.parse_args()

print '&&&& RUNNING {0}'.format(TEST_NAME)
assert args.numloops > 0
test_cmd = os.path.join(os.path.dirname(os.path.realpath(__file__)), TEST_NAME)

for i in xrange(args.numloops):
    with open(OUTPUT_FILE_NAME(i), "w") as output_file:
      print '#### RUN {0} -> {1}'.format(i, OUTPUT_FILE_NAME(i))

      p = None

      try:
          p = subprocess.Popen(test_cmd, stdout=output_file, stderr=output_file)
          p.communicate()
      except OSError as ex:
          with open(OUTPUT_FILE_NAME(i)) as error_file:
            for line in error_file:
              print line,
          print '#### ERROR : Caught OSError `{0}`.'.format(ex)
          print '&&&& FAILED {0}'.format(TEST_NAME)
          sys.exit(-1)

    with open(OUTPUT_FILE_NAME(i)) as input_file:
      for line in input_file:
        print line,

    if p.returncode != 0:
        print '#### ERROR : Process exited with code {0}.'.format(p.returncode)
        print '&&&& FAILED {0} {1}'.format(TEST_NAME, POSTPROCESS_NAME)
        sys.exit(p.returncode)

print '&&&& PASSED {0}'.format(TEST_NAME)

post_cmd = [os.path.join(os.path.dirname(os.path.realpath(__file__)), POSTPROCESS_NAME)]

post_cmd += ["--dependent-variable=STL Average Walltime,STL Walltime Uncertainty,STL Trials"]
post_cmd += ["--dependent-variable=STL Average Throughput,STL Throughput Uncertainty,STL Trials"]
post_cmd += ["--dependent-variable=Thrust Average Walltime,Thrust Walltime Uncertainty,Thrust Trials"]
post_cmd += ["--dependent-variable=Thrust Average Throughput,Thrust Throughput Uncertainty,Thrust Trials"]

post_cmd += [OUTPUT_FILE_NAME(i) for i in range(args.numloops)] 

printable_cmd = ' '.join(map(lambda e: '"' + str(e) + '"', post_cmd))
print '&&&& RUNNING {0}'.format(printable_cmd)

with open(COMBINED_OUTPUT_FILE_NAME, "w") as output_file:
    p = None

    try:
        p = subprocess.Popen(post_cmd, stdout=output_file, stderr=output_file)
        p.communicate()
    except OSError as ex:
        with open(COMBINED_OUTPUT_FILE_NAME) as error_file:
          for line in error_file:
            print line,
        print '#### ERROR : Caught OSError `{0}`.'.format(ex)
        print '&&&& FAILED {0}'.format(printable_cmd)
        sys.exit(-1)

    with open(COMBINED_OUTPUT_FILE_NAME) as input_file:
      for line in input_file:
        print line,

    if p.returncode != 0:
        print '#### ERROR : Process exited with code {0}.'.format(p.returncode)
        print '&&&& FAILED {0}'.format(printable_cmd)
        sys.exit(p.returncode)

    with open(COMBINED_OUTPUT_FILE_NAME) as input_file:
      reader = csv.DictReader(input_file)

      variable_units = reader.next() # Get units header row

      distinguishing_variables = reader.fieldnames

      measured_variables = [
        ("STL Average Walltime",      "-"),
        ("STL Average Throughput",    "+"),
        ("Thrust Average Walltime",   "-"),
        ("Thrust Average Throughput", "+")
      ]

      for record in reader:
        for variable, directionality in measured_variables:
          print "&&&& PERF {0}_{1}_{2}bit_{3}mib_{4} {5} {6}{7}".format(
            record["Algorithm"],
            record["Element Type"],
            record["Element Size"],
            record["Total Input Size"],
            variable.replace(" ", "_").lower(),
            record[variable],
            directionality,
            variable_units[variable]
          )
                  
print '&&&& PASSED {0}'.format(printable_cmd)

