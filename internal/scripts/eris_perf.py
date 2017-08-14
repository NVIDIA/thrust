#!/usr/bin/env python

"""In order to run performance tests in Eris, we create this script to
1) Run the benchmark app multiple times and report the average score
2) Print Eris style banner '&&&& PERF' so it can be parsed by Eris."""

import argparse
import os
import subprocess
from collections import defaultdict

TEST_NAME = 'bench'


def collect_perf_data(text, scores):
    test_prefix = ''
    for line in text.splitlines():
        if 'Performance' in line:
            test_prefix = line.split('(')[0].replace(' ', '').replace('-', '')
        elif 'Benchmarking with input size' not in line and 'Thrust' not in line:
            # An example test log snippet
            # Core Primitive Performance for 32-bit integer (elements per second)
            #       Algorithm,          STL,    TBB (n/a),       Thrust
            #          reduce,   4546060288,            0,  27218771968

            # We concatenate the generic target name and the algorithm
            # name as the perf subtest name. The fourth column is the
            # score of Thrust implementation.
            test_name = test_prefix + '_' + line.split(',')[0].strip()
            score = int(line.split(',')[3].strip())
            scores[test_name] += score


def dump_perf_results(scores, numloops):
    print 'Performance result in compact view:'
    for (test_name, score) in sorted(scores.items()):
        print '&&&& PERF {0} {1} {2}'.format(test_name,
                                             float(score) / numloops,
                                             'elementsPerSecond')


def main():
    parser = argparse.ArgumentParser(description='Wrapper test script for Thrust benchmark app')
    parser.add_argument(
            '-n', '--numloops', default=5, type=int,
            metavar='N', help='Run the benchmark for N times')
    args = parser.parse_args()

    print '&&&& RUNNING {0}'.format(TEST_NAME)
    assert args.numloops > 0
    test_cmd = os.path.join(os.path.dirname(os.path.realpath(__file__)), TEST_NAME)
    scores = defaultdict(float)
    for i in xrange(args.numloops):
        print 'Test loop {0}'.format(i+1)
        p = subprocess.Popen(test_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            out, err = p.communicate()
        except OSError as ex:
            print 'Failed to run Thrust benchmark: {0}'.format(ex)
            print '&&&& FAILED {0}'.format(TEST_NAME)
            return -1

        print out

        try:
            collect_perf_data(out, scores)
        except Exception as ex:
            print 'Failed to parse the performance results from the test output: {0}'.format(ex)
            print '&&&& FAILED {0}'.format(TEST_NAME)
            return -1

    dump_perf_results(scores, args.numloops)
    print '&&&& PASSED {0}'.format(TEST_NAME)


if __name__ == '__main__':
    main()
