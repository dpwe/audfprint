#!/usr/bin/env python

# comp_file_lines.py
#
# Python script to count number of exact matching lines between two files, no edit distance
# 2014-09-07 Dan Ellis dpwe@ee.columbia.edu

import sys

verbose = 0

if len(sys.argv) < 2:
    print "Usage:", sys.argv[0], "file1.txt file2.txt [verbose]"

else:
    file1 = sys.argv[1]
    file2 = sys.argv[2]

    if len(sys.argv) > 3:
        verbose = 1

    # Read in the files
    with open(file1) as f:
        item1s = [val.rstrip("\n") for val in f]

    with open(file2) as f:
        item2s = [val.rstrip("\n") for val in f]

    # Now, make a boolean vector of correctness
    import numpy as np
    correct = np.zeros(len(item1s), np.float)
    for ix, items in enumerate(zip(item1s, item2s)):
        if items[0] == items[1]:
            correct[ix] = 1.0
        else:
            if verbose:
                print items

    print int(np.sum(correct)),"correct out of", len(correct), "= %.1f%%" % (100.0*np.mean(correct))
