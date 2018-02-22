# coding=utf-8
# Prints profile report generated after a run like:
# > time python -m cProfile -o profile.out audfprint.py match -d ../uspop+gtzan-dflt/data.fpdb --density 70 --fanout 8 --search-depth 100 --min-count 20 --ncores 1 --verbose 1 --opfile u+g-s500-m20-v.out3 --list ../100queries.list

import pstats
import sys

prof_file = 'profile.out'
if len(sys.argv) > 1:
    prof_file = sys.argv[1]
p = pstats.Stats(prof_file)
p.sort_stats('time')
p.print_stats(200)
