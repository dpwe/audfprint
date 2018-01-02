# coding=utf-8
import cProfile
import pstats

# noinspection PyUnresolvedReferences
import audfprint

argv = ["audfprint", "match", "-d", "fpdbase.pklz", "--density", "200", "query.mp3"]

cProfile.run('audfprint.main(argv)', 'fpmstats')

p = pstats.Stats('fpmstats')

p.sort_stats('time')
p.print_stats(10)
