import audfprint
import cProfile
import pstats

argv = ["audfprint", "match", "-d", "fpdbase.pklz", "--density", "200", "query.mp3"]

cProfile.run('audfprint.main(argv)', 'fpmstats')

p = pstats.Stats('fpmstats')

p.sort_stats('time')
p.print_stats(10)
