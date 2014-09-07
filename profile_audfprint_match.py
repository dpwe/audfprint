import audfprint
import cProfile
import pstats

argv = ["audfprint", "match", "-d", "tmp.fpdb", "--density", "200", "query.mp3", "query2.mp3"]

cProfile.run('audfprint.main(argv)', 'fpmstats')

p = pstats.Stats('fpmstats')

p.sort_stats('time')
p.print_stats(10)
