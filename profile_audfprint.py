import audfprint
import cProfile
import pstats

argv = ["audfprint", "new", "-d", "tmp.fpdb", "--density", "200", "Nine_Lives/01-Nine_Lives.mp3", "Nine_Lives/02-Falling_In_Love.mp3", "Nine_Lives/03-Hole_In_My_Soul.mp3", "Nine_Lives/04-Taste_Of_India.mp3", "Nine_Lives/05-Full_Circle.mp3", "Nine_Lives/06-Something_s_Gotta_Give.mp3", "Nine_Lives/07-Ain_t_That_A_Bitch.mp3", "Nine_Lives/08-The_Farm.mp3", "Nine_Lives/09-Crash.mp3", "Nine_Lives/10-Kiss_Your_Past_Good-bye.mp3", "Nine_Lives/11-Pink.mp3", "Nine_Lives/12-Attitude_Adjustment.mp3", "Nine_Lives/13-Fallen_Angels.mp3"]

cProfile.run('audfprint.main(argv)', 'fpstats')

p = pstats.Stats('fpstats')

p.sort_stats('time')
p.print_stats(10)
