# audfprint/python/Makefile
#
# This is just to test the python version of audfprint by running each command,
# in both single proc and multiproc mode.
#
# You can test python3 compatibility with calls such as:
# make test_onecore AUDFPRINT="python3 audfprint.py --density 100 --skip-existing"
#
# 2014-09-20 Dan Ellis dpwe@ee.columbia.edu

#AUDFPRINT=python audfprint.py --skip-existing --continue-on-error
AUDFPRINT=python audfprint.py --density 100 --skip-existing

test: test_onecore test_onecore_precomp test_onecore_newmerge test_onecore_precomppk test_mucore test_mucore_precomp test_remove
	rm -rf precompdir precompdir_mu
	rm -f fpdbase*.pklz

test_onecore: fpdbase.pklz
	${AUDFPRINT} match --dbase fpdbase.pklz query.mp3

test_remove: fpdbase.pklz
	${AUDFPRINT} remove --dbase fpdbase.pklz Nine_Lives/05-Full_Circle.mp3 Nine_Lives/01-Nine_Lives.mp3
	${AUDFPRINT} list --dbase fpdbase.pklz
	${AUDFPRINT} add --dbase fpdbase.pklz Nine_Lives/01-Nine_Lives.mp3 Nine_Lives/05-Full_Circle.mp3
	${AUDFPRINT} list --dbase fpdbase.pklz
	${AUDFPRINT} match --dbase fpdbase.pklz query.mp3

fpdbase.pklz: audfprint.py audfprint_analyze.py audfprint_match.py hash_table.py
	${AUDFPRINT} new --dbase fpdbase.pklz Nine_Lives/0*.mp3
	${AUDFPRINT} add --dbase fpdbase.pklz Nine_Lives/1*.mp3

test_onecore_precomp: precompdir
	${AUDFPRINT} new --dbase fpdbase0.pklz precompdir/Nine_Lives/0*
	${AUDFPRINT} new --dbase fpdbase1.pklz precompdir/Nine_Lives/1*
	${AUDFPRINT} merge --dbase fpdbase1.pklz fpdbase0.pklz
	${AUDFPRINT} match --dbase fpdbase1.pklz precompdir/query.afpt

test_onecore_newmerge: precompdir
	${AUDFPRINT} new --dbase fpdbase0.pklz precompdir/Nine_Lives/0*
	${AUDFPRINT} new --dbase fpdbase1.pklz precompdir/Nine_Lives/1*
	rm -f fpdbase2.pklz
	${AUDFPRINT} newmerge --dbase fpdbase2.pklz fpdbase0.pklz fpdbase1.pklz
	${AUDFPRINT} match --dbase fpdbase2.pklz precompdir/query.afpt

precompdir: audfprint.py audfprint_analyze.py audfprint_match.py hash_table.py
	rm -rf precompdir
	mkdir precompdir
	${AUDFPRINT} precompute --precompdir precompdir Nine_Lives/*.mp3
	${AUDFPRINT} precompute --precompdir precompdir --shifts 4 query.mp3

test_onecore_precomppk: precomppkdir
	${AUDFPRINT} new --dbase fpdbase0.pklz precomppkdir/Nine_Lives/0*
	${AUDFPRINT} new --dbase fpdbase1.pklz precomppkdir/Nine_Lives/1*
	${AUDFPRINT} merge --dbase fpdbase1.pklz fpdbase0.pklz
	${AUDFPRINT} match --dbase fpdbase1.pklz precomppkdir/query.afpk
	rm -rf precomppkdir

precomppkdir: audfprint.py audfprint_analyze.py audfprint_match.py hash_table.py
	rm -rf precomppkdir
	mkdir precomppkdir
	${AUDFPRINT} precompute --precompute-peaks --precompdir precomppkdir Nine_Lives/*.mp3
	${AUDFPRINT} precompute --precompute-peaks --precompdir precomppkdir --shifts 4 query.mp3

test_mucore: fpdbase_mu.pklz
	${AUDFPRINT} match --dbase fpdbase_mu.pklz --ncores 4 query.mp3

fpdbase_mu.pklz: audfprint.py audfprint_analyze.py audfprint_match.py hash_table.py
	${AUDFPRINT} new --dbase fpdbase_mu.pklz --ncores 4 Nine_Lives/0*.mp3
	${AUDFPRINT} add --dbase fpdbase_mu.pklz --ncores 4 Nine_Lives/1*.mp3

test_mucore_precomp: precompdir_mu
	${AUDFPRINT} new --dbase fpdbase_mu0.pklz --ncores 4 precompdir_mu/Nine_Lives/0*
	${AUDFPRINT} new --dbase fpdbase_mu.pklz --ncores 4 precompdir_mu/Nine_Lives/1*
	${AUDFPRINT} merge --dbase fpdbase_mu.pklz fpdbase_mu0.pklz
	${AUDFPRINT} match --dbase fpdbase_mu.pklz --ncores 4 precompdir_mu/query.afpt precompdir_mu/query.afpt precompdir_mu/query.afpt precompdir_mu/query.afpt precompdir_mu/query.afpt precompdir_mu/query.afpt precompdir_mu/query.afpt

precompdir_mu: audfprint.py audfprint_analyze.py audfprint_match.py hash_table.py
	rm -rf precompdir_mu
	mkdir precompdir_mu
	${AUDFPRINT} precompute --ncores 4 --precompdir precompdir_mu Nine_Lives/*.mp3
	${AUDFPRINT} precompute --ncores 4 --precompdir precompdir_mu --shifts 4 query.mp3 query.mp3 query.mp3 query.mp3 query.mp3 query.mp3

test_hash_mask: 
	${AUDFPRINT} new --dbase fpdbase.pklz --hashbits 16 Nine_Lives/*.mp3
	${AUDFPRINT} match --dbase fpdbase.pklz query.mp3
