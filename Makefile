# audfprint/python/Makefile
#
# This is just to test the python version of audfprint by running each command,
# in both single proc and multiproc mode
#
# 2014-09-20 Dan Ellis dpwe@ee.columbia.edu

test: test_onecore test_onecore_precomp test_onecore_newmerge test_onecore_precomppk test_mucore test_mucore_precomp test_remove
	rm -rf precompdir precompdir_mu
	rm -f fpdbase*.pklz

test_onecore: fpdbase.pklz
	python audfprint.py match --dbase fpdbase.pklz query.mp3

test_remove: fpdbase.pklz
	python audfprint.py remove --dbase fpdbase.pklz Nine_Lives/01-Nine_Lives.mp3
	python audfprint.py list --dbase fpdbase.pklz
	python audfprint.py add --dbase fpdbase.pklz Nine_Lives/01-Nine_Lives.mp3
	python audfprint.py match --dbase fpdbase.pklz query.mp3

fpdbase.pklz: audfprint.py audfprint_analyze.py audfprint_match.py hash_table.py
	python audfprint.py new --dbase fpdbase.pklz Nine_Lives/0*.mp3
	python audfprint.py add --dbase fpdbase.pklz Nine_Lives/1*.mp3

test_onecore_precomp: precompdir
	python audfprint.py new --dbase fpdbase0.pklz precompdir/Nine_Lives/0*
	python audfprint.py new --dbase fpdbase1.pklz precompdir/Nine_Lives/1*
	python audfprint.py merge --dbase fpdbase1.pklz fpdbase0.pklz
	python audfprint.py match --dbase fpdbase1.pklz precompdir/query.afpt

test_onecore_newmerge: precompdir
	python audfprint.py new --dbase fpdbase0.pklz precompdir/Nine_Lives/0*
	python audfprint.py new --dbase fpdbase1.pklz precompdir/Nine_Lives/1*
	rm -f fpdbase2.pklz
	python audfprint.py newmerge --dbase fpdbase2.pklz fpdbase0.pklz fpdbase1.pklz
	python audfprint.py match --dbase fpdbase2.pklz precompdir/query.afpt

precompdir: audfprint.py audfprint_analyze.py audfprint_match.py hash_table.py
	rm -rf precompdir
	mkdir precompdir
	python audfprint.py precompute --precompdir precompdir Nine_Lives/*.mp3
	python audfprint.py precompute --precompdir precompdir --shifts 4 query.mp3

test_onecore_precomppk: precomppkdir
	python audfprint.py new --dbase fpdbase0.pklz precomppkdir/Nine_Lives/0*
	python audfprint.py new --dbase fpdbase1.pklz precomppkdir/Nine_Lives/1*
	python audfprint.py merge --dbase fpdbase1.pklz fpdbase0.pklz
	python audfprint.py match --dbase fpdbase1.pklz precomppkdir/query.afpk
	rm -rf precomppkdir

precomppkdir: audfprint.py audfprint_analyze.py audfprint_match.py hash_table.py
	rm -rf precomppkdir
	mkdir precomppkdir
	python audfprint.py precompute --precompute-peaks --precompdir precomppkdir Nine_Lives/*.mp3
	python audfprint.py precompute --precompute-peaks --precompdir precomppkdir --shifts 4 query.mp3

test_mucore: fpdbase_mu.pklz
	python audfprint.py match --dbase fpdbase_mu.pklz --ncores 4 query.mp3

fpdbase_mu.pklz: audfprint.py audfprint_analyze.py audfprint_match.py hash_table.py
	python audfprint.py new --dbase fpdbase_mu.pklz --ncores 4 Nine_Lives/0*.mp3
	python audfprint.py add --dbase fpdbase_mu.pklz --ncores 4 Nine_Lives/1*.mp3

test_mucore_precomp: precompdir_mu
	python audfprint.py new --dbase fpdbase_mu0.pklz --ncores 4 precompdir_mu/Nine_Lives/0*
	python audfprint.py new --dbase fpdbase_mu.pklz --ncores 4 precompdir_mu/Nine_Lives/1*
	python audfprint.py merge --dbase fpdbase_mu.pklz fpdbase_mu0.pklz
	python audfprint.py match --dbase fpdbase_mu.pklz --ncores 4 precompdir_mu/query.afpt precompdir_mu/query.afpt precompdir_mu/query.afpt precompdir_mu/query.afpt precompdir_mu/query.afpt precompdir_mu/query.afpt precompdir_mu/query.afpt

precompdir_mu: audfprint.py audfprint_analyze.py audfprint_match.py hash_table.py
	rm -rf precompdir_mu
	mkdir precompdir_mu
	python audfprint.py precompute --ncores 4 --precompdir precompdir_mu Nine_Lives/*.mp3
	python audfprint.py precompute --ncores 4 --precompdir precompdir_mu --shifts 4 query.mp3 query.mp3 query.mp3 query.mp3 query.mp3 query.mp3
