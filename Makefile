# audfprint/python/Makefile
#
# This is just to test the python version of audfprint by running each command,
# in both single proc and multiproc mode
#
# 2014-09-20 Dan Ellis dpwe@ee.columbia.edu

test: test_onecore test_onecore_precomp test_mucore test_mucore_precomp
	rm -rf precompdir precompdir_mu

test_onecore: fpdbase.pklz audfprint_match.py
	python audfprint.py match --dbase fpdbase.pklz query.mp3

fpdbase.pklz: audfprint.py
	python audfprint.py new --dbase fpdbase.pklz Nine_Lives/0*.mp3
	python audfprint.py add --dbase fpdbase.pklz Nine_Lives/1*.mp3

test_onecore_precomp: precompdir
	python audfprint.py new --dbase fpdbase0.pklz precompdir/Nine_Lives/0*
	python audfprint.py new --dbase fpdbase.pklz precompdir/Nine_Lives/1*
	python audfprint.py merge --dbase fpdbase.pklz fpdbase0.pklz
	python audfprint.py match --dbase fpdbase.pklz precompdir/query.afpt

precompdir: audfprint.py
	rm -rf precompdir
	mkdir precompdir
	python audfprint.py precompute --precompdir precompdir Nine_Lives/*.mp3
	python audfprint.py precompute --precompdir precompdir --shifts 4 query.mp3

test_mucore: fpdbase_mu.pklz audfprint_match.py
	python audfprint.py match --dbase fpdbase_mu.pklz --ncores 4 query.mp3

fpdbase_mu.pklz: audfprint.py
	python audfprint.py new --dbase fpdbase_mu.pklz --ncores 4 Nine_Lives/0*.mp3
	python audfprint.py add --dbase fpdbase_mu.pklz --ncores 4 Nine_Lives/1*.mp3

test_mucore_precomp: precompdir_mu
	python audfprint.py new --dbase fpdbase_mu0.pklz --ncores 4 precompdir_mu/Nine_Lives/0*
	python audfprint.py new --dbase fpdbase_mu.pklz --ncores 4 precompdir_mu/Nine_Lives/1*
	python audfprint.py merge --dbase fpdbase_mu.pklz fpdbase_mu0.pklz
	python audfprint.py match --dbase fpdbase_mu.pklz --ncores 4 precompdir_mu/query.afpt precompdir_mu/query.afpt precompdir_mu/query.afpt precompdir_mu/query.afpt precompdir_mu/query.afpt precompdir_mu/query.afpt precompdir_mu/query.afpt

precompdir_mu: audfprint.py
	rm -rf precompdir_mu
	mkdir precompdir_mu
	python audfprint.py precompute --ncores 4 --precompdir precompdir_mu Nine_Lives/*.mp3
	python audfprint.py precompute --ncores 4 --precompdir precompdir_mu --shifts 4 query.mp3 query.mp3 query.mp3 query.mp3 query.mp3 query.mp3
