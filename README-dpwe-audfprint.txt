README-dpwe-audfprint.txt

README for the dpwe-audfprint submission to MIREX 2014: Audio Fingerprinting

2014-09-14 Dan Ellis dpwe@ee.columbia.edu

INTRODUCTION

dpwe-audfprint is an audio fingerprinting system based on audfprint.py [1].  
It is configured to conform to the specifications for the 2014 MIREX 
evaluation of audio fingerprinting systems [2].  This document includes 
instructions on downloading and installing the system, as well as results 
on the publicly-released development data.

DOWNLOAD AND INSTALLATION

dpwe-audfprint runs under Python and has been developed on a Ubuntu 12.04 
Linux system with Python 2.7.3.  It depends on two libraries, librosa (a 
set of audio processing utilities developed at LabROSA), and docopt 
(a command line processing utility).  Note that librosa also depends on 
matplotlib, which sometimes causes problems during installation; 
hopefully, it is already present on your system.  To install:

  # Install librosa
  pip install librosa
  # Install docopt
  pip install docopt
  # Download the audfprint code
  wget https://github.com/dpwe/audfprint/archive/master.zip
  unzip master.zip
  cd audfprint-master


BUILDING THE DATABASE

The "builder" program, following the conventions from the MIREX page, is 
called dpwe_builder.py.  It will build a database from a list of soundfiles 
<ref.list> and store the database in a directory <db> with the following 
command:

  mkdir db
  ./dpwe_builder.py ref.list db


QUERYING THE DATABASE

There is a corresponding "matcher" program to identify query soundfiles, 
named in <queries.list>, against the database written by dpwe_builder.py, 
writing the matches to <matches.out>:

  ./dpwe_matcher.py queries.list db matches.out

<matches.out> consists of one line per entry in <queries.list>, in the 
format:

  <path_from_queries>\t<matching_file_from_ref>

where "\t" indicates a tab character.  If no matching item is found, the 
second part of the line is blank.


ALTERNATE CONFIGURATIONS

dpwe_builder and dpwe_matcher accept an optional configuration file after 
an initial "-C" option, e.g.

  ./dpwe_builder.py -C config.txt ref.list db
  ./dpwe_builder.py -C config.txt queries.list db matches.out

See config.txt for an example.  You can control density (approximate number of
landmarks/sec), fanout (number of hashes per landmark), and ncores (number of 
cores used in multiprocessing) for both dpwe_builder and dpwe_matcher.

The audfprint package includes three example config files:

  config.txt   - specifies density=70 and fanout=8, the defaults
  config1.txt  - has density=50 and fanout=6, for a smaller, faster, and 
                 less accurate example
  config2.txt  - has density=100 and fanout=10, for a larger, slower, and 
  	         more accurate example

EXAMPLE PERFORMANCE

On a 12-core Xeon E5-2420 (1.9 GHz) machine, using ncores=4, using the 
standard GTZAN set of 1000 30-second clips as reference, and the released 
set of 1062 queries as queries:

(a) Analysis takes 5:09.46 across 500 min of files, or 1.03% of real time

  porkpie:~/tmp/mirex/audfprint-master > time ./dpwe_builder.py ref.list db
  ./dpwe_builder.py density: 100 fanout: 5 ncores: 4
  ht 0 has 250 files 1091630 hashes
  ht 1 has 250 files 1098682 hashes
  ht 2 has 250 files 1101037 hashes
  ht 3 has 250 files 1097195 hashes
  Saved fprints for 1000 files ( 4388544 hashes) to db/data.fpdb
  985.797u 19.617s 5:09.46 324.8% 0+0k 0+33048io 0pf+0w

(b) The database occupies 16.9 MB for 500 min of files, or 33.8 kB per minute:

  porkpie:~/tmp/mirex/audfprint-master > ls -l db/
  total 16536
  -rw-r--r-- 1 dpwe dpwe 16905657 Sep 14 10:36 data.fpdb

(c) Matching takes 7:22.56 to process 1062 of 10 s each, or 4.2% of real time:

  porkpie:~/tmp/mirex/audfprint-master > time ./dpwe_matcher.py queries.list db matches.out
  ./dpwe_matcher.py density: 100 fanout: 5 ncores: 4
  Read fprints for 1000 files ( 4388544 hashes) from db/data.fpdb
  1716.323u 39.962s 7:22.56 396.8%        0+0k 0+827720io 0pf+0w

(d) Accuracy for these settings on this database is 77.4%:

  porkpie:~/tmp/mirex/audfprint-master > python comp_file_lines.py matches.out truth.out
  822 correct out of 1062 = 77.4%

The results for all configuration sets are summarized below

Config         density  fanout  builder time  dbase size  matcher time  correct
config.txt        70      8       1.07% RT     34.2 kB/m    3.4% RT      78.0%
config_low.txt    50      6
config_high.txt  100     10


[1] https://github.com/dpwe/audfprint
[2] http://www.music-ir.org/mirex/wiki/2014:Audio_Fingerprinting
