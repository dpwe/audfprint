README-dpwe-audfprint.txt

README for the dpwe-audfprint submission to MIREX 2014: Audio Fingerprinting

2014-09-14 Dan Ellis dpwe@ee.columbia.edu


INTRODUCTION

dpwe-audfprint is an audio fingerprinting system based on audfprint.py [1].  
It is configured to conform to the specifications for the 2014 MIREX 
evaluation of audio fingerprinting systems [2].  This document includes 
instructions on downloading and installing the system, as well as results 
on the publicly-released GTZAN development data.


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

dpwe_builder.py and dpwe_matcher.py accept an optional configuration file 
after an initial "-C" option, e.g.

  ./dpwe_builder.py -C config.txt ref.list db
  ./dpwe_builder.py -C config.txt queries.list db matches.out

See config.txt for an example.  You can control density (approximate number of
landmarks/sec), fanout (number of hashes per landmark), and ncores (number of 
cores used in multiprocessing) for both dpwe_builder and dpwe_matcher.

The audfprint package includes four example config files:

  config.txt       - specifies density=70 and fanout=8, the defaults
  config_low.txt   - has density=50 and fanout=6, for a smaller, faster, and 
                     less accurate example
  config2_high.txt - has density=100 and fanout=10, for a larger, slower, and 
                     more accurate example
  config2_tiny.txt - has density=20 and fanout=3, corresponding to the 
                     default settings for audfprint.py, for a very fast/small 
                     reference database.


EXAMPLE PERFORMANCE

On a 12-core Xeon E5-2420 (1.9 GHz) machine, using ncores=4, for the 
standard GTZAN set of 1000 30-second clips as reference, and the released 
set of 1062 queries as queries:

(a) Analysis takes 5:22.44 across 500 min of files, or 1.07% of real time

  porkpie:~/tmp/mirex/audfprint-master > time ./dpwe_builder.py ref.list db
  ./dpwe_builder.py density: 70 fanout: 8 ncores: 4
  ht 0 has 250 files 1114852 hashes
  ht 1 has 250 files 1114865 hashes
  ht 2 has 250 files 1120895 hashes
  ht 3 has 250 files 1112672 hashes
  Saved fprints for 1000 files ( 4463284 hashes) to db/data.fpdb
  996.194u 19.521s 5:22.44 315.0% 0+0k 0+33400io 0pf+0w

(b) The database occupies 16.9 MB for 500 min of files, or 34.2 kB per minute:

  porkpie:~/tmp/mirex/audfprint-master > ls -l db/
  total 16712
  -rw-r--r-- 1 dpwe dpwe 17087508 Sep 14 15:39 data.fpdb

(c) Matching takes 5:59.00 to process 1062 queries of 10 s each, or 3.4% of 
    real time:

  porkpie:~/tmp/mirex/audfprint-master > time ./dpwe_matcher.py queries.list db matches.out
  ./dpwe_matcher.py density: 70 fanout: 8 ncores: 4
  Read fprints for 1000 files ( 4463284 hashes) from db/data.fpdb
  1402.459u 19.917s 5:59.00 396.2%        0+0k 0+827720io 0pf+0w

(d) Accuracy for these settings on this database is 77.4%:

  porkpie:~/tmp/mirex/audfprint-master > ./comp_file_lines.py matches.out truth.out
  828 correct out of 1062 = 78.0%

The results for all configuration sets are summarized below:

CONFIG         DENSITY  FANOUT  BUILDER TIME  DBASE SIZE  MATCHER TIME  CORRECT
config_tiny.txt   20      3       0.80% RT      7.8 kB/m    1.0% RT      60.8%
config_low.txt    50      6       0.97% RT     22.5 kB/m    2.0% RT      77.7%
config.txt        70      8       1.07% RT     34.2 kB/m    3.4% RT      78.0%
config_high.txt  100     10       1.19% RT     52.4 kB/m    7.8% RT      78.2%

All numbers are running in multiprocessor mode (--ncores 4).


[1] http://github.com/dpwe/audfprint
    Note: the actual blob used for the results in this file is:
    https://github.com/dpwe/audfprint/tree/0301a20b02aa8ef381c29f211963aa8ca26cb32b

[2] http://www.music-ir.org/mirex/wiki/2014:Audio_Fingerprinting
