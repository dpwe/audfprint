audfprint
=========

Landmark-based audio fingerprinting.

```
Landmark-based audio fingerprinting.
Create a new fingerprint dbase with "new",
append new files to an existing database with "add",
or identify noisy query excerpts with "match".
"precompute" writes a *.fpt file under precompdir
with precomputed fingerprint for each input wav file.
"merge" combines previously-created databases into
an existing database; "newmerge" combines existing
databases to create a new one.

Usage: audfprint (new | add | match | precompute | merge | newmerge | list | remove) [options] [<file>]...

Options:
  -d <dbase>, --dbase <dbase>     Fingerprint database file
  -n <dens>, --density <dens>     Target hashes per second [default: 20.0]
  -h <bits>, --hashbits <bits>    How many bits in each hash [default: 20]
  -b <val>, --bucketsize <val>    Number of entries per bucket [default: 100]
  -t <val>, --maxtime <val>       Largest time value stored [default: 16384]
  -u <val>, --maxtimebits <val>   maxtime as a number of bits (16384 == 14 bits)
  -r <val>, --samplerate <val>    Resample input files to this [default: 11025]
  -p <dir>, --precompdir <dir>    Save precomputed files under this dir [default: .]
  -i <val>, --shifts <val>        Use this many subframe shifts building fp [default: 0]
  -w <val>, --match-win <val>     Maximum tolerable frame skew to count as a match [default: 2]
  -N <val>, --min-count <val>     Minimum number of matching landmarks to count as a match [default: 5]
  -x <val>, --max-matches <val>   Maximum number of matches to report for each query [default: 1]
  -X, --exact-count               Flag to use more precise (but slower) match counting
  -R, --find-time-range           Report the time support of each match
  -Q, --time-quantile <val>       Quantile at extremes of time support [default: 0.05]
  -S <val>, --freq-sd <val>       Frequency peak spreading SD in bins [default: 30.0]
  -F <val>, --fanout <val>        Max number of hash pairs per peak [default: 3]
  -P <val>, --pks-per-frame <val>  Maximum number of peaks per frame [default: 5]
  -D <val>, --search-depth <val>  How far down to search raw matching track list [default: 100]
  -H <val>, --ncores <val>        Number of processes to use [default: 1]
  -o <name>, --opfile <name>      Write output (matches) to this file, not stdout [default: ]
  -K, --precompute-peaks          Precompute just landmarks (else full hashes)
  -k, --skip-existing             On precompute, skip items if output file already exists
  -C, --continue-on-error         Keep processing despite errors reading input
  -l, --list                      Input files are lists, not audio
  -T, --sortbytime                Sort multiple hits per file by time (instead of score)
  -v <val>, --verbose <val>       Verbosity level [default: 1]
  -I, --illustrate                Make a plot showing the match
  -J, --illustrate-hpf            Plot the match, using onset enhancement
  -W <dir>, --wavdir <dir>        Find sound files under this dir [default: ]
  -V <ext>, --wavext <ext>        Extension to add to wav file names [default: ]
  --version                       Report version number
  --help                          Print this message
```

Uses librosa, get https://github.com/bmcfee/librosa or "pip install librosa"

Uses docopt, see http://docopt.org , get https://github.com/docopt/docopt or "pip install docopt"

Also uses numpy and scipy, which need to be installed, e.g. "pip install numpy", "pip install scipy".

Parallel processing relies on joblib module (and multiprocessing module), which are likely already installed.

This version uses `ffmpeg` to read input files.  You must have a working `ffmpeg` binary in your path (try `ffmpeg -V` at the command prompt).

Based on Matlab prototype, http://www.ee.columbia.edu/~dpwe/resources/matlab/audfprint/ .  This python code will actually read and use databases created by the Matlab code (version 0.90 upwards).

Usage
-----

Build a database of fingerprints from a set of reference audio files:
```
> python audfprint.py new --dbase fpdbase.pklz Nine_Lives/0*.mp3
Wed Sep 10 10:52:18 2014 ingesting #0:Nine_Lives/01-Nine_Lives.mp3 ...
Wed Sep 10 10:52:20 2014 ingesting #1:Nine_Lives/02-Falling_In_Love.mp3 ...
Wed Sep 10 10:52:22 2014 ingesting #2:Nine_Lives/03-Hole_In_My_Soul.mp3 ...
Wed Sep 10 10:52:25 2014 ingesting #3:Nine_Lives/04-Taste_Of_India.mp3 ...
Wed Sep 10 10:52:28 2014 ingesting #4:Nine_Lives/05-Full_Circle.mp3 ...
Wed Sep 10 10:52:31 2014 ingesting #5:Nine_Lives/06-Something_s_Gotta_Give.mp3 ...
Wed Sep 10 10:52:32 2014 ingesting #6:Nine_Lives/07-Ain_t_That_A_Bitch.mp3 ...
Wed Sep 10 10:52:35 2014 ingesting #7:Nine_Lives/08-The_Farm.mp3 ...
Wed Sep 10 10:52:37 2014 ingesting #8:Nine_Lives/09-Crash.mp3 ...
Added 63241 hashes (24.8 hashes/sec)
Processed 9 files (2547.3 s total dur) in 21.6 s sec = 0.008 x RT
Saved fprints for 9 files ( 63241 hashes) to fpdbase.pklz
```
Add more reference tracks to an existing database:
```
> python audfprint.py add --dbase fpdbase.pklz Nine_Lives/1*.mp3
Read fprints for 9 files ( 63241 hashes) from fpdbase.pklz
Wed Sep 10 10:53:14 2014 ingesting #0:Nine_Lives/10-Kiss_Your_Past_Good-bye.mp3 ...
Wed Sep 10 10:53:16 2014 ingesting #1:Nine_Lives/11-Pink.mp3 ...
Wed Sep 10 10:53:18 2014 ingesting #2:Nine_Lives/12-Attitude_Adjustment.mp3 ...
Wed Sep 10 10:53:20 2014 ingesting #3:Nine_Lives/13-Fallen_Angels.mp3 ...
Added 27067 hashes (22.0 hashes/sec)
Processed 4 files (1228.6 s total dur) in 13.0 s sec = 0.011 x RT
Saved fprints for 13 files ( 90308 hashes) to fpdbase.pklz
```
Match a fragment recorded of music playing in the background against the database:
```
> python audfprint.py match --dbase fpdbase.pklz query.mp3
Read fprints for 13 files ( 90308 hashes) from fpdbase.pklz
Analyzed query.mp3 of 5.573 s to 204 hashes
Matched query.mp3 5.573 sec 204 raw hashes as Nine_Lives/05-Full_Circle.mp3 at 50.085 s with 8 of 9 hashes
Processed 1 files (5.8 s total dur) in 2.6 s sec = 0.443 x RT
```
The query contained audio from `Nine_Lives/05-Full_Circle.mp3` starting at 50.085 sec into the track.  There were a total of 17 landmark hashes shared between the query and that track, and 14 of them had a consistent time offset.  Generally, anything more than 5 or 6 consistently-timed matching hashes indicate a true match, and random chance will result in fewer than 1% of the raw common hashes being temporally consistent.

Merge a previously-computed database into an existing one:
```
> python audfprint.py merge --dbase fpdbase.pklz fpdbase0.pklz
Wed Apr  8 18:31:29 2015 Reading hash table fpdbase.pklz
Read fprints for 4 files ( 126989 hashes) from fpdbase.pklz
Read fprints for 9 files ( 280424 hashes) from fpdbase0.pklz
Saved fprints for 13 files ( 407413 hashes) to fpdbase.pklz
```
Merge two existing databases to create a new, third one:
```
> python audfprint.py newmerge --dbase fpdbase_new.pklz fpdbase.pklz fpdbase0.pklz
Read fprints for 4 files ( 126989 hashes) from fpdbase.pklz
Read fprints for 9 files ( 280424 hashes) from fpdbase0.pklz
Saved fprints for 13 files ( 407363 hashes) to fpdbase_new.pklz
```

Locating Matches
----------------

To find out not just that two files match, and not just the relative timing
between them that makes them line up, but the exact *time ranges* that match
in both query and reference files, use `--find-time-range`:

```
python audfprint.py match --dbase fpdbase.pklz query.mp3 --find-time-range
Sun Aug  9 18:13:54 2015 Reading hash table fpdbase.pklz
Read fprints for 9 files ( 158827 hashes) from fpdbase.pklz
Sun Aug  9 18:13:57 2015 Analyzed #0 query.mp3 of 5.619 s to 928 hashes
Matched    3.6 s starting at    0.8 s in query.mp3 to time   50.9 s in Nine_Lives/05-Full_Circle.mp3 with    12 of    39 common hashes at rank  0
Processed 1 files (5.8 s total dur) in 2.6 s sec = 0.451 x RT
```

Notice how the message includes the precise duration and time points
in both query and reference item spanning the matches.  Because a
single spurious match elsewhere in the file can cause misleading
results, these times are calculated after discarding a small number of
the earliest and latest matches; this proportion is set by
`--time-quantile` which is 0.01 by default (1% of matches ignored at
beginning and end of match region when calculating match time range).

Scaling
-------
The fingerprint database records 2^20 (~1M) distinct fingerprints, with (by default) 100 entries for each fingerprint bucket.  When the bucket fills, track entries are dropped at random; since matching depends only on making a minimum number of matches, but no particular match, dropping some of the more popular ones does not prevent matching.  The Matlab version has been successfully used for databases of 100k+ tracks.  Reducing the hash density (`--density`) leads to smaller reference database size, and the capacity to record more reference items before buckets begin to fill; a density of 7.0 works well.

Times (in units of 256 samples, i.e., 23 ms at the default 11kHz sampling rate) are stored in the bottom 14 bits of each database entry, meaning that times larger than 2^14*0.023 = 380 sec, or about 6 mins, are aliased.  If you want to correctly identify time offsets in tracks longer than this, you need to use a larger `--maxtimebits`; e.g. `--maxtimebits 16` increases the time range to 65,536 frames, or about 25 minutes at 11 kHz.  The trade-off is that the remaining bits in each 32 bit entry (i.e., 18 bits for the default 14 bit times) are used to store the track ID.  Thus, by default, the database can only remember 2^18 = 262k tracks; using a larger `--maxtimebits` will reduce this; similarly, you can increase the number of distinct tracks by reducing `--maxtimebits`, which doesn't prevent matching tracks, but progressively reduces discrimination as the number of distinct time slots reduces (and can make the reported time offsets, and time ranges for `--find-time-ranges`, completely wrong for longer tracks).


