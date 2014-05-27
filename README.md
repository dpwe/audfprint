audfprint
=========

Audio landmark-based fingerprinting.  

Create a new fingerprint dbase with new, 
append new files to an existing database with add, 
or identify noisy query excerpts with match.

Usage: audfprint (new | add | match) (-d <dbase> | --dbase <dbase>) [options] <file>...

Options:

  -n <dens>, --density <dens>     Target hashes per second [default: 7.0]

  -h <bits>, --hashbits <bits>    How many bits in each hash [default: 20]

  -b <val>, --bucketsize <val>    Number of entries per bucket [default: 100]

  -t <val>, --maxtime <val>       Largest time value stored [default: 16384]

  -l, --list                      Input files are lists, not audio

  --version                       Report version number

  --help                          Print this message
