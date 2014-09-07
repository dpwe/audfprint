#!/usr/bin/env python

# dpwe_builder.py
#
# Fingerprint database builder for MIREX 2014 audio fingerprint competition
#
# 2014-09-06 Dan Ellis dpwe@ee.columbia.edu


# From http://www.music-ir.org/mirex/wiki/2014:Audio_Fingerprinting#Time_and_hardware_limits

# 1. Database Builder
# Command format:
# builder %fileList4db% %dir4db%
# where %fileList4db% is a file containing the input list of database audio files, with name convention as uniqueKey.mp3. For example:
# ./AFP/database/000001.mp3
# ./AFP/database/000002.mp3
# ./AFP/database/000003.mp3
# ./AFP/database/000004.mp3
# ...
# The output file(s), which containing all the information of the database to be used for audio fingerprinting, should be placed placed into the directory %dir4db%. The total size of the database file(s) is restricted to a certain amount, as explained next.

import sys, os
import audfprint

fileList4db = sys.argv[1]
dir4db = sys.argv[2]

# Params
density = 100
fanout  = 3
ncores  = 4

argv = ["audfprint", "new", 
        "-d", os.path.join(dir4db, "data.fpdb"), 
        "--density", str(density), 
        "--fanout", str(fanout),
        "--multiproc", 
        "--ncores", str(ncores), 
        "--list", fileList4db]

# Run audfprint
audfprint.main(argv)
