#!/usr/bin/env python

# dpwe_matcher.py
#
# Fingerprint database matcher for MIREX 2014 audio fingerprint competition
#
# 2014-09-06 Dan Ellis dpwe@ee.columbia.edu


# From http://www.music-ir.org/mirex/wiki/2014:Audio_Fingerprinting#Time_and_hardware_limits

# 2. Matcher
# Command format:
# matcher %fileList4query% %dir4db% %resultFile%
# where %fileList4query% is a file containing the list of query clips. For example:
# ./AFP/query/q000001.wav
# ./AFP/query/q000002.wav
# ./AFP/query/q000003.wav
# ./AFP/query/q000004.wav
# ...
# The result file gives retrieved result for each query, with the format:
# %queryFilePath%	%dbFilePath%
# where these two fields are separated by a tab. Here is a more specific example:
# ./AFP/query/q000001.wav	./AFP/database/0000004.mp3
# ./AFP/query/q000002.wav	./AFP/database/0000054.mp3
# ./AFP/query/q000003.wav	./AFP/database/0001002.mp3
# ..

import sys, os
import audfprint

config_file = None
if len(sys.argv) < 2:
    print "Usage:", sys.argv[0], "[-C config.txt] fileList4query dir4db resultFile"
    sys.exit()

else:
    if sys.argv[1] == "-C":
        config_file = sys.argv[2]
        fileList4query = sys.argv[3]
        dir4db = sys.argv[4]
        resultFile = sys.argv[5]
    else:
        fileList4query = sys.argv[1]
        dir4db = sys.argv[2]
        resultFile = sys.argv[3]


# Default params
defaults = {'density': "100", 
            'fanout': "5", 
            'ncores': "4"}

# Parse input file
import ConfigParser
config = ConfigParser.ConfigParser(defaults)
section = 'audfprint'
config.add_section(section)

if config_file:
    config.read(config_file)

density = config.getint(section, 'density')
fanout = config.getint(section, 'fanout')
ncores = config.getint(section, 'ncores')

print density, fanout, ncores

# Run the command
argv = ["audfprint", "match", 
        "-d", os.path.join(dir4db, "data.fpdb"), 
        "--density", str(density), 
        "--fanout", str(fanout),
        "--multiproc", 
        "--ncores", str(ncores),
        "--verbose", 0, 
        "--opfile", resultFile, 
        "--list", fileList4query]

# Run audfprint
audfprint.main(argv)
