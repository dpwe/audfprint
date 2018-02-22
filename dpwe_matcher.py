#!/usr/bin/env python
# coding=utf-8

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
from __future__ import print_function

import os
import sys

import audfprint

try:
    # noinspection PyCompatibility
    from ConfigParser import ConfigParser  # Py2
except ImportError:
    from configparser import ConfigParser  # Py3
import errno

config_file = None
if len(sys.argv) < 2:
    print("Usage:", sys.argv[0], "[-C config.txt] fileList4query dir4db resultFile")
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
defaults = {'density': "70",
            'fanout': "8",
            'search_depth': "2000",
            'min_count': "5",
            'ncores': "8"}

# Parse input file
config = ConfigParser(defaults)
section = 'dpwe_matcher'

config.add_section(section)

if config_file:
    if len(config.read(config_file)) == 0:
        raise IOError(errno.ENOENT, "Cannot read config file", config_file)

density = config.getint(section, 'density')
fanout = config.getint(section, 'fanout')
search_depth = config.getint(section, 'search_depth')
min_count = config.getint(section, 'min_count')
ncores = config.getint(section, 'ncores')

print(sys.argv[0], "density:", density, "fanout:", fanout,
      "search_depth", search_depth, "min_count", min_count,
      "ncores:", ncores)

# Run the command
argv = ["audfprint", "match",
        "-d", os.path.join(dir4db, "data.fpdb"),
        "--density", str(density),
        "--fanout", str(fanout),
        "--search-depth", str(search_depth),
        "--min-count", str(min_count),
        "--ncores", str(ncores),
        "--verbose", 0,
        "--opfile", resultFile,
        "--list", fileList4query]

# Run audfprint
audfprint.main(argv)
