#!/usr/bin/env python
# coding=utf-8

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
    print("Usage:", sys.argv[0], "[-C config.txt] fileList4db dir4db")
    sys.exit()

else:
    if sys.argv[1] == "-C":
        config_file = sys.argv[2]
        fileList4db = sys.argv[3]
        dir4db = sys.argv[4]
    else:
        fileList4db = sys.argv[1]
        dir4db = sys.argv[2]

# Default params
defaults = {'density': "70",
            'fanout': "8",
            'bucketsize': "500",
            'ncores': "8"}

# Parse input file
config = ConfigParser(defaults)
section = 'dpwe_builder'
config.add_section(section)

if config_file:
    if len(config.read(config_file)) == 0:
        raise IOError(errno.ENOENT, "Cannot read config file", config_file)

density = config.getint(section, 'density')
fanout = config.getint(section, 'fanout')
bucketsize = config.getint(section, 'bucketsize')
ncores = config.getint(section, 'ncores')

print(sys.argv[0], "density:", density, "fanout:", fanout,
      "bucketsize:", bucketsize, "ncores:", ncores)

# Ensure the database directory exists
audfprint.ensure_dir(dir4db)

# Run the command
argv = ["audfprint", "new",
        "-d", os.path.join(dir4db, "data.fpdb"),
        "--density", str(density),
        "--fanout", str(fanout),
        "--bucketsize", str(bucketsize),
        "--ncores", str(ncores),
        "--list", fileList4db]

# Run audfprint
audfprint.main(argv)
