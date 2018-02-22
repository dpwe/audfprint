#!/usr/bin/python
# coding=utf-8
"""
audfprint.py

Implementation of acoustic-landmark-based robust fingerprinting.
Port of the Matlab implementation.

2014-05-25 Dan Ellis dpwe@ee.columbia.edu
"""
from __future__ import division, print_function

import multiprocessing  # for new/add
import os  # For command line interface
import sys  # For __main__
import time  # For reporting progress time

import docopt  # For command line interface
import joblib  # for match

import audfprint_analyze  # The actual analyzer class/code
import audfprint_match  # Access to match functions, used in command line interface
import hash_table  # My hash_table implementation


def filename_list_iterator(filelist, wavdir, wavext, listflag):
    """ Iterator to yeild all the filenames, possibly interpreting them
        as list files, prepending wavdir """
    if not listflag:
        for filename in filelist:
            yield os.path.join(wavdir, filename + wavext)
    else:
        for listfilename in filelist:
            with open(listfilename, 'r') as f:
                for filename in f:
                    yield os.path.join(wavdir, filename.rstrip('\n') + wavext)


# for saving precomputed fprints
def ensure_dir(dirname):
    """ ensure that the named directory exists """
    if len(dirname):
        if not os.path.exists(dirname):
            # There's a race condition for multiprocessor; don't worry if the
            # directory gets created before we get to it.
            try:
                os.makedirs(dirname)
            except:
                pass


# Command line interface

# basic operations, each in a separate function

def file_precompute_peaks_or_hashes(analyzer, filename, precompdir,
                                    precompext=None, hashes_not_peaks=True,
                                    skip_existing=False,
                                    strip_prefix=None):
    """ Perform precompute action for one file, return list
        of message strings """
    # If strip_prefix is specified and matches the start of filename,
    # remove it from filename.
    if strip_prefix and filename[:len(strip_prefix)] == strip_prefix:
        tail_filename = filename[len(strip_prefix):]
    else:
        tail_filename = filename
    # Form the output filename to check if it exists.
    # strip relative directory components from file name
    # Also remove leading absolute path (comp == '')
    relname = '/'.join([comp for comp in tail_filename.split('/')
                        if comp != '.' and comp != '..' and comp != ''])
    root = os.path.splitext(relname)[0]
    if precompext is None:
        if hashes_not_peaks:
            precompext = audfprint_analyze.PRECOMPEXT
        else:
            precompext = audfprint_analyze.PRECOMPPKEXT
    opfname = os.path.join(precompdir, root + precompext)
    if skip_existing and os.path.isfile(opfname):
        return ["file " + opfname + " exists (and --skip-existing); skipping"]
    else:
        # Do the analysis
        if hashes_not_peaks:
            type = "hashes"
            saver = audfprint_analyze.hashes_save
            output = analyzer.wavfile2hashes(filename)
        else:
            type = "peaks"
            saver = audfprint_analyze.peaks_save
            output = analyzer.wavfile2peaks(filename)
        # save the hashes or peaks file
        if len(output) == 0:
            message = "Zero length analysis for " + filename + " -- not saving."
        else:
            # Make sure the output directory exists
            ensure_dir(os.path.split(opfname)[0])
            # Write the file
            saver(opfname, output)
            message = ("wrote " + opfname + " ( %d %s, %.3f sec)"
                       % (len(output), type, analyzer.soundfiledur))
        return [message]


def file_precompute(analyzer, filename, precompdir, type='peaks', skip_existing=False, strip_prefix=None):
    """ Perform precompute action for one file, return list
        of message strings """
    print(time.ctime(), "precomputing", type, "for", filename, "...")
    hashes_not_peaks = (type == 'hashes')
    return file_precompute_peaks_or_hashes(analyzer, filename, precompdir,
                                           hashes_not_peaks=hashes_not_peaks,
                                           skip_existing=skip_existing,
                                           strip_prefix=strip_prefix)


def make_ht_from_list(analyzer, filelist, hashbits, depth, maxtime, pipe=None):
    """ Populate a hash table from a list, used as target for
        multiprocess division.  pipe is a pipe over which to push back
        the result, else return it """
    # Create new ht instance
    ht = hash_table.HashTable(hashbits=hashbits, depth=depth, maxtime=maxtime)
    # Add in the files
    for filename in filelist:
        hashes = analyzer.wavfile2hashes(filename)
        ht.store(filename, hashes)
    # Pass back to caller
    if pipe:
        pipe.send(ht)
    else:
        return ht


def do_cmd(cmd, analyzer, hash_tab, filename_iter, matcher, outdir, type, report, skip_existing=False, strip_prefix=None):
    """ Breaks out the core part of running the command.
        This is just the single-core versions.
    """
    if cmd == 'merge' or cmd == 'newmerge':
        # files are other hash tables, merge them in
        for filename in filename_iter:
            hash_tab2 = hash_table.HashTable(filename)
            if "samplerate" in hash_tab.params:
                assert hash_tab.params["samplerate"] == hash_tab2.params["samplerate"]
            else:
                # "newmerge" fails to setup the samplerate param
                hash_tab.params["samplerate"] = hash_tab2.params["samplerate"]
            hash_tab.merge(hash_tab2)

    elif cmd == 'precompute':
        # just precompute fingerprints, single core
        for filename in filename_iter:
            report(file_precompute(analyzer, filename, outdir, type, skip_existing=skip_existing, strip_prefix=strip_prefix))

    elif cmd == 'match':
        # Running query, single-core mode
        for num, filename in enumerate(filename_iter):
            msgs = matcher.file_match_to_msgs(analyzer, hash_tab, filename, num)
            report(msgs)

    elif cmd == 'new' or cmd == 'add':
        # Adding files
        tothashes = 0
        ix = 0
        for filename in filename_iter:
            report([time.ctime() + " ingesting #" + str(ix) + ": "
                    + filename + " ..."])
            dur, nhash = analyzer.ingest(hash_tab, filename)
            tothashes += nhash
            ix += 1

        report(["Added " + str(tothashes) + " hashes "
                + "(%.1f" % (tothashes / float(analyzer.soundfiletotaldur))
                + " hashes/sec)"])
    elif cmd == 'remove':
        # Removing files from hash table.
        for filename in filename_iter:
            hash_tab.remove(filename)

    elif cmd == 'list':
        hash_tab.list(lambda x: report([x]))

    else:
        raise ValueError("unrecognized command: " + cmd)


def multiproc_add(analyzer, hash_tab, filename_iter, report, ncores):
    """Run multiple threads adding new files to hash table"""
    # run ncores in parallel to add new files to existing HASH_TABLE
    # lists store per-process parameters
    # Pipes to transfer results
    rx = [[] for _ in range(ncores)]
    tx = [[] for _ in range(ncores)]
    # Process objects
    pr = [[] for _ in range(ncores)]
    # Lists of the distinct files
    filelists = [[] for _ in range(ncores)]
    # unpack all the files into ncores lists
    ix = 0
    for filename in filename_iter:
        filelists[ix % ncores].append(filename)
        ix += 1
    # Launch each of the individual processes
    for ix in range(ncores):
        rx[ix], tx[ix] = multiprocessing.Pipe(False)
        pr[ix] = multiprocessing.Process(target=make_ht_from_list,
                                         args=(analyzer, filelists[ix],
                                               hash_tab.hashbits,
                                               hash_tab.depth,
                                               (1 << hash_tab.maxtimebits),
                                               tx[ix]))
        pr[ix].start()
    # gather results when they all finish
    for core in range(ncores):
        # thread passes back serialized hash table structure
        hash_tabx = rx[core].recv()
        report(["hash_table " + str(core) + " has "
                + str(len(hash_tabx.names))
                + " files " + str(sum(hash_tabx.counts)) + " hashes"])
        # merge in all the new items, hash entries
        hash_tab.merge(hash_tabx)
        # finish that thread...
        pr[core].join()


def matcher_file_match_to_msgs(matcher, analyzer, hash_tab, filename):
    """Cover for matcher.file_match_to_msgs so it can be passed to joblib"""
    return matcher.file_match_to_msgs(analyzer, hash_tab, filename)


def do_cmd_multiproc(cmd, analyzer, hash_tab, filename_iter, matcher,
                     outdir, type, report, skip_existing=False,
                     strip_prefix=None, ncores=1):
    """ Run the actual command, using multiple processors """
    if cmd == 'precompute':
        # precompute fingerprints with joblib
        msgslist = joblib.Parallel(n_jobs=ncores)(
                joblib.delayed(file_precompute)(analyzer, file, outdir, type, skip_existing, strip_prefix=strip_prefix)
                for file in filename_iter
        )
        # Collapse into a single list of messages
        for msgs in msgslist:
            report(msgs)

    elif cmd == 'match':
        # Running queries in parallel
        msgslist = joblib.Parallel(n_jobs=ncores)(
                # Would use matcher.file_match_to_msgs(), but you
                # can't use joblib on an instance method
                joblib.delayed(matcher_file_match_to_msgs)(matcher, analyzer,
                                                           hash_tab, filename)
                for filename in filename_iter
        )
        for msgs in msgslist:
            report(msgs)

    elif cmd == 'new' or cmd == 'add':
        # We add by forking multiple parallel threads each running
        # analyzers over different subsets of the file list
        multiproc_add(analyzer, hash_tab, filename_iter, report, ncores)

    else:
        # This is not a multiproc command
        raise ValueError("unrecognized multiproc command: " + cmd)


# Command to separate out setting of analyzer parameters
def setup_analyzer(args):
    """Create a new analyzer object, taking values from docopts args"""
    # Create analyzer object; parameters will get set below
    analyzer = audfprint_analyze.Analyzer()
    # Read parameters from command line/docopts
    analyzer.density = float(args['--density'])
    analyzer.maxpksperframe = int(args['--pks-per-frame'])
    analyzer.maxpairsperpeak = int(args['--fanout'])
    analyzer.f_sd = float(args['--freq-sd'])
    analyzer.shifts = int(args['--shifts'])
    # fixed - 512 pt FFT with 256 pt hop at 11025 Hz
    analyzer.target_sr = int(args['--samplerate'])
    analyzer.n_fft = 512
    analyzer.n_hop = analyzer.n_fft // 2
    # set default value for shifts depending on mode
    if analyzer.shifts == 0:
        # Default shift is 4 for match, otherwise 1
        analyzer.shifts = 4 if args['match'] else 1
    analyzer.fail_on_error = not args['--continue-on-error']
    return analyzer


# Command to separate out setting of matcher parameters
def setup_matcher(args):
    """Create a new matcher objects, set parameters from docopt structure"""
    matcher = audfprint_match.Matcher()
    matcher.window = int(args['--match-win'])
    matcher.threshcount = int(args['--min-count'])
    matcher.max_returns = int(args['--max-matches'])
    matcher.search_depth = int(args['--search-depth'])
    matcher.sort_by_time = args['--sortbytime']
    matcher.exact_count = args['--exact-count'] | args['--illustrate'] | args['--illustrate-hpf']
    matcher.illustrate = args['--illustrate'] | args['--illustrate-hpf']
    matcher.illustrate_hpf = args['--illustrate-hpf']
    matcher.verbose = args['--verbose']
    matcher.find_time_range = args['--find-time-range']
    matcher.time_quantile = float(args['--time-quantile'])
    return matcher


# Command to construct the reporter object
def setup_reporter(args):
    """ Creates a logging function, either to stderr or file"""
    opfile = args['--opfile']
    if opfile and len(opfile):
        f = open(opfile, "w")

        def report(msglist):
            """Log messages to a particular output file"""
            for msg in msglist:
                f.write(msg + "\n")
    else:
        def report(msglist):
            """Log messages by printing to stdout"""
            for msg in msglist:
                print(msg)
    return report


# CLI specified via usage message thanks to docopt
USAGE = """
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
"""

__version__ = 20150406


def main(argv):
    """ Main routine for the command-line interface to audfprint """
    # Other globals set from command line
    args = docopt.docopt(USAGE, version=__version__, argv=argv[1:])

    # Figure which command was chosen
    poss_cmds = ['new', 'add', 'precompute', 'merge', 'newmerge', 'match',
                 'list', 'remove']
    cmdlist = [cmdname
               for cmdname in poss_cmds
               if args[cmdname]]
    if len(cmdlist) != 1:
        raise ValueError("must specify exactly one command")
    # The actual command as a str
    cmd = cmdlist[0]

    # Setup output function
    report = setup_reporter(args)

    # Keep track of wall time
    initticks = time.clock()

    # Command line sanity.
    if args["--maxtimebits"]:
        args["--maxtimebits"] = int(args["--maxtimebits"])
    else:
        args["--maxtimebits"] = hash_table._bitsfor(int(args["--maxtime"]))

    # Setup the analyzer if we're using one (i.e., unless "merge")
    analyzer = setup_analyzer(args) if not (
            cmd is "merge" or cmd is "newmerge"
            or cmd is "list" or cmd is "remove") else None

    precomp_type = 'hashes'

    # Set up the hash table, if we're using one (i.e., unless "precompute")
    if cmd is not "precompute":
        # For everything other than precompute, we need a database name
        # Check we have one
        dbasename = args['--dbase']
        if not dbasename:
            raise ValueError("dbase name must be provided if not precompute")
        if cmd == "new" or cmd == "newmerge":
            # Check that the output directory can be created before we start
            ensure_dir(os.path.split(dbasename)[0])
            # Create a new hash table
            hash_tab = hash_table.HashTable(
                    hashbits=int(args['--hashbits']),
                    depth=int(args['--bucketsize']),
                    maxtime=(1 << int(args['--maxtimebits'])))
            # Set its samplerate param
            if analyzer:
                hash_tab.params['samplerate'] = analyzer.target_sr

        else:
            # Load existing hash table file (add, match, merge)
            if args['--verbose']:
                report([time.ctime() + " Reading hash table " + dbasename])
            hash_tab = hash_table.HashTable(dbasename)
            if analyzer and 'samplerate' in hash_tab.params \
                    and hash_tab.params['samplerate'] != analyzer.target_sr:
                # analyzer.target_sr = hash_tab.params['samplerate']
                print("db samplerate overridden to ", analyzer.target_sr)
    else:
        # The command IS precompute
        # dummy empty hash table
        hash_tab = None
        if args['--precompute-peaks']:
            precomp_type = 'peaks'

    # Create a matcher
    matcher = setup_matcher(args) if cmd == 'match' else None

    filename_iter = filename_list_iterator(
            args['<file>'], args['--wavdir'], args['--wavext'], args['--list'])

    #######################
    # Run the main commmand
    #######################

    # How many processors to use (multiprocessing)
    ncores = int(args['--ncores'])
    if ncores > 1 and not (cmd == "merge" or cmd == "newmerge" or
                           cmd == "list" or cmd == "remove"):
        # merge/newmerge/list/remove are always single-thread processes
        do_cmd_multiproc(cmd, analyzer, hash_tab, filename_iter,
                         matcher, args['--precompdir'],
                         precomp_type, report,
                         skip_existing=args['--skip-existing'],
                         strip_prefix=args['--wavdir'],
                         ncores=ncores)
    else:
        do_cmd(cmd, analyzer, hash_tab, filename_iter,
               matcher, args['--precompdir'], precomp_type, report,
               skip_existing=args['--skip-existing'],
               strip_prefix=args['--wavdir'])

    elapsedtime = time.clock() - initticks
    if analyzer and analyzer.soundfiletotaldur > 0.:
        print("Processed "
              + "%d files (%.1f s total dur) in %.1f s sec = %.3f x RT" \
              % (analyzer.soundfilecount, analyzer.soundfiletotaldur,
                 elapsedtime, (elapsedtime / analyzer.soundfiletotaldur)))

    # Save the hash table file if it has been modified
    if hash_tab and hash_tab.dirty:
        # We already created the directory, if "new".
        hash_tab.save(dbasename)


# Run the main function if called from the command line
if __name__ == "__main__":
    main(sys.argv)
