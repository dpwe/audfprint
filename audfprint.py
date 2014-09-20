"""
audfprint.py

Implementation of acoustic-landmark-based robust fingerprinting.
Port of the Matlab implementation.

2014-05-25 Dan Ellis dpwe@ee.columbia.edu
"""
from __future__ import print_function

import numpy as np
import librosa
import scipy.signal
# For reading/writing hashes to file
import struct
# For glob2hashtable
import glob
# For reporting progress time
import time
# For command line interface
import docopt
import os
# For __main__
import sys
# For multiprocessing options
import multiprocessing  # for new/add
import joblib           # for match

# My hash_table implementation
import hash_table
# Access to match functions, used in command line interface
import audfprint_match

################ Globals ################
# Special extension indicating precomputed fingerprint
PRECOMPEXT = '.afpt'


def locmax(vec, indices=False):
    """ Return a boolean vector of which points in vec are local maxima.
        End points are peaks if larger than single neighbors.
        if indices=True, return the indices of the True values instead
        of the boolean vector.
    """
    # vec[-1]-1 means last value can be a peak
    #nbr = np.greater_equal(np.r_[vec, vec[-1]-1], np.r_[vec[0], vec])
    # the np.r_ was killing us, so try an optimization...
    nbr = np.zeros(len(vec)+1, dtype=bool)
    nbr[0] = True
    nbr[1:-1] = np.greater_equal(vec[1:], vec[:-1])
    maxmask = (nbr[:-1] & ~nbr[1:])
    if indices:
        return np.nonzero(maxmask)[0]
    else:
        return maxmask

# Constants for Analyzer
# DENSITY controls the density of landmarks found (approx DENSITY per sec)
DENSITY = 20.0
# OVERSAMP > 1 tries to generate extra landmarks by decaying faster
OVERSAMP = 1
## 512 pt FFT @ 11025 Hz, 50% hop
#t_win = 0.0464
#t_hop = 0.0232
# Just specify n_fft
N_FFT = 512
N_HOP = 256
# spectrogram enhancement
HPF_POLE = 0.98

# Globals defining packing of landmarks into hashes
F1_BITS = 8
DF_BITS = 6
DT_BITS = 6
# derived constants
B1_MASK = (1 << F1_BITS) - 1
B1_SHIFT = DF_BITS + DT_BITS
DF_MASK = (1 << DF_BITS) - 1
DF_SHIFT = DT_BITS
DT_MASK = (1 << DT_BITS) - 1

def landmarks2hashes(landmarks):
    """Convert a list of (time, bin1, bin2, dtime) landmarks
    into a list of (time, hash) pairs where the hash combines
    the three remaining values.
    """
    # build up and return the list of hashed values
    return [(time_,
             (((bin1 & B1_MASK) << B1_SHIFT)
              | (((bin2 - bin1) & DF_MASK) << DF_SHIFT)
              | (dtime & DT_MASK)))
            for time_, bin1, bin2, dtime in landmarks]

def hashes2landmarks(hashes):
    """Convert the mashed-up landmarks in hashes back into a list
    of (time, bin1, bin2, dtime) tuples.
    """
    landmarks = []
    for time_, hash_ in hashes:
        dtime = hash_ & DT_MASK
        bin1 = (hash_ >> B1_SHIFT) & B1_MASK
        dbin = (hash_ >> DF_SHIFT) & DF_MASK
        # Sign extend frequency difference
        if dbin > (1 << (DF_BITS-1)):
            dbin -= (1 << DF_BITS)
        landmarks.append((time_, bin1, bin1+dbin, dtime))
    return landmarks


class Analyzer(object):
    """ A class to wrap up all the parameters associated with
        the analysis of soundfiles into fingerprints """
    # Parameters

    # optimization: cache pre-calculated Gaussian profile
    __sp_width = None
    __sp_len = None
    __sp_vals = []

    def __init__(self, density=DENSITY):
        self.density = density
        self.target_sr = 11025
        self.n_fft = N_FFT
        self.n_hop = N_HOP
        self.shifts = 1
        # how wide to spreak peaks
        self.f_sd = 30.0
        # Maximum number of local maxima to keep per frame
        self.maxpksperframe = 5
        # Limit the num of pairs we'll make from each peak (Fanout)
        self.maxpairsperpeak = 3
        # Values controlling peaks2landmarks
        # +/- 31 bins in freq (LIMITED TO -32..31 IN LANDMARK2HASH)
        self.targetdf = 31
        # min time separation (traditionally 1, upped 2014-08-04)
        self.mindt = 2
        # max lookahead in time (LIMITED TO <64 IN LANDMARK2HASH)
        self.targetdt = 63
        # global stores duration of most recently-read soundfile
        self.soundfiledur = 0.0
        # .. and total amount of sound processed
        self.soundfiletotaldur = 0.0
        # .. and count of files
        self.soundfilecount = 0


    def spreadpeaksinvector(self, vector, width=4.0):
        """ Create a blurred version of vector, where each of the local maxes
            is spread by a gaussian with SD <width>.
        """
        npts = len(vector)
        peaks = locmax(vector, indices=True)
        return self.spreadpeaks(zip(peaks, vector[peaks]),
                                npoints=npts, width=width)

    def spreadpeaks(self, peaks, npoints=None, width=4.0, base=None):
        """ Generate a vector consisting of the max of a set of Gaussian bumps
        :params:
          peaks : list
            list of (index, value) pairs giving the center point and height
            of each gaussian
          npoints : int
            the length of the output vector (needed if base not provided)
          width : float
            the half-width of the Gaussians to lay down at each point
          base : np.array
            optional initial lower bound to place Gaussians above
        :returns:
          vector : np.array(npoints)
            the maximum across all the scaled Gaussians
        """
        if base is None:
            vec = np.zeros(npoints)
        else:
            npoints = len(base)
            vec = np.copy(base)
        #binvals = np.arange(len(vec))
        #for pos, val in peaks:
        #   vec = np.maximum(vec, val*np.exp(-0.5*(((binvals - pos)
        #                                /float(width))**2)))
        if width != self.__sp_width or npoints != self.__sp_len:
            # Need to calculate new vector
            self.__sp_width = width
            self.__sp_len = npoints
            self.__sp_vals = np.exp(-0.5*((np.arange(-npoints, npoints+1)
                                           / float(width))**2))
        # Now the actual function
        for pos, val in peaks:
            vec = np.maximum(vec, val*self.__sp_vals[np.arange(npoints)
                                                     + npoints - pos])
        return vec

    def _fp_fwd(self, sgram, a_dec):
        """ forward pass of findpeaks
            initial threshold envelope based on peaks in first 10 frames
        """
        (srows, scols) = np.shape(sgram)
        sthresh = self.spreadpeaksinvector(
            np.max(sgram[:, :np.minimum(10, scols)], axis=1), self.f_sd
        )
        ## Store sthresh at each column, for debug
        #thr = np.zeros((srows, scols))
        peaks = np.zeros((srows, scols))
        # optimization of mask update
        __sp_pts = len(sthresh)
        __sp_v = self.__sp_vals

        for col in range(scols):
            s_col = sgram[:, col]
            # Find local magnitude peaks that are above threshold
            sdmaxposs = np.nonzero(locmax(s_col) * (s_col > sthresh))[0]
            # Work down list of peaks in order of their absolute value
            # above threshold
            valspeaks = sorted(zip(s_col[sdmaxposs], sdmaxposs), reverse=True)
            for val, peakpos in valspeaks[:self.maxpksperframe]:
                # What we actually want
                #sthresh = spreadpeaks([(peakpos, s_col[peakpos])],
                #                      base=sthresh, width=f_sd)
                # Optimization - inline the core function within spreadpeaks
                sthresh = np.maximum(sthresh,
                                     val*__sp_v[(__sp_pts - peakpos):
                                                (2*__sp_pts - peakpos)])
                peaks[peakpos, col] = 1
            sthresh *= a_dec
        return peaks

    def _fp_bwd(self, sgram, peaks, a_dec):
        """ backwards pass of findpeaks """
        scols = np.shape(sgram)[1]
        # Backwards filter to prune peaks
        sthresh = self.spreadpeaksinvector(sgram[:, -1], self.f_sd)
        for col in range(scols, 0, -1):
            pkposs = np.nonzero(peaks[:, col-1])[0]
            peakvals = sgram[pkposs, col-1]
            for val, peakpos in sorted(zip(peakvals, pkposs), reverse=True):
                if val >= sthresh[peakpos]:
                    # Setup the threshold
                    sthresh = self.spreadpeaks([(peakpos, val)], base=sthresh,
                                               width=self.f_sd)
                    # Delete any following peak (threshold should, but be sure)
                    if col < scols:
                        peaks[peakpos, col] = 0
                else:
                    # delete the peak
                    peaks[peakpos, col-1] = 0
            sthresh = a_dec*sthresh
        return peaks

    def find_peaks(self, d, sr):
        """ Find the local peaks in the spectrogram as basis for fingerprints.
            Returns a list of (time_frame, freq_bin) pairs.

        :params:
          d - np.array of float
            Input waveform as 1D vector

          sr - int
            Sampling rate of d (not used)

        :returns:
          pklist - list of (int, int)
            Ordered list of landmark peaks found in STFT.  First value of
            each pair is the time index (in STFT frames, i.e., units of
            n_hop/sr secs), second is the FFT bin (in units of sr/n_fft
            Hz).
        """
        # masking envelope decay constant
        a_dec = (1.0 - 0.01*(self.density*np.sqrt(self.n_hop/352.8)/35.0)) \
                **(1.0/OVERSAMP)
        # Take spectrogram
        mywin = np.hanning(self.n_fft+2)[1:-1]
        sgram = np.abs(librosa.stft(d, n_fft=self.n_fft,
                                    hop_length=self.n_hop,
                                    window=mywin))
        sgram = np.log(np.maximum(sgram, np.max(sgram)/1e6))
        sgram = sgram - np.mean(sgram)
        # High-pass filter onset emphasis
        # [:-1,] discards top bin (nyquist) of sgram so bins fit in 8 bits
        sgram = np.array([scipy.signal.lfilter([1, -1],
                                               [1, -(HPF_POLE)** \
                                                (1/OVERSAMP)], s_row)
                          for s_row in sgram])[:-1,]

        peaks = self._fp_fwd(sgram, a_dec)

        peaks = self._fp_bwd(sgram, peaks, a_dec)

        # build a list of peaks
        scols = np.shape(sgram)[1]
        pklist = [[] for _ in xrange(scols)]
        for col in xrange(scols):
            pklist[col] = np.nonzero(peaks[:, col])[0]
        return pklist

    def peaks2landmarks(self, pklist):
        """ Take a list of local peaks in spectrogram
            and form them into pairs as landmarks.
            peaks[i] is a list of the fft bins identified as landmark peaks
            for time frame i (which will be empty for many frames).
            Return a list of (col, peak, peak2, col2-col) landmark descriptors.
        """
        # Form pairs of peaks into landmarks
        scols = len(pklist)
        # Build list of landmarks <starttime F1 endtime F2>
        landmarks = []
        for col in xrange(scols):
            for peak in pklist[col]:
                pairsthispeak = 0
                for col2 in xrange(col+self.mindt,
                                   min(scols, col+self.targetdt)):
                    if pairsthispeak < self.maxpairsperpeak:
                        for peak2 in pklist[col2]:
                            if abs(peak2-peak) < self.targetdf:
                                #and abs(peak2-peak) + abs(col2-col) > 2 ):
                                if pairsthispeak < self.maxpairsperpeak:
                                    # We have a pair!
                                    landmarks.append((col, peak,
                                                      peak2, col2-col))
                                    pairsthispeak += 1

        return landmarks

    def wavfile2hashes(self, filename):
        """ Read a soundfile and return its fingerprint hashes as a
            list of (time, hash) pairs.  If specified, resample to sr first.
            shifts > 1 causes hashes to be extracted from multiple shifts of
            waveform, to reduce frame effects.  """
        ext = os.path.splitext(filename)[1]
        if ext == PRECOMPEXT:
            # short-circuit - precomputed fingerprint file
            hashes = hashes_load(filename)
            dur = np.max(hashes, axis=0)[0]*self.n_hop/float(self.target_sr)
        else:
            [d, sr] = librosa.load(filename, sr=self.target_sr)
            # Store duration in a global because it's hard to handle
            dur = float(len(d))/sr
            # Calculate hashes with optional part-frame shifts
            query_hashes = []
            for shift in range(self.shifts):
                shiftsamps = int(float(shift)/self.shifts*self.n_hop)
                query_hashes += landmarks2hashes(
                    self.peaks2landmarks(
                        self.find_peaks(d[shiftsamps:],
                                        sr)
                    )
                )
            # remove duplicate elements by pushing through a set
            hashes = sorted(list(set(query_hashes)))

        # instrumentation to track total amount of sound processed
        self.soundfiledur = dur
        self.soundfiletotaldur += dur
        self.soundfilecount += 1
        #print("wavfile2hashes: read", len(hashes), "hashes from", filename)
        return hashes

    ########### functions to link to actual hash table index database #######

    def ingest(self, hashtable, filename):
        """ Read an audio file and add it to the database
        :params:
          hashtable : HashTable object
            the hash table to add to
          filename : str
            name of the soundfile to add
        :returns:
          dur : float
            the duration of the track
          nhashes : int
            the number of hashes it mapped into
        """
        #sr = 11025
        #print("ingest: sr=",sr)
        #d, sr = librosa.load(filename, sr=sr)
        # librosa.load on mp3 files prepends 396 samples compared
        # to Matlab audioread ??
        #hashes = landmarks2hashes(peaks2landmarks(find_peaks(d, sr,
        #                                                     density=density,
        #                                                     n_fft=n_fft,
        #                                                     n_hop=n_hop)))
        hashes = self.wavfile2hashes(filename)
        hashtable.store(filename, hashes)
        #return (len(d)/float(sr), len(hashes))
        #return (np.max(hashes, axis=0)[0]*n_hop/float(sr), len(hashes))
        # soundfiledur is set up in wavfile2hashes, use result here
        return self.soundfiledur, len(hashes)



########### functions to read/write hashes to file for a single track #####

# Format string for writing binary data to file
HASH_FMT = '<2i'
HASH_MAGIC = 'audfprinthashV00'  # 16 chars, FWIW

def hashes_save(hashfilename, hashes):
    """ Write out a list of (time, hash) pairs as 32 bit ints """
    with open(hashfilename, 'wb') as f:
        f.write(HASH_MAGIC)
        for time_, hash_ in hashes:
            f.write(struct.pack(HASH_FMT, time_, hash_))

def hashes_load(hashfilename):
    """ Read back a set of hashes written by hashes_save """
    hashes = []
    fmtsize = struct.calcsize(HASH_FMT)
    with open(hashfilename, 'rb') as f:
        magic = f.read(len(HASH_MAGIC))
        if magic != HASH_MAGIC:
            raise IOError('%s is not a hash file (magic %s)'
                          % (hashfilename, magic))
        data = f.read(fmtsize)
        while data is not None and len(data) == fmtsize:
            hashes.append(struct.unpack(HASH_FMT, data))
            data = f.read(fmtsize)
    return hashes


######## function signature for Gordon feature extraction
######## which stores the precalculated hashes for each track separately

extract_features_analyzer = None

def extract_features(track_obj, *args, **kwargs):
    """ Extract the audfprint fingerprint hashes for one file.
    :params:
      track_obj : object
        Gordon's internal structure defining a track; we use
        track_obj.fn_audio to find the actual audio file.
    :returns:
      hashes : list of (int, int)
        The times (in frames) and hashes analyzed from the audio file.
    """
    global extract_features_analyzer
    if extract_features_analyzer == None:
        extract_features_analyzer = Analyzer()

    density = None
    n_fft = None
    n_hop = None
    sr = None
    if "density" in kwargs:
        density = kwargs["density"]
    if "n_fft" in kwargs:
        n_fft = kwargs["n_fft"]
    if "n_hop" in kwargs:
        n_hop = kwargs["n_hop"]
    if "sr" in kwargs:
        sr = kwargs["sr"]
    extract_features_analyzer.density = density
    extract_features_analyzer.n_fft = n_fft
    extract_features_analyzer.n_hop = n_hop
    extract_features_analyzer.target_sr = sr
    return extract_features_analyzer.wavfile2hashes(track_obj.fn_audio)


# Handy function to build a new hash table from a file glob pattern
g2h_analyzer = None

def glob2hashtable(pattern, density=20.0):
    """ Build a hash table from the files matching a glob pattern """
    global g2h_analyzer
    if g2h_analyzer == None:
        g2h_analyzer = Analyzer(density=density)

    ht = hash_table.HashTable()
    filelist = glob.glob(pattern)
    initticks = time.clock()
    totdur = 0.0
    tothashes = 0
    for ix, file_ in enumerate(filelist):
        print(time.ctime(), "ingesting #", ix, ":", file_, "...")
        dur, nhash = g2h_analyzer.ingest(ht, file_)
        totdur += dur
        tothashes += nhash
    elapsedtime = time.clock() - initticks
    print("Added", tothashes, "(", tothashes/float(totdur), "hashes/sec) at ",
          elapsedtime/totdur, "x RT")
    return ht

DO_TEST = False
if DO_TEST:
    test_fn = '/Users/dpwe/Downloads/carol11k.wav'
    test_ht = hash_table.HashTable()
    test_analyzer = Analyzer()

    test_analyzer.ingest(test_ht, test_fn)
    test_ht.save('httest.pklz')

def filenames(filelist, wavdir, listflag):
    """ Iterator to yeild all the filenames, possibly interpreting them
        as list files, prepending wavdir """
    if not listflag:
        for filename in filelist:
            yield os.path.join(wavdir, filename)
    else:
        for listfilename in filelist:
            with open(listfilename, 'r') as f:
                for filename in f:
                    yield os.path.join(wavdir, filename.rstrip('\n'))

# for saving precomputed fprints
def ensure_dir(fname):
    """ ensure that the directory for the named path exists """
    head = os.path.split(fname)[0]
    if len(head):
        if not os.path.exists(head):
            os.makedirs(head)

# Command line interface

# basic operations, each in a separate function

def file_precompute(analyzer, filename, precompdir, precompext=PRECOMPEXT):
    """ Perform precompute action for one file, return list
        of message strings """
    hashes = analyzer.wavfile2hashes(filename)
    # strip relative directory components from file name
    # Also remove leading absolute path (comp == '')
    relname = '/'.join([comp for comp in filename.split('/')
                        if comp != '.' and comp != '..' and comp != ''])
    root = os.path.splitext(relname)[0]
    opfname = os.path.join(precompdir, root+precompext)
    # Make sure the directory exists
    ensure_dir(opfname)
    # save the hashes file
    hashes_save(opfname, hashes)
    return ["wrote " + opfname + " ( %d hashes, %.3f sec)" \
                                   % (len(hashes), analyzer.soundfiledur)]

def make_ht_from_list(analyzer, filelist, proto_hash_tab, pipe=None):
    """ Populate a hash table from a list, used as target for
        multiprocess division.  pipe is a pipe over which to push back
        the result, else return it """
    # Clone HT params
    hashbits = proto_hash_tab.hashbits
    depth = proto_hash_tab.depth
    maxtime = proto_hash_tab.maxtime
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

def do_cmd(cmd, analyzer, hash_tab, filename_iter, matcher, outdir, report):
    """ Breaks out the core part of running the command.
        This is just the single-core versions.
    """
    if cmd == 'merge':
        # files are other hash tables, merge them in
        for filename in filename_iter:
            hash_tab2 = hash_table.HashTable(filename)
            hash_tab.merge(hash_tab2)

    elif cmd == 'precompute':
        # just precompute fingerprints, single core
        for filename in filename_iter:
            report(file_precompute(analyzer, filename,
                                   outdir, PRECOMPEXT))

    elif cmd == 'match':
        # Running query, single-core mode
        for filename in filename_iter:
            msgs = matcher.file_match_to_msgs(analyzer, hash_tab, filename)
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

        report(["Added " +  str(tothashes) + " hashes "
                + "(%.1f" % (tothashes/float(analyzer.soundfiletotaldur))
                + " hashes/sec)"])
    else:
        raise ValueError("unrecognized command: "+cmd)

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
                                               hash_tab, tx[ix]))
        pr[ix].start()
    # gather results when they all finish
    for core in range(ncores):
        # thread passes back serialized hash table structure
        hash_tabx = rx[core].recv()
        report(["hash_table " + str(core) + " has "
                + str(len(hash_tabx.names))
                + " files " + str(sum(hash_tabx.counts)) + " hashes"])
        if len(hash_tab.counts) == 0:
            # Avoid merge into empty hash table, just keep the first one
            hash_tab = hash_tabx
        else:
            # merge in all the new items, hash entries
            hash_tab.merge(hash_tabx)
        # finish that thread...
        pr[core].join()


def matcher_file_match_to_msgs(matcher, analyzer, hash_tab, filename):
    """Cover for matcher.file_match_to_msgs so it can be passed to joblib"""
    return matcher.file_match_to_msgs(analyzer, hash_tab, filename)

def do_cmd_multiproc(cmd, analyzer, hash_tab, filename_iter, matcher,
                     outdir, report, ncores):
    """ Run the actual command, using multiple processors """
    if cmd == 'precompute':
        # precompute fingerprints with joblib
        msgslist = joblib.Parallel(n_jobs=ncores)(
            joblib.delayed(file_precompute)(analyzer, file, outdir,
                                            PRECOMPEXT)
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
        raise ValueError("unrecognized multiproc command: "+cmd)

# Command to separate out setting of analyzer parameters
def setup_analyzer(args):
    """Create a new analyzer object, taking values from docopts args"""
    # Create analyzer object; parameters will get set below
    analyzer = Analyzer()
    # Read parameters from command line/docopts
    analyzer.density = float(args['--density'])
    analyzer.maxpksperframe = int(args['--pks-per-frame'])
    analyzer.maxpairsperpeak = int(args['--fanout'])
    analyzer.f_sd = float(args['--freq-sd'])
    analyzer.shifts = int(args['--shifts'])
    # fixed - 512 pt FFT with 256 pt hop at 11025 Hz
    analyzer.target_sr = int(args['--samplerate'])
    analyzer.n_fft = 512
    analyzer.n_hop = analyzer.n_fft/2
    # set default value for shifts depending on mode
    if analyzer.shifts == 0:
        # Default shift is 4 for match, otherwise 1
        analyzer.shifts = 4 if args['match'] else 1
    return analyzer

# Command to separate out setting of matcher parameters
def setup_matcher(args):
    """Create a new matcher objects, set parameters from docopt structure"""
    matcher = audfprint_match.Matcher()
    matcher.window = int(args['--match-win'])
    matcher.threshold = int(args['--min-count'])
    matcher.max_returns = int(args['--max-matches'])
    matcher.sort_by_time = args['--sortbytime']
    matcher.illustrate = args['--illustrate']
    matcher.verbose = args['--verbose']
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
                f.write(msg+"\n")
    else:
        def report(msglist):
            """Log messages by printing to stdout"""
            for msg in msglist:
                print(msg)
    return report

# CLI specified via usage message thanks to docopt
USAGE = """
Audio landmark-based fingerprinting.
Create a new fingerprint dbase with new,
append new files to an existing database with add,
or identify noisy query excerpts with match.
"Precompute" writes a *.fpt file under fptdir with
precomputed fingerprint for each input wav file.

Usage: audfprint (new | add | match | precompute | merge) [options] <file>...

Options:
  -d <dbase>, --dbase <dbase>     Fingerprint database file
  -n <dens>, --density <dens>     Target hashes per second [default: 20.0]
  -h <bits>, --hashbits <bits>    How many bits in each hash [default: 20]
  -b <val>, --bucketsize <val>    Number of entries per bucket [default: 100]
  -t <val>, --maxtime <val>       Largest time value stored [default: 16384]
  -r <val>, --samplerate <val>    Resample input files to this [default: 11025]
  -p <dir>, --precompdir <dir>    Save precomputed files under this dir [default: .]
  -i <val>, --shifts <val>        Use this many subframe shifts building fp [default: 0]
  -w <val>, --match-win <val>     Maximum tolerable frame skew to count as a matlch [default: 1]
  -N <val>, --min-count <val>     Minimum number of matching landmarks to count as a match [default: 5]
  -x <val>, --max-matches <val>   Maximum number of matches to report for each query [default: 1]
  -S <val>, --freq-sd <val>       Frequency peak spreading SD in bins [default: 30.0]
  -F <val>, --fanout <val>        Max number of hash pairs per peak [default: 3]
  -P <val>, --pks-per-frame <val>  Maximum number of peaks per frame [default: 5]
  -H <val>, --ncores <val>        Number of processes to use [default: 1]
  -o <name>, --opfile <name>      Write output (matches) to this file, not stdout [default: ]
  -l, --list                      Input files are lists, not audio
  -T, --sortbytime                Sort multiple hits per file by time (instead of score)
  -v <val>, --verbose <val>       Verbosity level [default: 1]
  -I, --illustrate                Make a plot showing the match
  -W <dir>, --wavdir <dir>        Find sound files under this dir [default: ]
  --version                       Report version number
  --help                          Print this message
"""

__version__ = 20140906

def main(argv):
    """ Main routine for the command-line interface to audfprint """
    # Other globals set from command line
    args = docopt.docopt(USAGE, version=__version__, argv=argv[1:])

    # Figure which command was chosen
    poss_cmds = ['new', 'add', 'precompute', 'merge', 'match']
    cmdlist = [cmdname
               for cmdname in poss_cmds
               if args[cmdname]]
    if len(cmdlist) != 1:
        raise ValueError("must specify exactly one command")
    # The actual command as a str
    cmd = cmdlist[0]

    # Setup the analyzer if we're using one (i.e., unless "merge")
    analyzer = setup_analyzer(args) if cmd is not "merge" else None

    # Set up the hash table, if we're using one (i.e., unless "precompute")
    if cmd is not "precompute":
        # For everything other than precompute, we need a database name
        # Check we have one
        dbasename = args['--dbase']
        if not dbasename:
            raise ValueError("dbase name must be provided if not precompute")
        if cmd == "new":
            # Create a new hash table
            hash_tab = hash_table.HashTable(hashbits=int(args['--hashbits']),
                                            depth=int(args['--bucketsize']),
                                            maxtime=int(args['--maxtime']))
            # Set its samplerate param
            hash_tab.params['samplerate'] = analyzer.target_sr
        else:
            # Load existing hash table file (add, match, merge)
            hash_tab = hash_table.HashTable(dbasename)
            if analyzer and 'samplerate' in hash_tab.params \
                   and hash_tab.params['samplerate'] != analyzer.target_sr:
                analyzer.target_sr = hash_tab.params['samplerate']
                print("samplerate set to", analyzer.target_sr,
                      "per", dbasename)
    else:
        # The command IS precompute
        # dummy empty hash table
        hash_tab = None

    # Setup output function
    report = setup_reporter(args)

    # Keep track of wall time
    initticks = time.clock()

    # Create a matcher
    matcher = setup_matcher(args) if cmd == 'match' else None

    filename_iter = filenames(args['<file>'],
                              args['--wavdir'],
                              args['--list'])

    #######################
    # Run the main commmand
    #######################

    # How many processors to use (multiprocessing)
    ncores = int(args['--ncores'])
    if ncores > 1 and cmd != "merge":
        # "merge" is always a single-thread process
        do_cmd_multiproc(cmd, analyzer, hash_tab, filename_iter,
                         matcher, args['--precompdir'], report, ncores)
    else:
        do_cmd(cmd, analyzer, hash_tab, filename_iter,
               matcher, args['--precompdir'], report)

    elapsedtime = time.clock() - initticks
    if analyzer and analyzer.soundfiletotaldur > 0.:
        print("Processed "
              + "%d files (%.1f s total dur) in %.1f s sec = %.3f x RT" \
              % (analyzer.soundfilecount, analyzer.soundfiletotaldur,
                 elapsedtime, (elapsedtime/analyzer.soundfiletotaldur)))

    # Save the hash table file if it has been modified
    if hash_tab and hash_tab.dirty:
        hash_tab.save(dbasename)


# Run the main function if called from the command line
if __name__ == "__main__":
    main(sys.argv)
