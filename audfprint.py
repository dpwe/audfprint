"""
audfprint.py

Implementation of acoustic-landmark-based robust fingerprinting.
Port of the Matlab implementation.

2014-05-25 Dan Ellis dpwe@ee.columbia.edu
"""

import numpy as np
import matplotlib.pyplot as plt
import librosa
import scipy.signal

import hash_table

def locmax(x, indices=False):
    """ Return a boolean vector of which points in x are local maxima.  
        End points are peaks if larger than single neighbors.
        if indices=True, return the indices of the True values instead 
        of the boolean vector. 
    """
    # x[-1]-1 means last value can be a peak
    #nbr = np.greater_equal(np.r_[x, x[-1]-1], np.r_[x[0], x])
    # the np.r_ was killing us, so try an optimization...
    nbr = np.zeros(len(x)+1, dtype=bool)
    nbr[0] = True
    nbr[1:-1] = np.greater_equal(x[1:], x[:-1])
    maxmask = (nbr[:-1] & ~nbr[1:])
    if indices:
        return np.nonzero(maxmask)[0]
    else:
        return maxmask

class Analyzer:
    # Parameters
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

    density = None
    n_fft = None
    n_hop = None
    # spectrogram enhancement
    hpf_pole = 0.98

    # How many offsets to provide in analysis
    shifts = 1

    # how wide to spreak peaks
    f_sd = None
    # Maximum number of local maxima to keep per frame
    maxpksperframe = None
    # Fanout
    maxpairsperpeak = None  # Limit the number of pairs we'll make from each peak

    # Values controlling peaks2landmarks
    targetdf = 31   # +/- 50 bins in freq (LIMITED TO -32..31 IN LANDMARK2HASH)
    mindt    = 2    # min time separation (traditionally 1, upped 2014-08-04)
    targetdt = 63   # max lookahead in time (LIMITED TO <64 IN LANDMARK2HASH)

    # optimization: cache pre-calculated Gaussian profile
    __sp_width = None
    __sp_len = None
    __sp_vals = []

    # Globals defining packing of landmarks into hashes
    F1bits = 8
    DFbits = 6
    DTbits = 6

    b1mask  = (1 << F1bits) - 1
    b1shift = DFbits + DTbits
    dfmask  = (1 << DFbits) - 1
    dfshift = DTbits
    dtmask  = (1 << DTbits) - 1

    # Special extension indicating precomputed fingerprint
    precompext = '.afpt'

    # global stores duration of most recently-read soundfile
    soundfiledur = 0.0
    # .. and total amount of sound processed
    soundfiletotaldur = 0.0
    # .. and count of files
    soundfilecount = 0

    def __init__(self, f_sd=30.0, maxpksperframe=5, maxpairsperpeak=3, density=DENSITY, target_sr=11025, n_fft=N_FFT, n_hop=N_HOP, shifts=1):
        self.f_sd = f_sd
        self.maxpksperframe = maxpksperframe
        self.maxpairsperpeak = maxpairsperpeak
        self.density = density
        self.target_sr = target_sr
        self.n_fft = n_fft
        self.n_hop = n_hop
        self.shifts = shifts

    def spreadpeaksinvector(self, vector, width=4.0):
        """ Create a blurred version of vector, where each of the local maxes 
            is spread by a gaussian with SD <width>.
        """
        npts = len(vector)
        peaks = locmax(vector, indices=True)
        return self.spreadpeaks(zip(peaks, vector[peaks]), npoints=npts, width=width)

    def spreadpeaks(self, peaks, npoints=None, width=4.0, base=None):
        """ Generate a vector consisting of the max of a set of Gaussian bumps
        :params:
          peaks : list
            list of (index, value) pairs giving the center point and height of each gaussian
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
            Y = np.zeros(npoints)
        else:
            npoints = len(base)
            Y = np.copy(base)
        #binvals = np.arange(len(Y))
        #for pos, val in peaks:
        #   Y = np.maximum(Y, val*np.exp(-0.5*(((binvals - pos)/float(width))**2)))
        if width != self.__sp_width or npoints != self.__sp_len:
            # Need to calculate new vector
            self.__sp_width = width
            self.__sp_len = npoints
            self.__sp_vals = np.exp(-0.5*((np.arange(-npoints, npoints+1)/float(width))**2))
        # Now the actual function
        for pos, val in peaks:
            Y = np.maximum(Y, val*self.__sp_vals[np.arange(npoints) + npoints - pos])
        return Y

    def fp_fwd(self, S, a_dec):
        # forward pass of findpeaks
        # initial threshold envelope based on peaks in first 10 frames
        (srows, scols) = np.shape(S)
        sthresh = self.spreadpeaksinvector(np.max(S[:, :np.minimum(10, scols)], axis=1), 
                                           self.f_sd)
        ## Store sthresh at each column, for debug
        #thr = np.zeros((srows, scols))
        peaks = np.zeros((srows, scols))
        # optimization of mask update
        __sp_pts = len(sthresh)
        __sp_v = self.__sp_vals

        for col in range(scols):
            Scol = S[:, col]
            # Find local magnitude peaks that are above threshold
            sdmaxposs = np.nonzero(locmax(Scol) * (Scol>sthresh))[0]
            # Work down list of peaks in order of their absolute value above threshold
            valspeaks = sorted(zip(Scol[sdmaxposs], sdmaxposs), reverse=True)
            for val, peakpos in valspeaks[:self.maxpksperframe]:
                # What we actually want
                #sthresh = spreadpeaks([(peakpos, Scol[peakpos])], base=sthresh, width=f_sd)
                # Optimization - inline the core function within spreadpeaks
                sthresh = np.maximum(sthresh, val*__sp_v[(__sp_pts - peakpos):(2*__sp_pts - peakpos)])
                peaks[peakpos, col] = 1
            sthresh *= a_dec
        return peaks

    def fp_bwd(self, S, peaks, a_dec):
        # backwards pass of findpeaks
        (srows, scols) = np.shape(S)
        # Backwards filter to prune peaks
        sthresh = self.spreadpeaksinvector(S[:,-1], self.f_sd)
        for col in range(scols, 0, -1):
            pkposs = np.nonzero(peaks[:, col-1])[0]
            peakvals = S[pkposs, col-1]
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
            Sampling rate of d

        :returns:
          pklist - list of (int, int)
            Ordered list of landmark peaks found in STFT.  First value of
            each pair is the time index (in STFT frames, i.e., units of 
            n_hop/sr secs), second is the FFT bin (in units of sr/n_fft 
            Hz).
        """
        # masking envelope decay constant
        a_dec = (1.0 - 0.01*(self.density*np.sqrt(self.n_hop/352.8)/35.0))**(1.0/self.OVERSAMP)
        # Take spectrogram
        mywin = np.hanning(self.n_fft+2)[1:-1]
        S = np.abs(librosa.stft(d, n_fft=self.n_fft, hop_length=self.n_hop, window=mywin))
        S = np.log(np.maximum(S, np.max(S)/1e6))
        S = S - np.mean(S)
        # High-pass filter onset emphasis
        # [:-1,] discards top bin (nyquist) of sgram so bins fit in 8 bits
        S = np.array([scipy.signal.lfilter([1, -1], 
                                           [1, -(self.hpf_pole)**(1/self.OVERSAMP)], Srow) 
                      for Srow in S])[:-1,]

        peaks = self.fp_fwd(S, a_dec)

        peaks = self.fp_bwd(S, peaks, a_dec)

        # build a list of peaks
        (srows, scols) = np.shape(S)
        pklist = [[] for _ in xrange(scols)]
        for col in xrange(scols):
            pklist[col] = np.nonzero(peaks[:,col])[0]
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
                for col2 in xrange(col+self.mindt, min(scols, col+self.targetdt)):
                    if (pairsthispeak < self.maxpairsperpeak):
                        for peak2 in pklist[col2]:
                            if abs(peak2-peak) < self.targetdf :
                                #and abs(peak2-peak) + abs(col2-col) > 2 ):
                                if (pairsthispeak < self.maxpairsperpeak):
                                    # We have a pair!
                                    landmarks.append( (col, peak, peak2, col2-col) )
                                    pairsthispeak += 1

        return landmarks


    def landmarks2hashes(self, landmarks):
        """ Convert a list of (time, bin1, bin2, dtime) landmarks 
            into a list of (time, hash) pairs where the hash combines 
            the three remaining values.
        """
        # build up and return the list of hashed values
        return [ ( time, 
                   ( ((bin1 & self.b1mask) << self.b1shift) 
                     | (((bin2 - bin1) & self.dfmask) << self.dfshift)
                     | (dtime & self.dtmask)) ) 
                 for time, bin1, bin2, dtime in landmarks ]

    def hashes2landmarks(self, hashes):
        """ Convert the mashed-up landmarks in hashes back into a list 
            of (time, bin1, bin2, dtime) tuples.
        """
        landmarks = []
        for time, hash in hashes:
            dtime = hash & self.dtmask
            bin1 = (hash >> self.b1shift) & self.b1mask
            dbin = (hash >> self.dfshift) & self.dfmask
            # Sign extend frequency difference
            if dbin > (1 << (self.DFbits-1)):
                dbin -= (1 << self.DFbits)
            landmarks.append( (time, bin1, bin1+dbin, dtime) )
        return landmarks

    def wavfile2hashes(self, filename):
        """ Read a soundfile and return its fingerprint hashes as a 
            list of (time, hash) pairs.  If specified, resample to sr first. 
            shifts > 1 causes hashes to be extracted from multiple shifts of 
            waveform, to reduce frame effects.  """
        root, ext = os.path.splitext(filename)
        if ext == self.precompext:
            # short-circuit - precomputed fingerprint file
            hashes = self.hashes_load(filename)
        else:
            [d, sr] = librosa.load(filename, sr=self.target_sr)
            # Store duration in a global because it's hard to handle
            dur = float(len(d))/sr
            self.soundfiledur = dur
            # instrumentation to track total amount of sound processed
            self.soundfiletotaldur += dur
            self.soundfilecount += 1
            # Calculate hashes with optional part-frame shifts
            hq = []
            for shift in range(self.shifts):
                shiftsamps = int(float(shift)/self.shifts*self.n_hop)
                hq += self.landmarks2hashes(self.peaks2landmarks(
                                               self.find_peaks(d[shiftsamps:], sr)))
            # remove duplicate elements by pushing through a set
            hashes = sorted(list(set(hq)))

        #print "wavfile2hashes: read", len(hashes), "hashes from", filename
        return hashes

    ########### functions to link to actual hash table index database #######

    def ingest(self, ht, filename):
        """ Read an audio file and add it to the database
        :params:
          ht : HashTable object
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
        #print "ingest: sr=",sr
        #d, sr = librosa.load(filename, sr=sr)
        # librosa.load on mp3 files prepends 396 samples compared 
        # to Matlab audioread ??
        #hashes = landmarks2hashes(peaks2landmarks(find_peaks(d, sr, 
        #                                                     density=density, 
        #                                                     n_fft=n_fft, 
        #                                                     n_hop=n_hop)))
        hashes = self.wavfile2hashes(filename)
        ht.store(filename, hashes)
        #return (len(d)/float(sr), len(hashes))
        #return (np.max(hashes, axis=0)[0]*n_hop/float(sr), len(hashes))
        # soundfiledur is set up in wavfile2hashes, use result here
        return self.soundfiledur, len(hashes)



########### functions to read/write hashes to file for a single track #####

import struct

# Format string for writing binary data to file
hash_fmt = '<2i'
hash_magic = 'audfprinthashV00'  # 16 chars, FWIW

def hashes_save(hashfilename, hashes):
    """ Write out a list of (time, hash) pairs as 32 bit ints """
    with open(hashfilename, 'wb') as f:
        f.write(hash_magic)
        for time, hash in hashes:
            f.write(struct.pack(hash_fmt, time, hash))

def hashes_load(hashfilename):
    """ Read back a set of hashes written by hashes_save """
    hashes = [];
    fmtsize = struct.calcsize(hash_fmt)
    with open(hashfilename, 'rb') as f:
        magic = f.read(len(hash_magic))
        if magic != hash_magic:
            raise IOError('%s is not a hash file (magic %s)' 
                          % (hashfilename, magic) )
        data = f.read(fmtsize)
        while data is not None and len(data) == fmtsize:
            hashes.append(struct.unpack(hash_fmt, data))
            data = f.read(fmtsize)
    return hashes


######## function signature for Gordon feature extraction
######## which stores the precalculated hashes for each track separately

analyzer = None

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
    global analyzer
    if analyzer == None:
        analyzer = Analyzer()

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
    analyzer.density = density
    analyzer.n_fft = n_fft
    analyzer.n_hop = n_hop
    analyzer.target_sr = sr
    return analyzer.wavfile2hashes(track_obj.fn_audio)


# Handy function to build a new hash table from a file glob pattern
import glob, time
def glob2hashtable(pattern, density=20.0):
    """ Build a hash table from the files matching a glob pattern """
    global analyzer
    if analyzer == None:
        analyzer = Analyzer(density=density)

    ht = hash_table.HashTable()
    filelist = glob.glob(pattern)
    initticks = time.clock()
    totdur = 0.0
    tothashes = 0
    for ix, file in enumerate(filelist):
        print time.ctime(), "ingesting #", ix, ":", file, "..."
        dur, nhash = analyzer.ingest(ht, file)
        totdur += dur
        tothashes += nhash
    elapsedtime = time.clock() - initticks
    print "Added",tothashes,"(",tothashes/float(totdur),"hashes/sec) at ", elapsedtime/totdur, "x RT"
    return ht

test = False
if test:
    fn = '/Users/dpwe/Downloads/carol11k.wav'
    ht = hash_table.HashTable()
    analyzer = Analyzer()

    analyzer.ingest(ht, fn)
    ht.save('httest.pklz')

def filenames(filelist, wavdir, listflag):
  """ Iterator to yeild all the filenames, possibly interpreting them as list files, prepending wavdir """
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
    head, tail = os.path.split(fname)
    if len(head):
        if not os.path.exists(head):
            os.makedirs(head)

# Command line interface

import audfprint_match
import docopt
import time 
import os

# basic operations, each in a separate function

def file_match(analyzer, ht, qry, match_win, min_count, max_matches, sortbytime, illustrate, verbose):
    """ Perform a match on a single input file, return result string """
    rslts, dur, nhash = audfprint_match.match_file(analyzer, ht, qry, 
                                                   window=match_win, 
                                                   threshcount=min_count, 
                                                   verbose=verbose)
    t_hop = analyzer.n_hop/float(analyzer.target_sr)
    qrymsg = qry + (' %.3f '%dur) + "sec " + str(nhash) + " raw hashes"
    msgrslt = []
    if len(rslts) == 0:
        # No matches returned at all
        nhashaligned = 0
        msgrslt.append("NOMATCH "+qrymsg)
    else:
        if sortbytime:
            rslts = sorted(rslts, key=lambda x:-x[2])
        for hitix in range(min(len(rslts), max_matches)):
            # figure the number of raw and aligned matches for top hit
            tophitid, nhashaligned, bestaligntime, nhashraw = rslts[hitix]
            msgrslt.append("Matched " + qrymsg + " as " + ht.names[tophitid] \
                           + (" at %.3f " % (bestaligntime*t_hop)) + "s " \
                           + "with " + str(nhashaligned) + " of " + str(nhashraw) + " hashes")
            if illustrate:
                audfprint_match.illustrate_match(analyzer, ht, qry, 
                                                 window=match_win, 
                                                 sortbytime=sortbytime)
    return msgrslt

def file_precompute(analyzer, file, precompdir, precompext):
    """ Perform precompute action for one file """
    hashes = analyzer.wavfile2hashes(file)
    # strip relative directory components from file name
    # Also remove leading absolute path (comp == '')
    relname = '/'.join([comp for comp in file.split('/') 
                        if comp != '.' and comp != '..' and comp != ''])
    root, ext = os.path.splitext(relname)
    opfname = os.path.join(precompdir, root+precompext)
    # Make sure the directory exists
    ensure_dir(opfname)
    # save the hashes file
    hashes_save(opfname, hashes)
    return "wrote " + opfname + " ( %d hashes, %.3f sec)" % (len(hashes), analyzer.soundfiledur)

def hash_sender(filename_generator, analyzer, hqueue):
    """ function that is run in a separate thread to generate the hashes for a sequence of files and push them onto hqueue """
    ix = 0
    for filename in filename_generator:
        hashes = analyzer.wavfile2hashes(filename)
        #print "hash_sender: sending", filename, "and", len(hashes), "hashes"
        #hqueue.put_nowait( (filename, hashes, analyzer.soundfiledur) )
        hqueue.send( (filename, hashes, analyzer.soundfiledur) )
        ix += 1
    # Send empty data to indicate end
    #hqueue.put_nowait( (None, None, 0.0) )  # Keep pushing data into pipe
    hqueue.send( (None, None, 0.0) )  # Keep pushing data into pipe
    # Close out
    #hqueue.close()
    #hqueue.join_thread()

def make_ht_from_list(analyzer, filelist, ht, pipe):
    """ Populate a hash table from a list """
    for filename in filelist:
        hashes = analyzer.wavfile2hashes(filename)
        ht.store(filename, hashes)
    if pipe:
        pipe.send(ht)
    else:
        return ht

usage = """
Audio landmark-based fingerprinting.  
Create a new fingerprint dbase with new, 
append new files to an existing database with add, 
or identify noisy query excerpts with match.
"Precompute" writes a *.fpt file under fptdir with 
precomputed fingerprint for each input wav file.

Usage: audfprint (new | add | match | precompute | merge) [-d <dbase> | --dbase <dbase>] [options] <file>...

Options:
  -n <dens>, --density <dens>     Target hashes per second [default: 20.0]
  -h <bits>, --hashbits <bits>    How many bits in each hash [default: 20]
  -b <val>, --bucketsize <val>    Number of entries per bucket [default: 100]
  -t <val>, --maxtime <val>       Largest time value stored [default: 16384]
  -r <val>, --samplerate <val>    Resample input files to this [default: 11025]
  -p <dir>, --precompdir <dir>    Save precomputed files under this dir [default: .]
  -i <val>, --shifts <val>        Use this many subframe shifts building fp [default: 0]
  -w <val>, --match-win <val>     Maximum tolerable frame skew to count as a match [default: 1]
  -N <val>, --min-count <val>     Minimum number of matching landmarks to count as a match [default: 5]
  -x <val>, --max-matches <val>   Maximum number of matches to report for each query [default: 1]
  -S <val>, --freq-sd <val>       Frequency peak spreading SD in bins [default: 30.0]
  -F <val>, --fanout <val>        Max number of hash pairs per peak [default: 3]
  -P <val>, --pks-per-frame <val>   Maximum number of peaks per frame [default: 5]
  -l, --list                      Input files are lists, not audio
  -T, --sortbytime                Sort multiple hits per file by time (instead of score)
  -v, --verbose                   Verbose reporting
  -I, --illustrate                Make a plot showing the match
  -M, --multiproc                 Experimental multi-core support
  -W <dir>, --wavdir <dir>        Find sound files under this dir [default: ]
  --version                       Report version number
  --help                          Print this message
"""

__version__ = 20140802

def main(argv):
    # Other globals set from command line
    args = docopt.docopt(usage, version=__version__, argv=argv[1:]) 
    if args['new']:
        cmd = 'new'
    elif args['add']:
        cmd = 'add'
    elif args['precompute']:
        cmd = 'precompute'
    elif args['merge']:
        cmd = 'merge'
    else:
        cmd = 'match'
    dbasename = args['<dbase>']
    density = float(args['--density'])
    hashbits = int(args['--hashbits'])
    bucketsize = int(args['--bucketsize'])
    maxtime = int(args['--maxtime'])
    samplerate = int(args['--samplerate'])
    listflag = args['--list']
    verbose = args['--verbose']
    illustrate = args['--illustrate']
    multiproc = args['--multiproc']
    sortbytime = args['--sortbytime']
    files = args['<file>']
    precompdir = args['--precompdir']
    shifts = int(args['--shifts'])
    match_win = int(args['--match-win'])
    min_count = int(args['--min-count'])
    max_matches = int(args['--max-matches'])
    wavdir = args['--wavdir']
    maxpksperframe = int(args['--pks-per-frame'])
    maxpairsperpeak = int(args['--fanout'])
    f_sd = float(args['--freq-sd'])

    # fixed - 512 pt FFT with 256 pt hop at 11025 Hz
    target_sr = samplerate # not always 11025, but always n_fft=512
    n_fft = 512
    n_hop = n_fft/2

    # Keep track of wall time
    initticks = time.clock()

    # Create analyzer object
    analyzer = Analyzer(f_sd=f_sd, maxpksperframe=maxpksperframe, maxpairsperpeak=maxpairsperpeak, 
                        density=density, n_fft=n_fft, n_hop=n_hop, target_sr=target_sr)

    if cmd == 'new':
        # Create a new hash table
        ht = hash_table.HashTable(hashbits=hashbits, depth=bucketsize, 
                                  maxtime=maxtime)
    elif cmd != 'precompute':
        # Load existing
        ht = hash_table.HashTable(dbasename)
        if 'samplerate' in ht.params:
            if ht.params['samplerate'] != target_sr:
                target_sr = ht.params['samplerate']
                print "samplerate set to",target_sr,"per",dbasename
    else:
        # dummy empty hash table for precompute
        ht = None

    # set default value for shifts depending on mode
    if shifts == 0:
        if cmd == 'match':
            # Default shifts is 4 for match
            shifts = 4
        else:
            # default shifts is 1 for new/add/precompute
            shifts = 1
    # Store in analyzer
    analyzer.shifts = shifts

    if cmd == 'merge':
        # files are other hash tables, merge them in
        for file in filenames(files, wavdir, listflag):
            ht2 = hash_table.HashTable(file)
            ht.merge(ht2)

    elif multiproc and cmd == 'match':
        # Running queries in parallel
        import joblib
        nthreads = 4
        msgslist = joblib.Parallel(n_jobs=nthreads)(joblib.delayed(file_match)(analyzer, ht, file, match_win, min_count, max_matches, sortbytime, illustrate, verbose) for file in filenames(files, wavdir, listflag))
        print "\n".join(["\n".join(msgs) for msgs in msgslist])

    elif cmd == 'match':
        # Running query
        for file in filenames(files, wavdir, listflag):
            msgs = file_match(analyzer, ht, file, match_win, min_count, 
                              max_matches, sortbytime, illustrate, verbose)
            print "\n".join(msgs)

    elif cmd == 'precompute':
        # just precompute fingerprints
        for file in filenames(files, wavdir, listflag):
            msgs = file_precompute(analyzer, file, precompdir, precompext)
            print "\n".join(msgs)
    
    elif multiproc: # and cmd == "new":
        # run nthreads in parallel to add new files to existing HT
        import multiprocessing
        nthreads = 4
        # lists store per-process parameters
        # Pipes to transfer results
        rx = [[] for i in range(nthreads)]
        tx = [[] for i in range(nthreads)]
        # Process objects
        pr = [[] for i in range(nthreads)]
        # Lists of the distinct files
        filelists = [[] for i in range(nthreads)]
        # unpack all the files into nthreads lists
        for ix, file in enumerate(filenames(files, wavdir, listflag)):
            filelists[ix % nthreads].append(file)
        # Launch each of the individual processes
        for i in range(nthreads):
            rx[i], tx[i] = multiprocessing.Pipe(False)
            pr[i] = multiprocessing.Process(target=make_ht_from_list, 
                                            args=(analyzer, filelists[i], ht, tx[i]))
            pr[i].start()
        # gather results when they all finish
        for i in range(nthreads):
            # thread passes back serialized hash table structure
            htx = rx[i].recv()
            print "ht",i,"has",len(htx.names),"files",sum(htx.counts),"hashes"
            if len(ht.counts) == 0:
                # Avoid merge into empty hash table, just keep the first one
                ht = htx
            else:
                # merge in all the new items, hash entries
                ht.merge(htx)
            # finish that thread...
            pr[i].join()

    elif multiproc:
        # add tracks with experimental pipeline
        # This version is no longer run.  It sets up separate processes for 
        # analyze (wavfile2hashes) and ht.store.  However, it seems like it 
        # gets blocked in the pipe that transfers between them, because it 
        # runs slower.  Anyway, the compute time is something like 80% in 
        # wavfile2hashes, so it would only effect a marginal speedup.
        import multiprocessing
        # Setup a separate process to generate hashes
        #q = multiprocessing.Queue()
        rx, q = multiprocessing.Pipe(False)
        p = multiprocessing.Process(target=hash_sender, 
                                    args=(filenames(files, wavdir, listflag), analyzer, q))
        p.start()
        running = True
        ix = 0
        tothashes = 0
        totdur = 0.0
        while running:
            #filename, hashes, dur = q.get()
            filename, hashes, dur = rx.recv()
            #print "main: got", filename
            if filename:
                print time.ctime(), "ingesting #", ix, ":", filename, "(", len(hashes), "hashes) ..."
                ht.store(filename, hashes)
                tothashes += len(hashes)
                totdur += dur
                ix += 1
            else:
                running = False
        p.join()

        print "Added", tothashes, "hashes", \
              "(%.1f" % (tothashes/totdur), "hashes/sec)"
        analyzer.soundfiletotaldur = totdur
        analyzer.soundfilecount = ix

    else:
        # Adding files - command was 'new' or 'add'
        tothashes = 0
        for ix, file in enumerate(filenames(files, wavdir, listflag)):
            print time.ctime(), "ingesting #", ix, ":", file, "..."
            dur, nhash = analyzer.ingest(ht, file)
            tothashes += nhash

        print "Added", tothashes, "hashes", \
              "(%.1f" % (tothashes/float(analyzer.soundfiletotaldur)), "hashes/sec)"

    elapsedtime = time.clock() - initticks
    totdur = analyzer.soundfiletotaldur
    if totdur > 0.:
        print "Processed %d files (%.1f s total dur) in %.1f s sec = %.3f x RT" \
            % (analyzer.soundfilecount, totdur, elapsedtime, (elapsedtime/totdur))

    if ht and ht.dirty:
        ht.save(dbasename, {"samplerate":samplerate})


# Run the main function if called from the command line
if __name__ == "__main__":
    import sys
    main(sys.argv)
