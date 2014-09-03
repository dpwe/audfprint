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

# Parameters
# DENSITY controls the density of landmarks found (approx DENSITY per sec)
DENSITY = 20.0
# OVERSAMP > 1 tries to generate extra landmarks by decaying faster
OVERSAMP = 1
## 512 pt FFT @ 11025 Hz, 50% hop
#t_win = 0.0464
#t_hop = 0.0232
# Just specify n_fft
# spectrogram enhancement
hpf_pole = 0.98
# how wide to spreak peaks
f_sd = 30.0
# Maximum number of local maxima to keep per frame
maxpksperframe = 5

# Values controlling peaks2landmarks
targetdf = 31   # +/- 50 bins in freq (LIMITED TO -32..31 IN LANDMARK2HASH)
mindt    = 2    # min time separation (traditionally 1, upped 2014-08-04)
targetdt = 63   # max lookahead in time (LIMITED TO <64 IN LANDMARK2HASH)
maxpairsperpeak=3  # Limit the number of pairs we'll make from each peak

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

def spreadpeaksinvector(vector, width=4.0):
    """ Create a blurred version of vector, where each of the local maxes 
        is spread by a gaussian with SD <width>.
    """
    npts = len(vector)
    peaks = locmax(vector, indices=True)
    return spreadpeaks(zip(peaks, vector[peaks]), npoints=npts, width=width)

# optimization: cache pre-calculated Gaussian profile
__sp_width = None
__sp_len = None
__sp_vals = []

#def init_spreadpeaks(npoints=None, width=4.0):
#    """ optimiziation - set global vals just once, must be called before spreadpeaks or spreadpeaksinvector """
#    global __sp_width, __sp_len, __sp_vals
#    if width != __sp_width or npoints != __sp_len:
#        # Need to calculate new vector
#        __sp_width = width
#        __sp_len = npoints
#        __sp_vals = np.exp(-0.5*((np.arange(-npoints, npoints+1)/float(width))**2))

def spreadpeaks(peaks, npoints=None, width=4.0, base=None):
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
    global __sp_width, __sp_len, __sp_vals
    # You can comment out this check if you use init_spreadpeaks
    if width != __sp_width or npoints != __sp_len:
        # Need to calculate new vector
        __sp_width = width
        __sp_len = npoints
        __sp_vals = np.exp(-0.5*((np.arange(-npoints, npoints+1)/float(width))**2))
    # Now the actual function
    for pos, val in peaks:
        Y = np.maximum(Y, val*__sp_vals[np.arange(npoints) + npoints - pos])

    return Y

def find_peaks(d, sr, density=None, n_fft=None, n_hop=None):
    """Find the local peaks in the spectrogram as basis for fingerprints.
       Returns a list of (time_frame, freq_bin) pairs.

     :params:
        d - np.array of float
          Input waveform as 1D vector

        sr - int
          Sampling rate of d

        density - int
          Target hashes per second

        n_fft - int
          Size of FFT used for STFT analysis in samples

        n_hop - int
          Sample advance between STFT frames

     :returns:
        pklist - list of (int, int)
          Ordered list of landmark peaks found in STFT.  First value of
          each pair is the time index (in STFT frames, i.e., units of 
          n_hop/sr secs), second is the FFT bin (in units of sr/n_fft 
          Hz).

    """
    # Args 
    if density is None:
        density = DENSITY
    if n_fft is None:
        n_fft = 512
    if n_hop is None:
        n_hop = n_fft/2
    # optimized setup
    #init_spreadpeaks(npoints=n_fft/2, width=f_sd)
    # Base spectrogram
    #n_fft = int(np.round(sr*t_win))
    #n_hop = int(np.round(sr*t_hop))
    #print "find_peaks: sr=",sr,"n_fft=",n_fft,"n_hop=",n_hop
    # masking envelope decay constant
    a_dec = (1.0 - 0.01*(density*np.sqrt(n_hop/352.8)/35.0))**(1.0/OVERSAMP)
    # Take spectrogram
    mywin = np.hanning(n_fft+2)[1:-1]
    S = np.abs(librosa.stft(d, n_fft=n_fft, hop_length=n_hop, window=mywin))
    S = np.log(np.maximum(S, np.max(S)/1e6))
    S = S - np.mean(S)
    # High-pass filter onset emphasis
    # [:-1,] discards top bin (nyquist) of sgram so bins fit in 8 bits
    S = np.array([scipy.signal.lfilter([1, -1], 
                                       [1, -(hpf_pole)**(1/OVERSAMP)], Srow) 
                  for Srow in S])[:-1,]
    # initial threshold envelope based on peaks in first 10 frames
    (srows, scols) = np.shape(S)
    sthresh = spreadpeaksinvector(np.max(S[:,:np.minimum(10, scols)],axis=1), 
                                  f_sd)
    ## Store sthresh at each column, for debug
    #thr = np.zeros((srows, scols))
    peaks = np.zeros((srows, scols))
    # optimization of mask update
    global __sp_vals
    __sp_pts = len(sthresh)
    #
    for col in range(scols):
        Scol = S[:, col]
        sdiff = np.maximum(0, Scol - sthresh)
        sdmaxposs = np.nonzero(locmax(sdiff))[0]
        npeaksfound = 0
        # Work down list of peaks in order of their prominence above threshold 
        # (compatible with Matlab)
        #pkvals = Scol[sdmaxposs] - sthresh[sdmaxposs]   # for MATCOMPAT v0.90
        # Work down list of peaks in order of their absolute value above 
        # threshold (sensible)
        pkvals = Scol[sdmaxposs]
        for val, peakpos in sorted(zip(pkvals, sdmaxposs), reverse=True):
            if npeaksfound < maxpksperframe:
                if Scol[peakpos] > sthresh[peakpos]:
                    #print "frame:", col, " bin:", peakpos, " val:", Scol[peakpos], " thr:", sthresh[peakpos]
                    npeaksfound += 1
                    # What we actually want
                    #sthresh = spreadpeaks([(peakpos, Scol[peakpos])], 
                    #                      base=sthresh, width=f_sd)
                    # Optimization - inline the core function within spreadpeaks
                    sthresh = np.maximum(sthresh, Scol[peakpos]*__sp_vals[(__sp_pts - peakpos):(2*__sp_pts - peakpos)])
                    #
                    peaks[peakpos, col] = 1
        #thr[:, col] = sthresh
        sthresh *= a_dec

    # Backwards filter to prune peaks
    sthresh = spreadpeaksinvector(S[:,-1], f_sd)
    for col in range(scols, 0, -1):
        pkposs = np.nonzero(peaks[:, col-1])[0]
        peakvals = S[pkposs, col-1]
        for val, peakpos in sorted(zip(peakvals, pkposs), reverse=True):
            if val >= sthresh[peakpos]:
                # Setup the threshold
                sthresh = spreadpeaks([(peakpos, val)], base=sthresh, 
                                      width=f_sd)
                # Delete any following peak (threshold should, but be sure)
                if col < scols:
                    peaks[peakpos, col] = 0
            else:
                # delete the peak
                peaks[peakpos, col-1] = 0
        sthresh = a_dec*sthresh

    # build a list of peaks
    pklist = [[] for _ in xrange(scols)]
    for col in xrange(scols):
        pklist[col] = np.nonzero(peaks[:,col])[0]
    return pklist

def peaks2landmarks(pklist):
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
            for col2 in xrange(col+mindt, min(scols, col+targetdt)):
                for peak2 in pklist[col2]:
                    if ( (pairsthispeak < maxpairsperpeak)
                         and abs(peak2-peak) < targetdf
                         and abs(peak2-peak) + abs(col2-col) > 2  ):
                        # We have a pair!
                        landmarks.append( (col, peak, peak2, col2-col) )
                        pairsthispeak += 1

    return landmarks

# Globals defining packing of landmarks into hashes
F1bits = 8
DFbits = 6
DTbits = 6

b1mask  = (1 << F1bits) - 1
b1shift = DFbits + DTbits
dfmask  = (1 << DFbits) - 1
dfshift = DTbits
dtmask  = (1 << DTbits) - 1

def landmarks2hashes(landmarks):
    """ Convert a list of (time, bin1, bin2, dtime) landmarks 
        into a list of (time, hash) pairs where the hash combines 
        the three remaining values.
    """
    # build up and return the list of hashed values
    return [ ( time, 
               ( ((bin1 & b1mask) << b1shift) 
                 | (((bin2 - bin1) & dfmask) << dfshift)
                 | (dtime & dtmask)) ) 
             for time, bin1, bin2, dtime in landmarks ]

def hashes2landmarks(hashes):
    """ Convert the mashed-up landmarks in hashes back into a list 
        of (time, bin1, bin2, dtime) tuples.
    """
    landmarks = []
    for time, hash in hashes:
        dtime = hash & dtmask
        bin1 = (hash >> b1shift) & b1mask
        dbin = (hash >> dfshift) & dfmask
        # Sign extend frequency difference
        if dbin > (1 << (DFbits-1)):
            dbin -= (1 << DFbits)
        landmarks.append( (time, bin1, bin1+dbin, dtime) )
    return landmarks

# Special extension indicating precomputed fingerprint
precompext = '.afpt'

# global stores duration of most recently-read soundfile
soundfiledur = 0.0

def wavfile2hashes(filename, sr=None, density=None, n_fft=None, n_hop=None, shifts=1):
    """ Read a soundfile and return its fingerprint hashes as a 
        list of (time, hash) pairs.  If specified, resample to sr first. 
        shifts > 1 causes hashes to be extracted from multiple shifts of 
        waveform, to reduce frame effects.  """
    root, ext = os.path.splitext(filename)
    if ext == precompext:
        # short-circuit - precomputed fingerprint file
        hashes = hashes_load(filename)
    else:
        [d, sr] = librosa.load(filename, sr=sr)
        # Store duration in a global because it's hard to handle
        global soundfiledur
        soundfiledur = float(len(d))/sr
        # Calculate hashes with optional part-frame shifts
        hq = []
        for shift in range(shifts):
            if n_hop == None:
                n_hop = 256
            shiftsamps = int(float(shift)/shifts*n_hop)
            hq += landmarks2hashes(peaks2landmarks(
                                         find_peaks(d[shiftsamps:], sr, 
                                                    density=density, 
                                                    n_fft=n_fft, 
                                                    n_hop=n_hop)))
        # remove duplicate elements by pushing through a set
        hashes = sorted(list(set(hq)))

    #print "wavfile2hashes: read", len(hashes), "hashes from", filename
    return hashes


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
    return wavfile2hashes(track_obj.fn_audio, sr=sr, density=density, 
                          n_fft=n_fft, n_hop=n_hop)

########### functions to link to actual hash table index database #######

import hash_table

class myTrackObj:
    """ a local alternative to Gordon's track_obj """
    # the name of the audio file
    fn_audio = None  

def ingest(ht, filename, density=None, n_hop=None, n_fft=None, sr=None, shifts=1):
    """ Read an audio file and add it to the database
    :params:
      ht : HashTable object
        the hash table to add to
      filename : str
        name of the soundfile to add
      density : float (default: None)
        the target density of landmarks per sec
      n_hop : int (default: None)
        hop in samples between frames
      n_fft : int (default: None)
        size of each FFT
      sr : int (default: None)
        resample input files to this sampling rate
      shifts : int (default 1)
        how many sub-frame shifts to apply to waveform
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
    hashes = wavfile2hashes(filename, density=density, sr=sr, 
                            n_fft=n_fft, n_hop=n_hop, shifts=shifts)
    ht.store(filename, hashes)
    #return (len(d)/float(sr), len(hashes))
    #return (np.max(hashes, axis=0)[0]*n_hop/float(sr), len(hashes))
    # soundfiledur is set up in wavfile2hashes, use result here
    return soundfiledur, len(hashes)

# Handy function to build a new hash table from a file glob pattern
import glob, time
def glob2hashtable(pattern, density=None):
    """ Build a hash table from the files matching a glob pattern """
    ht = hash_table.HashTable()
    filelist = glob.glob(pattern)
    initticks = time.clock()
    totdur = 0.0
    tothashes = 0
    for ix, file in enumerate(filelist):
        print time.ctime(), "ingesting #", ix, ":", file, "..."
        dur, nhash = ingest(ht, file, density)
        totdur += dur
        tothashes += nhash
    elapsedtime = time.clock() - initticks
    print "Added",tothashes,"(",tothashes/float(totdur),"hashes/sec) at ", elapsedtime/totdur, "x RT"
    return ht

test = False
if test:
    fn = '/Users/dpwe/Downloads/carol11k.wav'
    ht = hash_table.HashTable()
    ingest(ht, fn)
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

usage = """
Audio landmark-based fingerprinting.  
Create a new fingerprint dbase with new, 
append new files to an existing database with add, 
or identify noisy query excerpts with match.
"Precompute" writes a *.fpt file under fptdir with 
precomputed fingerprint for each input wav file.

Usage: audfprint (new | add | match | precompute) [-d <dbase> | --dbase <dbase>] [options] <file>...

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
  -l, --list                      Input files are lists, not audio
  -T, --sortbytime                Sort multiple hits per file by time (instead of score)
  -v, --verbose                   Verbose reporting
  -I, --illustrate                Make a plot showing the match
  -W <dir>, --wavdir <dir>        Find sound files under this dir [default: ]
  --version                       Report version number
  --help                          Print this message
"""

__version__ = 20140802

def main(argv):
    args = docopt.docopt(usage, version=__version__, argv=argv[1:]) 

    if args['new']:
        cmd = 'new'
    elif args['add']:
        cmd = 'add'
    elif args['precompute']:
        cmd = 'precompute'
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
    sortbytime = args['--sortbytime']
    files = args['<file>']
    precompdir = args['--precompdir']
    shifts = int(args['--shifts'])
    match_win = int(args['--match-win'])
    min_count = int(args['--min-count'])
    max_matches = int(args['--max-matches'])
    wavdir = args['--wavdir']
    # fixed - 512 pt FFT with 256 pt hop at 11025 Hz
    target_sr = samplerate # not always 11025, but always n_fft=512
    n_fft = 512
    n_hop = n_fft/2

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
        ht = {'params':[]}

    t_hop = n_hop/float(target_sr)

    # set default value for shifts depending on mode
    if shifts == 0:
        if cmd == 'match':
            # Default shifts is 4 for match
            shifts = 4
        else:
            # default shifts is 1 for new/add/precompute
            shifts = 1

    if cmd == 'match':
        # Running query
        for qry in filenames(files, wavdir, listflag):
            rslts, dur, nhash = audfprint_match.match_file(ht, qry, 
                                                           density=density,
                                                           sr=target_sr, 
                                                           n_fft=n_fft, 
                                                           n_hop=n_hop, 
                                                           window=match_win, 
                                                           shifts=shifts, 
                                                           threshcount=min_count, 
                                                           verbose=verbose)

            # filter results to keep only the ones with enough hits
            rslts = [(tophitid, nhashaligned, bestaligntime, nhashraw)
                     for tophitid, nhashaligned, bestaligntime, nhashraw
                         in rslts 
                         if nhashaligned >= min_count]

            if len(rslts) == 0:
                # No matches returned at all
                nhashaligned = 0
                print "NOMATCH", qry, ('%.3f'%dur), "sec", \
                      nhash, "raw hashes"
            else:
                if sortbytime:
                    rslts = sorted(rslts, key=lambda x:-x[2])
                for hitix in range(min(len(rslts), max_matches)):
                    # figure the number of raw and aligned matches for top hit
                    tophitid, nhashaligned, bestaligntime, nhashraw = \
                        rslts[hitix]
                    print "Matched", qry, ('%.3f'%dur), "sec", \
                        nhash, "raw hashes", \
                        "as", ht.names[tophitid], \
                        "at %.3f" % (bestaligntime*t_hop), "s", \
                        "with", nhashaligned, "of", nhashraw, "hashes"
                    if illustrate:
                        audfprint_match.illustrate_match(ht, qry, 
                                                         density=density,
                                                         sr=target_sr, 
                                                         n_fft=n_fft, 
                                                         n_hop=n_hop, 
                                                         window=match_win, 
                                                         shifts=shifts, 
                                                         sortbytime=sortbytime)

    elif cmd == 'precompute':
        # just precompute fingerprints
        for file in filenames(files, wavdir, listflag):
            hashes = wavfile2hashes(file, density=density, sr=target_sr, 
                                    n_fft=n_fft, n_hop=n_hop, shifts=shifts)
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
            print "wrote", opfname, "( %d hashes, %.3f sec)" % (len(hashes), 
                                                               soundfiledur)

    else:
        # Adding files - command was 'new' or 'add'
        initticks = time.clock()
        totdur = 0
        tothashes = 0
        for ix, file in enumerate(filenames(files, wavdir, listflag)):
            print time.ctime(), "ingesting #", ix, ":", file, "..."
            dur, nhash = ingest(ht, file, density=density,
                                sr=target_sr, 
                                n_fft=n_fft, n_hop=n_hop, 
                                shifts=shifts)
            totdur += dur
            tothashes += nhash

        elapsedtime = time.clock() - initticks
        print "Added", tothashes, "hashes", \
              "(%.1f" % (tothashes/float(totdur)), "hashes/sec)", \
              "at %.3f" % (elapsedtime/totdur), "x RT"
        if ht.dirty:
            ht.save(dbasename, {"samplerate":samplerate})


# Run the main function if called from the command line
if __name__ == "__main__":
    import sys
    main(sys.argv)
