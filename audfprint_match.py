"""
audfprint_match.py

Fingerprint matching code for audfprint

2014-05-26 Dan Ellis dpwe@ee.columbia.edu
"""
import librosa
import audfprint
import numpy as np

def find_mode(data, window=0):
    """ Find the (mode, count) of a set of data, including a tolerance window +/- window if > 0 """
    vals = np.unique(data)
    counts = [len([x for x in data if abs(x-val) <= window]) for val in vals]
    bestix = np.argmax(counts)
    return (vals[bestix], counts[bestix])

def match_hashes(ht, hashes, hashesfor=None):
    """ Match audio against fingerprint hash table.
        Return top N matches as (id, filteredmatches, timoffs, rawmatches)
        If hashesfor specified, return the actual matching hashes for that 
        hit (0=top hit).
    """
    # find the implicated id, time pairs from hash table
    hits = ht.get_hits(hashes)
    # Sorted list of all the track ids that got hits
    idlist = np.r_[-1, sorted([id for id, time, hash, otime in hits]), -1]
    # Counts of unique entries in the sorted list - diff of locations of changes
    counts = np.diff(np.nonzero(idlist[:-1] != idlist[1:]))[0]
    # ids corresponding to each count - just read after the changes in the list
    ids = idlist[np.cumsum(counts)]

    # Find all the actual hits for a the most popular ids
    bestcountsids = sorted(zip(counts, ids), reverse=True)
    # Try the top 100 results
    resultlist = []
    window = 1
    for rawcount, tid in bestcountsids[:100]:
        (mode, filtcount) = find_mode([time for (id, time, hash, otime) in hits 
                                       if id == tid], 
                                      window=window)
        resultlist.append( (tid, filtcount, mode, rawcount) )
    results = sorted(resultlist, key=lambda x:x[1], reverse=True)

    if hashesfor is not None:
        # reconstruct & return the matching hashes from the top hit
        tid = resultlist[hashesfor][0]
        mode = resultlist[hashesfor][2]
        matchhashes = [((otime), hash) for (id, time, hash, otime) in hits
                       if id == tid and abs(time - mode) <= window]
        return results, matchhashes
    else:
        return results

# Special extension indicating precomputed fingerprint
import os
import audfprint

def match_file(ht, filename, density=None, sr=11025, n_fft=512, n_hop=256, verbose=False):
    """ Read in an audio file, calculate its landmarks, query against hash table.  Return top N matches as (id, filterdmatchcount, timeoffs, rawmatchcount), also length of input file in sec, and count of raw query hashes extracted
    """
    root, ext = os.path.splitext(filename)
    if ext == audfprint.precompext:
        # short-circuit - precomputed fingerprint file
        hq = audfprint.hashes_load(filename)
        if verbose:
            print "Read precomputed hashes from",filename,"to",len(hq),"hashes"
        durd = 0
    else:
        #print "match_file: sr=",sr
        d, sr = librosa.load(filename, sr=sr)
        durd = float(len(d))/sr
        # Collect landmarks offset by 0..3 quarter-windows
        #t_hop = 0.02322
        #win = int(np.round(sr * t_hop))
        #qwin = win/4
        qwin = n_hop/4
        hq = audfprint.landmarks2hashes(
                  audfprint.peaks2landmarks(
                      audfprint.find_peaks(d, sr, density=density, 
                                           n_fft=n_fft, n_hop=n_hop)))
        hq += audfprint.landmarks2hashes(
                  audfprint.peaks2landmarks(
                      audfprint.find_peaks(d[qwin:], sr, density=density, 
                                           n_fft=n_fft, n_hop=n_hop)))
        hq += audfprint.landmarks2hashes(
                 audfprint.peaks2landmarks(
                     audfprint.find_peaks(d[2*qwin:], sr, density=density, 
                                           n_fft=n_fft, n_hop=n_hop)))
        hq += audfprint.landmarks2hashes(
                 audfprint.peaks2landmarks(
                     audfprint.find_peaks(d[3*qwin:], sr, density=density, 
                                           n_fft=n_fft, n_hop=n_hop)))
        if verbose:
            print "Analyzed",filename,"of",('%.3f'%durd),"s to",len(hq),"hashes"
    # Run query
    return match_hashes(ht, hq), durd, len(hq)


dotest = False
if dotest:
    pat = '/Users/dpwe/projects/shazam/Nine_Lives/*mp3'
    qry = 'query.mp3'
    ht = audfprint.glob2hashtable(pat)
    rslts, dur, nhash = match_file(ht, qry)
    t_hop = 0.02322
    print "Matched", qry, "as", ht.names[rslts[0][0]], "at", t_hop*float(rslts[0][2]), "with", rslts[0][1], "of", rslts[0][3], "hashes"
