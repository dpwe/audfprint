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

def match_hashes(ht, hashes):
    """ Match audio against fingerprint hash table.
        Return top N matches as (id, filteredmatches, timoffs, rawmatches)
    """
    # find the implicated id, time pairs from hash table
    hits = ht.get_hits(hashes)
    # Sorted list of all the track ids that got hits
    idlist = np.r_[-1, sorted([id for id, time in hits]), -1]
    # Counts of unique entries in the sorted list - diff of locations of changes
    counts = np.diff(np.nonzero(idlist[:-1] != idlist[1:]))[0]
    # ids corresponding to each count - just read after the changes in the list
    ids = idlist[np.cumsum(counts)]

    # Find all the actual hits for a the most popular ids
    bestcountsids = sorted(zip(counts, ids), reverse=True)
    # Try the top 100 results
    resultlist = []
    for rawcount, tid in bestcountsids[:100]:
        (mode, filtcount) = find_mode([time for (id, time) in hits 
                                       if id == tid], 
                                      window=1)
        resultlist.append( (tid, filtcount, mode, rawcount) )
    return sorted(resultlist, key=lambda x:x[1], reverse=True)

def match_file(ht, filename):
    """ Read in an audio file, calculate its landmarks, query against hash table.  Return top N matches as (id, filterdmatchcount, timeoffs, rawmatchcount)
    """
    d, sr = librosa.load(filename, sr=11025)
    # Collect landmarks offset by 0..3 quarter-windows
    t_win = 0.04644
    win = int(np.round(sr * t_win))
    qwin = win/4
    hq = audfprint.landmarks2hashes(
             audfprint.peaks2landmarks(audfprint.find_peaks(d, sr)))
    hq += audfprint.landmarks2hashes(
             audfprint.peaks2landmarks(audfprint.find_peaks(d[qwin:], sr)))
    hq += audfprint.landmarks2hashes(
             audfprint.peaks2landmarks(audfprint.find_peaks(d[2*qwin:], sr)))
    hq += audfprint.landmarks2hashes(
             audfprint.peaks2landmarks(audfprint.find_peaks(d[3*qwin:], sr)))
    #print "Analyzed",filename,"to",len(hq),"hashes"
    # Run query
    return match_hashes(ht, hq)


dotest = False
if dotest:
    pat = '/Users/dpwe/projects/shazam/Nine_Lives/*mp3'
    qry = 'query.mp3'
    ht = audfprint.glob2hashtable(pat)
    rslts = match_file(ht, qry)
    t_hop = 0.02322
    print "Matched", qry, "as", ht.names[rslts[0][0]], "at", t_hop*float(rslts[0][2]), "with", rslts[0][1], "of", rslts[0][3], "hashes"
