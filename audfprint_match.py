"""
audfprint_match.py

Fingerprint matching code for audfprint

2014-05-26 Dan Ellis dpwe@ee.columbia.edu
"""
import librosa
import numpy as np
import time
# for checking phys mem size
import resource
# for localtest and illustrate
import audfprint_analyze
import matplotlib.pyplot as plt

from scipy import stats

def log(message):
    """ log info with stats """
    print time.ctime(), \
        "physmem=", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, \
        "utime=", resource.getrusage(resource.RUSAGE_SELF).ru_utime, \
        message

def locmax(vec, indices=False):
    """ Return a boolean vector of which points in vec are local maxima.
        End points are peaks if larger than single neighbors.
        if indices=True, return the indices of the True values instead
        of the boolean vector. (originally from audfprint.py)
    """
    # x[-1]-1 means last value can be a peak
    #nbr = np.greater_equal(np.r_[x, x[-1]-1], np.r_[x[0], x])
    # the np.r_ was killing us, so try an optimization...
    nbr = np.zeros(len(vec)+1, dtype=bool)
    nbr[0] = True
    nbr[1:-1] = np.greater_equal(vec[1:], vec[:-1])
    maxmask = (nbr[:-1] & ~nbr[1:])
    if indices:
        return np.nonzero(maxmask)[0]
    else:
        return maxmask

def find_modes(data, threshold=5, window=0):
    """ Find multiple modes in data,  Report a list of (mode, count)
        pairs for every mode greater than or equal to threshold.
        Only local maxima in counts are returned.
    """
    # TODO: Ignores window at present
    datamin = np.amin(data)
    fullvector = np.bincount(data - datamin)
    # Find local maxima
    localmaxes = np.nonzero(np.logical_and(locmax(fullvector),
                                           np.greater_equal(fullvector,
                                                            threshold)))[0]
    return localmaxes + datamin, fullvector[localmaxes]

class Matcher(object):
    """Provide matching for audfprint fingerprint queries to hash table"""

    def __init__(self):
        """Set up default object values"""
        # Tolerance window for time differences
        self.window = 1
        # Absolute minimum number of matching hashes to count as a match
        self.threshcount = 5
        # How many hits to return?
        self.max_returns = 1
        # How deep to search in return list?
        self.search_depth = 100
        # Sort those returns by time (instead of counts)?
        self.sort_by_time = False
        # Verbose reporting?
        self.verbose = False
        # Do illustration?
        self.illustrate = False
        # Careful counts?
        self.exact_count = False

    def match_hashes(self, ht, hashes, hashesfor=None):
        """ Match audio against fingerprint hash table.
            Return top N matches as (id, filteredmatches, timoffs, rawmatches)
            If hashesfor specified, return the actual matching hashes for that
            hit (0=top hit).
        """
        # find the implicated id, time pairs from hash table
        #log("nhashes=%d" % np.shape(hashes)[0])
        hits = ht.get_hits(hashes)
        #log("nhits=%d" % np.shape(hits)[0])
        allids = hits[:, 0]
        alltimes = hits[:, 1]
        allhashes = hits[:, 2]
        allotimes = hits[:, 3]

        maxotime = np.max(allotimes)
        ids = np.unique(allids)
        #log("nids=%d" % np.size(ids))
        #rawcounts = np.sum(np.equal.outer(ids, allids), axis=1)
        # much faster, and doesn't explode memory
        rawcounts = np.bincount(allids)[ids]
        # Divide the raw counts by the total number of hashes stored
        # for the ref track, to downweight large numbers of chance
        # matches against longer reference tracks.
        wtdcounts = rawcounts/(ht.hashesperid[ids].astype(float))
        #log("max(rawcounts)=%d" % np.amax(rawcounts))

        # Find all the actual hits for a the most popular ids
        bestcountsixs = np.argsort(wtdcounts)[::-1]
        # We will examine however many hits have rawcounts above threshold
        # up to a maximum of search_depth.
        maxdepth = np.minimum(np.count_nonzero(np.greater(rawcounts,
                                                          self.threshcount)),
                              self.search_depth)
        results = []
        #log("len(rawcounts)=%d max(bestcountsixs)=%d" %
        #    (len(rawcounts), max(bestcountsixs)))
        if not self.exact_count:
            # Make sure alltimes has no negative vals (so np.bincount is happy)
            mintime = np.amin(alltimes)
            alltimes -= mintime
            for ix in bestcountsixs[:maxdepth]:
                #log("ix=%d"%ix)
                rawcount = rawcounts[ix]
                tid  = ids[ix]
                tidtimes = alltimes[allids==tid]
                if np.amax(np.bincount(tidtimes)) <= self.threshcount:
                    continue
                modes, counts = find_modes(tidtimes, self.threshcount)
                if len(modes):
                    mode = modes[np.nonzero(np.equal(counts,
                                                     np.amax(counts)))[0][0]]
                    count = np.count_nonzero(np.less_equal(
                        np.abs(tidtimes - mode), self.window))
                    #log("tid %d raw %d count %d" % (tid, rawcount, count))
                    results.append((tid, count, mode+mintime, rawcount))
        else:
            # Slower, old process for exact match counts
            for ix  in bestcountsixs[:maxdepth]:
                rawcount = rawcounts[ix]
                tid = ids[ix]
                modes, counts = find_modes(alltimes[np.nonzero(allids==tid)[0]],
                                           window=self.window,
                                           threshold=self.threshcount)
                for (mode, filtcount) in zip(modes, counts):
                    # matchhashes may include repeats because multiple
                    # ref hashes may match a single query hash under window.
                    # Uniqify:
                    #matchhashes = sorted(list(set(matchhashes)))
                    matchix = np.nonzero((allids == tid) &
                                         (np.abs(alltimes-mode) <=
                                          self.window))[0]
                    matchhasheshash = np.unique(allotimes[matchix]
                                                + maxotime*allhashes[matchix])
                    matchhashes = [(hash_ % maxotime, hash_ / maxotime)
                                   for hash_ in matchhasheshash]
                    # much, much faster
                    filtcount = len(matchhashes)
                    if filtcount >= self.threshcount:
                        if hashesfor is not None:
                            results.append((tid, filtcount, mode, rawcount, matchhashes))
                        else:
                            results.append((tid, filtcount, mode, rawcount))

        results = sorted(results, key=lambda x: x[1], reverse=True)
        if hashesfor is None:
            return results
        else:
            hashesforhashes = results[hashesfor][4]
            results = [(tid, filtcnt, mode, rawcount)
                        for (tid, filtcnt, mode,
                             rawcount, matchhashes) in results]
            return results, results[hashesfor][4]

    def match_file(self, analyzer, ht, filename, number=None):
        """ Read in an audio file, calculate its landmarks, query against
            hash table.  Return top N matches as (id, filterdmatchcount,
            timeoffs, rawmatchcount), also length of input file in sec,
            and count of raw query hashes extracted
        """
        q_hashes = analyzer.wavfile2hashes(filename)
        # Fake durations as largest hash time
        if len(q_hashes) == 0:
            durd = 0.0
        else:
            durd = float(analyzer.n_hop * q_hashes[-1][0])/analyzer.target_sr
        if self.verbose:
            if number is not None:
                numberstring = "#%d"%number
            else:
                numberstring = ""
            print time.ctime(), "Analyzed", numberstring, filename, "of", \
                  ('%.3f'%durd), "s " \
                  "to", len(q_hashes), "hashes"
        # Run query
        rslts = self.match_hashes(ht, q_hashes)
        # Post filtering
        if self.sort_by_time:
            rslts = sorted(rslts, key=lambda x: -x[2])
        return (rslts[:self.max_returns], durd, len(q_hashes))

    def file_match_to_msgs(self, analyzer, ht, qry):
        """ Perform a match on a single input file, return list
            of message strings """
        rslts, dur, nhash = self.match_file(analyzer, ht, qry)
        t_hop = analyzer.n_hop/float(analyzer.target_sr)
        if self.verbose:
            qrymsg = qry + (' %.3f '%dur) + "sec " + str(nhash) + " raw hashes"
        else:
            qrymsg = qry

        msgrslt = []
        if len(rslts) == 0:
            # No matches returned at all
            nhashaligned = 0
            if self.verbose:
                msgrslt.append("NOMATCH "+qrymsg)
            else:
                msgrslt.append(qrymsg+"\t")
        else:
            for (tophitid, nhashaligned, bestaligntime, nhashraw) in rslts:
                # figure the number of raw and aligned matches for top hit
                if self.verbose:
                    msgrslt.append("Matched " + qrymsg + " as "
                                   + ht.names[tophitid] \
                                   + (" at %.3f " % (bestaligntime*t_hop))
                                   + "s " \
                                   + "with " + str(nhashaligned) \
                                   + " of " + str(nhashraw) + " hashes")
                else:
                    msgrslt.append(qrymsg + "\t" + ht.names[tophitid])
                if self.illustrate:
                    self.illustrate_match(analyzer, ht, qry)
        return msgrslt

    def illustrate_match(self, analyzer, ht, filename):
        """ Show the query fingerprints and the matching ones
            plotted over a spectrogram """
        # Make the spectrogram
        d, sr = librosa.load(filename, sr=analyzer.target_sr)
        sgram = np.abs(librosa.stft(d, n_fft=analyzer.n_fft,
                                    hop_length=analyzer.n_hop,
                                    window=np.hanning(analyzer.n_fft+2)[1:-1]))
        sgram = 20.0*np.log10(np.maximum(sgram, np.max(sgram)/1e6))
        sgram = sgram - np.max(sgram)
        librosa.display.specshow(sgram, sr=sr,
                                 y_axis='linear', x_axis='time',
                                 cmap='gray_r', vmin=-80.0, vmax=0)
        # Do the match?
        q_hashes = analyzer.wavfile2hashes(filename)
        # Run query, get back the hashes for match zero
        results, matchhashes = self.match_hashes(ht, q_hashes, hashesfor=0)
        if self.sort_by_time:
            results = sorted(results, key=lambda x: -x[2])
        # Convert the hashes to landmarks
        lms = audfprint_analyze.hashes2landmarks(q_hashes)
        mlms = audfprint_analyze.hashes2landmarks(matchhashes)
        # Overplot on the spectrogram
        plt.plot(np.array([[x[0], x[0]+x[3]] for x in lms]).T,
                 np.array([[x[1], x[2]] for x in lms]).T,
                 '.-g')
        plt.plot(np.array([[x[0], x[0]+x[3]] for x in mlms]).T,
                 np.array([[x[1], x[2]] for x in mlms]).T,
                 '.-r')
        # Add title
        plt.title(filename + " : Matched as " + ht.names[results[0][0]]
                  + (" with %d of %d hashes" % (len(matchhashes),
                                                len(q_hashes))))
        # Display
        plt.show()
        # Return
        return results

def localtest():
    """Function to provide quick test"""
    pat = '/Users/dpwe/projects/shazam/Nine_Lives/*mp3'
    qry = 'query.mp3'
    hash_tab = audfprint_analyze.glob2hashtable(pat)
    matcher = Matcher()
    rslts, dur, nhash = matcher.match_file(audfprint_analyze.g2h_analyzer,
                                           hash_tab, qry)
    t_hop = 0.02322
    print "Matched", qry, "(", dur, "s,", nhash, "hashes)", \
          "as", hash_tab.names[rslts[0][0]], \
          "at", t_hop*float(rslts[0][2]), "with", rslts[0][1], \
          "of", rslts[0][3], "hashes"

# Run the main function if called from the command line
if __name__ == "__main__":
    localtest()
