"""
hash_table.py

Python implementation of the very simple, fixed-array hash table
used for the audfprint fingerprinter.

2014-05-25 Dan Ellis dpwe@ee.columbia.edu
"""
from __future__ import print_function

import numpy as np
import random
import cPickle as pickle
import os, gzip
import scipy.io
import math

# Current format version
HT_VERSION = 20140920
# Earliest acceptable version
HT_COMPAT_VERSION = 20140920


def _bitsfor(maxval):
    """ Convert a maxval into a number of bits (left shift).
        Raises a ValueError if the maxval is not a power of 2. """
    maxvalbits = int(round(math.log(maxval)/math.log(2)))
    if maxval != (1 << maxvalbits):
        raise ValueError("maxval must be a power of 2, not %d" % maxval)
    return maxvalbits


class HashTable(object):
    """
    Simple hash table for storing and retrieving fingerprint hashes.

    :usage:
       >>> ht = HashTable(size=2**10, depth=100)
       >>> ht.store('identifier', list_of_landmark_time_hash_pairs)
       >>> list_of_ids_tracks = ht.get_hits(hash)
    """

    def __init__(self, filename=None, hashbits=20, depth=100, maxtime=16384):
        """ allocate an empty hash table of the specified size """
        if filename is not None:
            self.params = self.load(filename)
        else:
            self.hashbits = hashbits
            self.depth = depth
            self.maxtimebits = _bitsfor(maxtime)
            # allocate the big table
            size = 2**hashbits
            self.table = np.zeros((size, depth), dtype=np.uint32)
            # keep track of number of entries in each list
            self.counts = np.zeros(size, dtype=np.int32)
            # map names to IDs
            self.names = []
            # track number of hashes stored per id
            self.hashesperid = np.zeros(0, np.uint32)
            # Empty params
            self.params = {}
            # Record the current version
            self.ht_version = HT_VERSION
            # Mark as unsaved
            self.dirty = True

    def reset(self):
        """ Reset to empty state (but preserve parameters) """
        self.table[:,:] = 0
        self.counts[:] = 0
        self.names = []
        self.hashesperid.resize(0)
        self.dirty = True

    def store(self, name, timehashpairs):
        """ Store a list of hashes in the hash table
            associated with a particular name (or integer ID) and time.
        """
        id_ = self.name_to_id(name, add_if_missing=True)
        # Now insert the hashes
        hashmask = (1 << self.hashbits) - 1
        #mxtime = self.maxtime
        maxtime = 1 << self.maxtimebits
        timemask = maxtime - 1
        # Try sorting the pairs by hash value, for better locality in storing
        #sortedpairs = sorted(timehashpairs, key=lambda x:x[1])
        sortedpairs = timehashpairs
        # Tried making it an np array to permit vectorization, but slower...
        #sortedpairs = np.array(sorted(timehashpairs, key=lambda x:x[1]),
        #                       dtype=int)
        # Keep only the bottom part of the time value
        #sortedpairs[:,0] = sortedpairs[:,0] % self.maxtime
        # Keep only the bottom part of the hash value
        #sortedpairs[:,1] = sortedpairs[:,1] & hashmask
        idval = id_ << self.maxtimebits
        for time_, hash_ in sortedpairs:
            # How many already stored for this hash?
            count = self.counts[hash_]
            # Keep only the bottom part of the time value
            #time_ %= mxtime
            time_ &= timemask
            # Keep only the bottom part of the hash value
            hash_ &= hashmask
            # Mixin with ID
            val = (idval + time_) #.astype(np.uint32)
            if count < self.depth:
                # insert new val in next empty slot
                #slot = self.counts[hash_]
                self.table[hash_, count] = val
            else:
                # Choose a point at random
                slot = random.randint(0, count)
                # Only store if random slot wasn't beyond end
                if slot < self.depth:
                    self.table[hash_, slot] = val
            # Update record of number of vals in this bucket
            self.counts[hash_] = count + 1
        # Record how many hashes we (attempted to) save for this id
        self.hashesperid[id_] += len(timehashpairs)
        # Mark as unsaved
        self.dirty = True

    def get_entry(self, hash_):
        """ Return np.array of [id, time] entries
            associate with the given hash as rows.
        """
        vals = self.table[hash_, :min(self.depth, self.counts[hash_])]
        maxtimemask = (1 << self.matimebits) - 1
        ids = vals >> self.maxtimebits
        return np.c_[ids, vals & maxtimemask].astype(np.int32)

    def get_hits_orig(self, hashes):
        """ Return np.array of [id, delta_time, hash, time] rows
            associated with each element in hashes array of [time, hash] rows.
            This is the original version that actually calls get_entry().
        """
        # Allocate to largest possible number of hits
        hits = np.zeros((np.shape(hashes)[0]*self.depth, 4), np.int32)
        nhits = 0
        # Fill in
        for time_, hash_ in hashes:
            idstimes = self.get_entry(hash_)
            nids = np.shape(idstimes)[0]
            hitrows = nhits + np.arange(nids)
            hits[hitrows, 0] = idstimes[:, 0]
            hits[hitrows, 1] = idstimes[:, 1] - time_
            hits[hitrows, 2] = hash_
            hits[hitrows, 3] = time_
            nhits += nids
        # Discard the excess rows
        hits.resize( (nhits, 4) )
        return hits

    def get_hits(self, hashes):
        """ Return np.array of [id, delta_time, hash, time] rows
            associated with each element in hashes array of [time, hash] rows.
            This version has get_entry() inlined, it's about 30% faster.
        """
        # Allocate to largest possible number of hits
        nhashes = np.shape(hashes)[0]
        hits = np.zeros((nhashes*self.depth, 4), np.int32)
        nhits = 0
        maxtimemask = (1 << self.maxtimebits) - 1
        # Fill in
        for ix in xrange(nhashes):
            time_ = hashes[ix][0]
            hash_ = hashes[ix][1]
            nids = min(self.depth, self.counts[hash_])
            tabvals = self.table[hash_, :nids]
            hitrows = nhits + np.arange(nids)
            hits[hitrows, 0] = tabvals >> self.maxtimebits
            hits[hitrows, 1] = (tabvals & maxtimemask) - time_
            hits[hitrows, 2] = hash_
            hits[hitrows, 3] = time_
            nhits += nids
        # Discard the excess rows
        hits.resize( (nhits, 4) )
        return hits

    def save(self, name, params=None):
        """ Save hash table to file <name>,
            including optional addition params
        """
        # Merge in any provided params
        if params:
            for key in params:
                self.params[key] = params[key]
        with gzip.open(name, 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
        self.dirty = False
        nhashes = sum(self.counts)
        print("Saved fprints for", sum(n is not None for n in self.names), 
              "files (", nhashes, "hashes) to", name)
        # Report the proportion of dropped hashes (overfull table)
        dropped = nhashes - sum(np.minimum(self.depth, self.counts))
        print("Dropped hashes=", dropped, "(%.2f%%)" % (
            100.0*dropped/max(1, nhashes)))

    def load(self, name):
        """ Read either pklz or mat-format hash table file """
        ext = os.path.splitext(name)[1]
        if ext == '.mat':
            params = self.load_matlab(name)
        else:
            params = self.load_pkl(name)
        print("Read fprints for", sum(n is not None for n in self.names), 
              "files (", sum(self.counts), "hashes) from", name)
        return params

    def load_pkl(self, name):
        """ Read hash table values from file <name>, return params """
        with gzip.open(name, 'rb') as f:
            temp = pickle.load(f)
        assert temp.ht_version >= HT_COMPAT_VERSION
        params = temp.params
        self.hashbits = temp.hashbits
        self.depth = temp.depth
        if hasattr(temp, 'maxtimebits'):
            self.maxtimebits = temp.maxtimebits
        else:
            self.maxtimebits = _bitsfor(temp.maxtime)
        self.table = temp.table
        self.counts = temp.counts
        self.names = temp.names
        self.hashesperid = np.array(temp.hashesperid).astype(np.uint32)
        self.ht_version = temp.ht_version
        self.dirty = False
        return params

    def load_matlab(self, name):
        """ Read hash table from version saved by Matlab audfprint.
        :params:
          name : str
            filename of .mat matlab fp dbase file
        :returns:
          params : dict
            dictionary of parameters from the Matlab file including
              'mat_version' : float
                version read from Matlab file (must be >= 0.90)
              'hoptime' : float
                hoptime read from Matlab file (must be 0.02322)
              'targetsr' : float
                target sampling rate from Matlab file (must be 11025)
        """
        mht = scipy.io.loadmat(name)
        params = {}
        params['mat_version'] = mht['HT_params'][0][0][-1][0][0]
        assert params['mat_version'] >= 0.9
        self.hashbits = _bitsfor(mht['HT_params'][0][0][0][0][0])
        self.depth = mht['HT_params'][0][0][1][0][0]
        self.maxtimebits = _bitsfor(mht['HT_params'][0][0][2][0][0])
        params['hoptime'] = mht['HT_params'][0][0][3][0][0]
        params['targetsr'] = mht['HT_params'][0][0][4][0][0]
        params['nojenkins'] = mht['HT_params'][0][0][5][0][0]
        # Python doesn't support the (pointless?) jenkins hashing
        assert params['nojenkins']
        self.table = mht['HashTable'].T
        self.counts = mht['HashTableCounts'][0]
        self.names = [str(val[0]) if len(val) > 0 else []
                      for val in mht['HashTableNames'][0]]
        self.hashesperid = np.array(mht['HashTableLengths'][0]).astype(uint32)
        # Matlab uses 1-origin for the IDs in the hashes, so rather than
        # rewrite them all, we shift the corresponding decode tables
        # down one cell
        self.names.insert(0, '')
        self.hashesperid = np.append([0], self.hashesperid)
        # Otherwise unmodified database
        self.dirty = False
        return params

    def totalhashes(self):
        """ Return the total count of hashes stored in the table """
        return np.sum(self.counts)

    def merge(self, ht):
        """ Merge in the results from another hash table """
        # All the items go into our table, offset by our current size
        # Check compatibility
        assert self.maxtimebits == ht.maxtimebits
        ncurrent = len(self.names)
        #size = len(self.counts)
        self.names += ht.names
        self.hashesperid = np.append(self.hashesperid, ht.hashesperid)
        # All the table values need to be increased by the ncurrent
        idoffset = (1 << self.maxtimebits) * ncurrent
        for hash_ in np.nonzero(ht.counts)[0]:
            allvals = np.r_[self.table[hash_, :self.counts[hash_]],
                            ht.table[hash_, :ht.counts[hash_]] + idoffset]
            # ht.counts[hash_] may be more than the actual number of
            # hashes we obtained, if ht.counts[hash_] > ht.depth.
            # Subselect based on actual size.
            if len(allvals) > self.depth:
                # Our hash bin is filled: randomly subselect the
                # hashes, and update count to accurately track the
                # total number of hashes we've seen for this bin.
                somevals = np.random.permutation(allvals)[:self.depth]
                self.table[hash_, ] = somevals
                self.counts[hash_] += ht.counts[hash_]
            else:
                # Our bin isn't full.  Store all the hashes, and
                # accurately track how many values it contains.  This
                # may mean some of the hashes counted for full buckets
                # in ht are "forgotten" if ht.depth < self.depth.
                self.table[hash_, :len(allvals)] = allvals
                self.counts[hash_] = len(allvals)

        self.dirty = True

    def name_to_id(self, name, add_if_missing=False):
        """ Lookup name in the names list, or optionally add. """
        if type(name) is str:
            # lookup name or assign new
            if name not in self.names:
                if not add_if_missing:
                    raise ValueError("name " + name + " not found")
                # Use an empty slot in the list if one exists.
                try:
                    id_ = self.names.index(None)
                    self.names[id_] = name
                    self.hashesperid[id_] = 0
                except ValueError:
                    self.names.append(name)
                    self.hashesperid = np.append(self.hashesperid, [0])
            id_ = self.names.index(name)
        else:
            # we were passed in a numerical id
            id_ = name
        return id_

    def remove(self, name):
        """ Remove all data for named entity from the hash table. """
        id_ = self.name_to_id(name)
        # If we happen to be removing the first item (id_ == 0), this will 
        # match every empty entry in table.  This is very inefficient, but 
        # it still works, and it's just one ID.  We could have fixed it by 
        # making the IDs written in to table start an 1, but that would mess 
        # up backwards compatibility.
        id_in_table = (self.table >> self.maxtimebits) == id_
        hashes_removed = 0
        for hash_ in np.nonzero(np.max(id_in_table, axis=1))[0]:
            vals = self.table[hash_, :self.counts[hash_]]
            vals = [v for v, x in zip(vals, id_in_table[hash_]) 
                    if not x]
            self.table[hash_] = np.hstack([vals, 
                                           np.zeros(self.depth - len(vals))])
            # This will forget how many extra hashes we had dropped until now.
            self.counts[hash_] = len(vals)
            hashes_removed += np.sum(id_in_table[hash_])
        self.names[id_] = None
        self.hashesperid[id_] = 0
        self.dirty = True
        print("Removed", name, "(", hashes_removed, "hashes).")

    def retrieve(self, name):
        """Return a list of (time, hash) pairs by finding them in the table."""
        timehashpairs = []
        id_ = self.name_to_id(name)
        maxtimemask = (1 << self.maxtimebits) - 1
        # Still a bug for id_ 0.
        hashes_containing_id = np.nonzero(
            np.max((self.table >> self.maxtimebits) == id_, axis=1))[0]
        for hash_ in hashes_containing_id:
            entries = self.table[hash_, :self.counts[hash_]]
            matching_entries = np.nonzero(
                (entries >> self.maxtimebits) == id_)[0]
            times = (entries[matching_entries] & maxtimemask)
            timehashpairs.extend([(time, hash_) for time in times])
        return timehashpairs

    def list(self, print_fn=None):
        """ List all the known items. """
        if not print_fn:
            print_fn = print
        for name, count in zip(self.names, self.hashesperid):
            if name:
                print_fn(name + " (" + str(count) + " hashes)")
