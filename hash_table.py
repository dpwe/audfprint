"""
hash_table.py

Python implementation of the very simple, fixed-array hash table
used for the audfprint fingerprinter.

2014-05-25 Dan Ellis dpwe@ee.columbia.edu
"""

import numpy as np
import random
import cPickle as pickle
import os, gzip
import scipy.io

# Current format version
HT_VERSION = 20140920
# Earliest acceptable version
HT_COMPAT_VERSION = 20140920

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
            self.maxtime = maxtime
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
        if type(name) is str:
            # lookup name or assign new
            if name not in self.names:
                self.names.append(name)
                self.hashesperid.append(0)
            id_ = self.names.index(name)
        else:
            # we were passed in a numerical id
            id_ = name
        # Now insert the hashes
        hashmask = (1 << self.hashbits) - 1
        #mxtime = self.maxtime
        timemask = self.maxtime - 1
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
        idval = id_ * self.maxtime
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
        return np.c_[vals / self.maxtime, vals % self.maxtime].astype(np.int32)

    def get_hits(self, hashes):
        """ Return np.array of [id, delta_time, hash, time] rows
            associated with each element in hashes array of [time, hash] rows
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
        print "Saved fprints for", len(self.names), "files", \
              "(", sum(self.counts), "hashes)", \
              "to", name

    def load(self, name):
        """ Read either pklz or mat-format hash table file """
        ext = os.path.splitext(name)[1]
        if ext == '.mat':
            params = self.load_matlab(name)
        else:
            params = self.load_pkl(name)
        print "Read fprints for", len(self.names), "files", \
              "(", sum(self.counts), "hashes)", \
              "from", name
        return params

    def load_pkl(self, name):
        """ Read hash table values from file <name>, return params """
        with gzip.open(name, 'rb') as f:
            temp = pickle.load(f)
        assert temp.ht_version >= HT_COMPAT_VERSION
        params = temp.params
        self.hashbits = temp.hashbits
        self.depth = temp.depth
        self.maxtime = temp.maxtime
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
        self.hashbits = int(np.log(mht['HT_params'][0][0][0][0][0]) /
                            np.log(2.0))
        self.depth = mht['HT_params'][0][0][1][0][0]
        self.maxtime = mht['HT_params'][0][0][2][0][0]
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
        self.hashesperid = np.r_[[0], self.hashesperid]
        # Otherwise unmodified database
        self.dirty = False
        return params

    def totalhashes(self):
        """ Return the total count of hashes stored in the table """
        return np.sum(self.counts)

    def merge(self, ht):
        """ Merge in the results from another hash table """
        # All the items go into our table, offset by our current size
        ncurrent = len(self.names)
        #size = len(self.counts)
        self.names += ht.names
        self.hashesperid += ht.hashesperid
        # All the table values need to be increased by the ncurrent
        idoffset = self.maxtime * ncurrent
        for hash_ in np.nonzero(ht.counts)[0]:
            if self.counts[hash_] + ht.counts[hash_] <= self.depth:
                self.table[hash_,
                           self.counts[hash_]:(self.counts[hash_]
                                               + ht.counts[hash_])] \
                    = ht.table[hash_, :ht.counts[hash_]] + idoffset
            else:
                # Need to subselect
                allvals = np.r_[self.table[hash_, :self.counts[hash_]],
                                ht.table[hash_, :ht.counts[hash_]]]
                rperm = np.random.permutation(range(len(allvals)))
                self.table[hash_,] = allvals[rperm[:self.depth]]
            self.counts[hash_] += ht.counts[hash_]
        self.dirty = True
