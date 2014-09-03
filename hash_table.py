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

class HashTable:
    """
    Simple hash table for storing and retrieving fingerprint hashes.

    :usage:
       >>> ht = HashTable(size=2**10, depth=100)
       >>> ht.store('identifier', list_of_landmark_time_hash_pairs)
       >>> list_of_ids_tracks = ht.get_hits(hash)
    """
    # Current format version
    HT_VERSION = 20140525
    # Earliest acceptable version
    HT_COMPAT_VERSION = 20140525

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
            self.table = np.zeros( (size, depth), dtype=np.uint32 )
            # keep track of number of entries in each list
            self.counts = np.zeros( size, dtype=np.int32 )
            # map names to IDs
            self.names = []
            # track number of hashes stored per id
            self.hashesperid = []
            # Mark as unsaved
            self.dirty = True

    def store(self, name, timehashpairs):
        """ Store a list of hashes in the hash table associated with a particular name (or integer ID) and time. """
        if type(name) is str:
            # lookup name or assign new
            if name not in self.names:
                self.names.append(name)
                self.hashesperid.append(0)
            id = self.names.index(name)
        else:
            # we were passed in a numerical id
            id = name
        # Now insert the hashes
        hashmask = (1 << self.hashbits) - 1
        # Try sorting the pairs by hash value, for better locality in storing
        sortedpairs = sorted(timehashpairs, key=lambda x:x[1])
        for time, hash in sortedpairs:
            # Keep only the bottom part of the time value
            time %= self.maxtime
            # Keep only the bottom part of the hash value
            hash &= hashmask
            # Mixin with ID
            val = (id * self.maxtime + time) #.astype(np.uint32)
            if self.counts[hash] < self.depth:
                # insert new val in next empty slot
                slot = self.counts[hash]
            else:
                # Choose a point at random
                slot = random.randint(0, self.counts[hash])
            # Only store if random slot wasn't beyond end
            if slot < self.depth:
                self.table[hash, slot] = val
            # Update record of number of vals in this bucket
            self.counts[hash] += 1
        # Record how many hashes we (attempted to) save for this id
        self.hashesperid[id] += len(timehashpairs)
        # Mark as unsaved
        self.dirty = True

    def get_entry(self, hash):
        """ Return the list of (id, time) entries associate with the given hash"""
        return [ ( int(val / self.maxtime), int(val % self.maxtime) )
                 for val 
                   in self.table[hash, :min(self.depth, self.counts[hash])] ]

    def get_hits(self, hashes):
        """ Return a list of (id, delta_time, hash, time) tuples 
            associated with each element in hashes list of (time, hash) """
        return [ (id, rtime-time, hash, time) for time, hash in hashes
                                          for id, rtime in self.get_entry(hash)]

    def save(self, name, params=[]):
        """ Save hash table to file <name>, including optional addition params """
        self.params = params
        self.version = self.HT_VERSION
        with gzip.open(name, 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
        self.dirty = False
        print "Saved fprints for", len(self.names), "files", \
              "(", sum(self.counts), "hashes)", \
              "to", name

    def load(self, name):
        """ Read either pklz or mat-format hash table file """
        stem, ext = os.path.splitext(name)
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
        assert(temp.version >= self.HT_COMPAT_VERSION)
        params = temp.params
        self.hashbits = temp.hashbits
        self.depth = temp.depth
        self.maxtime = temp.maxtime
        self.table = temp.table
        self.counts = temp.counts
        self.names = temp.names
        self.hashesperid = temp.hashesperid
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
        assert(params['mat_version'] >= 0.9)
        self.hashbits = int(np.log(mht['HT_params'][0][0][0][0][0])/np.log(2.0))
        self.depth = mht['HT_params'][0][0][1][0][0]
        self.maxtime = mht['HT_params'][0][0][2][0][0]
        params['hoptime'] = mht['HT_params'][0][0][3][0][0]
        params['targetsr'] = mht['HT_params'][0][0][4][0][0]
        params['nojenkins'] = mht['HT_params'][0][0][5][0][0]
        # Python doesn't support the (pointless?) jenkins hashing
        assert(params['nojenkins'])
        self.table = mht['HashTable'].T
        self.counts = mht['HashTableCounts'][0]
        self.names = [str(val[0]) if len(val) > 0 else [] 
                      for val in mht['HashTableNames'][0]]
        self.hashesperid = mht['HashTableLengths'][0]
        # Matlab uses 1-origin for the IDs in the hashes, so rather than 
        # rewrite them all, we shift the corresponding decode tables 
        # down one cell
        self.names.insert(0,'')
        self.hashesperid = np.r_[[0], self.hashesperid]
        # Otherwise unmodified database
        self.dirty = False
        return params

    def totalhashes(self):
        """ Return the total count of hashes stored in the table """
        return np.sum(self.counts)
