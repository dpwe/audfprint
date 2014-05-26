"""
hash_table.py

Python implementation of the very simple, fixed-array hash table
used for the audfprint fingerprinter.

2014-05-25 Dan Ellis dpwe@ee.columbia.edu
"""

import numpy as np
import random
import cPickle as pickle

class HashTable:
    """
    Simple hash table for storing and retrieving fingerprint hashes.

    :usage:
       >>> ht = HashTable(size=2**10, depth=100)
       >>> ht.store('identifier', list_of_landmark_time_hash_pairs)
       >>> list_of_ids_tracks = ht.get_hits(hash)
    """
    def __init__(self, size=1048576, depth=100, maxtime=16384, name=None):
        """ allocate an empty hash table of the specified size """
        if name is not None:
            self.load(name)
        else:
            self.size = size
            self.depth = depth
            self.maxtime = maxtime
            # allocate the big table
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
        for time, hash in timehashpairs:
            # Keep only the bottom part of the time value
            time %= self.maxtime
            # Mixin with ID
            val = (id * self.maxtime + time) #.astype(np.uint32)
            # increment count of vals in this hash bucket
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
        idtimelist = []
        for slot in xrange(self.counts[hash]):
            val = self.table[hash, slot]
            time = int(val % self.maxtime)
            id = int(val / self.maxtime)
            idtimelist.append( (id, time) )
        return idtimelist

    def get_hits(self, hashes):
        """ Return a list of (id, delta_time) pairs associated with each element in hashes list of (time, hash) """
        iddtimelist = []
        for time, hash in hashes:
            idtimelist = [(id, rtime-time) 
                          for id, rtime in self.get_entry(hash)]
            iddtimelist += idtimelist
        return iddtimelist

    def save(self, name, params=[]):
        """ Save hash table to file <name>, including optional addition params """
        self.params = params
        self.version = 20140525
        with open(name, 'w') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
        self.dirty = False
        print "saved hash table to ", name

    def load(self, name):
        """ Read hash table values from file <name>, return params """
        with open(name, 'r') as f:
            temp = pickle.load(f)
        params = temp.params
        self.size = temp.size
        self.depth = temp.depth
        self.maxtime = temp.maxtime
        self.table = temp.table
        self.counts = temp.counts
        self.names = temp.names
        self.hashesperid = temp.hashesperid
        self.dirty = False
        return params

    def totalhashes(self):
        """ Return the total count of hashesh stored in the table """
        return np.sum(self.counts)
