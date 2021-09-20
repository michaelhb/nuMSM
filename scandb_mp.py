import sqlite3
import hashlib
from os import path
# import fasteners
from common import *
from collections import namedtuple

"""
Similar to ScanDB, but designed for multiple worker pools running. User provides a precomputed list of samples
to be run. These are initialised as "empty" entries with bau=NONE, lock=FALSE. 
"""

# Extension of ModelParams, adding in solver attributes for scanworker_mp.py

Sample = namedtuple("Sample", ModelParams._fields + ("tag", "description", "solvername", "n_kc", "kc_max", "heirarchy", "cutoff"))

class MPScanDB:

    def __init__(self, path_):
        self.conn = self.get_connection(path_)

        # Create table if not present
        if not self.table_exists(): self.create_table()

        # Speedup, maybe?
        c = self.conn.cursor()
        c.execute("PRAGMA journal_mode = WAL;")
        c.execute("PRAGMA synchronous = NORMAL;")
        self.conn.commit()

    def get_connection(self, path):
        conn = None
        try:
            conn = sqlite3.connect(path)
            print(sqlite3.version)
        except sqlite3.Error as e:
            print(e)
        finally:
            return conn

    def table_exists(self):
        c = self.conn.cursor()
        c.execute(''' SELECT count(name) FROM sqlite_master WHERE type='table' AND name='points' ''')
        return c.fetchone()[0] == 1

    def create_table(self):
        c = self.conn.cursor()
        c.execute('''CREATE TABLE points (
            lock bool, hash text, M real, dM real, Imw real, Rew real, delta real, eta real, 
            tag string, description string, solvername string, nkc integer, kcmax real, heirarchy integer, cutoff real,
            time real, bau real)''')
        self.conn.commit()

    def get_hash(self, mp, tag):
        hash_str = "M: {:.5f}, dM: {:.5e}, Imw: {:.5f}, Rew: {:.5f}, delta: {:.5f}, eta: {:.5f}, tag: {}".format(
            mp.M, mp.dM, mp.Imw, mp.Rew, mp.delta, mp.eta, tag
        )
        return hashlib.md5(hash_str.encode()).hexdigest()

    def get_bau(self, mp, tag):
        c = self.conn.cursor()
        hash = self.get_hash(mp, tag)

        # Return record if it exists
        c.execute('''SELECT bau, time FROM points WHERE hash = ? ''', (hash,))
        res = c.fetchall()

        if len(res) == 0:
            return (None, None)
        else:
            return res[0]

    def add_sample(self, mp, tag, description, solvername, nkc, kcmax, heirarchy, cutoff):
        """
        Add an (unprocessed) sample to the DB.
        """
        hash = self.get_hash(mp, tag)
        c = self.conn.cursor()
        c.execute('''INSERT INTO points VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''', (
            False, hash, mp.M, mp.dM, mp.Imw, mp.Rew, mp.delta, mp.eta,
            tag, description, solvername, nkc, kcmax, heirarchy, cutoff,
            None, None
        ))
        self.conn.commit()

    def get_and_lock(self, tag):
        """
        Does the following:
        1. Finds a sample with lock=FALSE
        2. Sets lock=TRUE
        3. Returns a Sample namedtuple for the client to process

        If there are 0 unprocessed samples, None will be returned.
        """
        # Get <= 1 unlocked records
        c = self.conn.cursor()

        if tag is None:
            c.execute('''SELECT 
                            hash, M, dM, Imw, Rew, delta, eta, 
                            tag, description, solvername, nkc, kcmax, heirarchy, cutoff 
                            FROM points WHERE lock = FALSE LIMIT 1''')
        else:
            c.execute('''SELECT 
                            hash, M, dM, Imw, Rew, delta, eta, 
                            tag, description, solvername, nkc, kcmax, heirarchy, cutoff 
                            FROM points WHERE lock = FALSE and tag = ? LIMIT 1''', (tag,))

        res = c.fetchall()
        sample = None

        # Lock the fetched records and build ModelParams
        for r in res:
            hash_, M_, dM_, Imw_, Rew_, delta_, eta_, tag_, description_, \
                solvername_, n_kc_, kc_max_, heirarchy_, cutoff_ = r
            c.execute('''UPDATE points SET lock = TRUE WHERE hash = ?;''', (hash_,))
            sample = Sample(M=M_, dM=dM_, Imw=Imw_, Rew=Rew_, delta=delta_, eta=eta_,
                              tag=tag_, description=description_, solvername=solvername_,
                              n_kc=n_kc_, kc_max=kc_max_, cutoff=cutoff_, heirarchy=heirarchy_)

        self.conn.commit()

        return sample

    def save_result(self, mp, tag, bau, time):
        """
        Uses the hash function on mp to update the bau and time for the relevant record.
        """
        hash = self.get_hash(mp, tag)
        c = self.conn.cursor()
        c.execute('''UPDATE points SET bau = ?, time = ? WHERE hash = ?;''', (bau, time, hash))
        self.conn.commit()

    def purge_hung_samples(self):
        c = self.conn.cursor()
        c.execute('''UPDATE points SET lock = FALSE WHERE bau IS NULL;''')
        self.conn.commit()

    def close_conn(self):
        self.conn.commit()
        self.conn.close()