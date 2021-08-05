import sqlite3
import hashlib
from os import path
# import fasteners
from common import *
"""
Similar to ScanDB, but designed for multiple worker pools running. User provides a precomputed list of samples
to be run. These are initialised as "empty" entries with bau=NONE, lock=FALSE. 
"""

# Replace this with something in the scratch filesystem when deploying to NCI
lock_file = path.abspath(path.join(path.dirname(__file__), 'lock'))
# lock = fasteners.InterProcessLock(lock_file)

class MPScanDB:

    def __init__(self, path_):
        self.conn = self.get_connection(path_)

        # Create table if not present
        if not self.table_exists(): self.create_table()

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
        c.execute('''CREATE TABLE points (lock bool, hash text, M real, dM real, Imw real, Rew real, delta real, eta real, time real, bau real)''')
        self.conn.commit()

    def get_hash(self, mp):
        hash_str = "M: {:.5f}, dM: {:.5e}, Imw: {:.5f}, Rew: {:.5f}, delta: {:.5f}, eta: {:.5f}".format(
            mp.M, mp.dM, mp.Imw, mp.Rew, mp.delta, mp.eta
        )
        return hashlib.md5(hash_str.encode()).hexdigest()

    def get_bau(self, mp):
        c = self.conn.cursor()
        hash = self.get_hash(mp)

        # Return record if it exists
        c.execute('''SELECT bau, time FROM points WHERE hash = ? ''', (hash,))
        res = c.fetchall()

        if len(res) == 0:
            return (None, None)
        else:
            return res[0]

    def add_sample(self, mp):
        """
        Add an (unprocessed) sample to the DB.
        """
        hash = self.get_hash(mp)
        c = self.conn.cursor()
        c.execute('''INSERT INTO points VALUES (?,?,?,?,?,?,?,?,?,?)''', (
            False, hash, mp.M, mp.dM, mp.Imw, mp.Rew, mp.delta, mp.eta, None, None
        ))
        self.conn.commit()

    def get_and_lock(self, N):
        """
        Does the following:
        1. Finds N samples with lock=FALSE
        2. Sets lock=TRUE
        3. Returns a list of ModelParams objects for the client to process.

        If there are < N unprocessed samples, all availble samples will be returned.
        If there are 0 unprocessed samples, an empty list will be returned.
        """
        # Get <= N unlocked records
        c = self.conn.cursor()
        c.execute('''SELECT hash, M, dM, Imw, Rew, delta, eta FROM points WHERE lock = FALSE LIMIT ?''', (N,))
        res = c.fetchall()
        mps = []

        # Lock the fetched records and build ModelParams
        for r in res:
            hash_, M_, dM_, Imw_, Rew_, delta_, eta_ = r
            c.execute('''UPDATE points SET lock = TRUE WHERE hash = ?;''', (hash_,))
            mps.append(ModelParams(M=M_, dM=dM_, Imw=Imw_, Rew=Rew_, delta=delta_, eta=eta_))

        self.conn.commit()

        return mps

    def save_result(self, mp, bau, time):
        """
        Uses the hash function on mp to update the bau and time for the relevant record.
        """
        hash = self.get_hash(mp)
        c = self.conn.cursor()
        c.execute('''UPDATE points SET bau = ?, time = ? WHERE hash = ?;''', (bau, time, hash))

    def purge_hung_samples(self):
        c = self.conn.cursor()
        c.execute('''UPDATE points SET lock = FALSE WHERE bau IS NULL;''')
        self.conn.commit()

    def close_conn(self):
        self.conn.commit()
        self.conn.close()