import sqlite3
import hashlib
import sys
from os import path
# import fasteners
from common import *
from collections import namedtuple

"""
Similar to ScanDB, but designed for multiple worker pools running. User provides a precomputed list of samples
to be run. These are initialised as "empty" entries with bau=NONE, lock=FALSE. 
"""

# Extension of ModelParams, adding in solver attributes for scanworker_mp.py

Sample = namedtuple("Sample", ModelParams._fields + ("tag", "description", "solvername", "n_kc",
    "kc_min", "kc_max", "quadscheme", "heirarchy", "cutoff"))

class MPScanDB:

    def __init__(self, path_, fast_insert=False, save_solutions=False, save_densities=False):
        self.conn = self.get_connection(path_)

        # Create tables if not present
        if not self.point_table_exists(): self.create_point_table()

        if save_solutions and not self.solution_table_exists():
            self.create_solution_table()

        if save_densities and not self.density_table_exists():
            self.create_density_table()

        # Speedup, maybe?
        if fast_insert:
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

    def point_table_exists(self):
        c = self.conn.cursor()
        c.execute(''' SELECT count(name) FROM sqlite_master WHERE type='table' AND name='points' ''')
        return c.fetchone()[0] == 1

    def solution_table_exists(self):
        c = self.conn.cursor()
        c.execute(''' SELECT count(name) FROM sqlite_master WHERE type='table' AND name='solutions' ''')
        return c.fetchone()[0] == 1

    def density_table_exists(self):
        c = self.conn.cursor()
        c.execute(''' SELECT count(name) FROM sqlite_master WHERE type='table' AND name='densities' ''')

    def create_point_table(self):
        c = self.conn.cursor()
        c.execute('''CREATE TABLE points (
            lock bool, hash text, M real, dM real, Imw real, Rew real, delta real, eta real, 
            tag string, description string, solvername string, nkc integer, kcmin real, kcmax real, quadscheme string, 
            heirarchy integer, cutoff real, time real, bau real)''')
        self.conn.commit()

    def create_solution_table(self):
        c = self.conn.cursor()
        c.execute('''CREATE TABLE solutions (hash text, temp real, bau real)''')
        self.conn.commit()

    def create_density_table(self):
        c = self.conn.cursor()

        # Create underlying table
        c.execute('''CREATE TABLE densities (hash text, temp double, kc real, entropy double, rp11 double, rp22 double, rpreal double, rpimag double, 
            rm11 double, rm22 double, rmreal double, rmimag real)''')

        # Create view in HNL basis
        c.execute('''
            CREATE VIEW IF NOT EXISTS densities_hnl AS
            SELECT
                temp, kc,
                (entropy/POW(temp,3))*(rp11 + 0.5*rm11) + 1.0/(EXP((SQRT(POW(M,2) + POW(temp*kc,2))/temp)) + 1.0) AS dn1,
                (entropy/POW(temp,3))*(rp11 - 0.5*rm11) + 1.0/(EXP((SQRT(POW(M,2) + POW(temp*kc,2))/temp)) + 1.0) AS dn1bar,
                (entropy/POW(temp,3))*(rp22 + 0.5*rm22) + 1.0/(EXP((SQRT(POW(M,2) + POW(temp*kc,2))/temp)) + 1.0) AS dn2,
                (entropy/POW(temp,3))*(rp22 - 0.5*rm22) + 1.0/(EXP((SQRT(POW(M,2) + POW(temp*kc,2))/temp)) + 1.0) AS dn2bar,
                1.0/(EXP((SQRT(POW(M,2) + POW(temp*kc,2))/temp)) + 1.0) AS fn
            FROM
                densities INNER JOIN points ON densities.hash = points.hash
            ORDER BY temp DESC, kc ASC;''')

        self.conn.commit()

    def get_hash(self, sample):

        # Special handling for null cutoff
        if sample.cutoff is None:
            cutoff = -1
        else:
            cutoff = sample.cutoff

        hash_str = "M: {:.5f}, dM: {:.5e}, Imw: {:.5f}, Rew: {:.5f}, delta: {:.5f}, eta: {:.5f}, tag: {}, " \
                   "hierarchy: {}, solver_class: {}, n_kc: {}, kc_min: {:.5f}, kc_max: {:.5f}, quadscheme: {}, cutoff: {:.5e}".format(
            sample.M, sample.dM, sample.Imw, sample.Rew, sample.delta, sample.eta, sample.tag,
            sample.heirarchy, sample.solvername, sample.n_kc, sample.kc_min, sample.kc_max, sample.quadscheme, cutoff
        )
        hash = hashlib.sha256(hash_str.encode()).hexdigest()
        return hash

    def hash_exists(self, sample):
        hash = self.get_hash(sample)
        c = self.conn.cursor()
        c.execute('''SELECT * FROM points WHERE hash = ?''', (hash,))
        res = c.fetchall()

        if len(res) == 0:
            return False
        else:
            return True

    def get_bau(self, sample):
        c = self.conn.cursor()
        hash = self.get_hash(sample)

        # Return record if it exists
        c.execute('''SELECT bau, time FROM points WHERE hash = ? ''', (hash,))
        res = c.fetchall()

        if len(res) == 0:
            return (None, None)
        else:
            return res[0]

    def add_sample(self, sample):
        """
        Add an (unprocessed) sample to the DB.
        """
        hash = self.get_hash(sample)

        if not self.hash_exists(sample):
            # Special handling for null cutoff
            if sample.cutoff is None:
                cutoff = -1
            else:
                cutoff = sample.cutoff

            c = self.conn.cursor()
            c.execute('''INSERT INTO points VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''', (
                False, hash, sample.M, sample.dM, sample.Imw, sample.Rew, sample.delta, sample.eta,
                sample.tag, sample.description, sample.solvername, sample.n_kc, sample.kc_min, sample.kc_max,
                sample.quadscheme, sample.heirarchy, cutoff,
                None, None
            ))
            self.conn.commit()
        else:
            print("Skipped hash {}".format(hash))

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
                            tag, description, solvername, nkc, kcmin, kcmax, quadscheme, heirarchy, cutoff 
                            FROM points WHERE lock = FALSE LIMIT 1''')
        else:
            c.execute('''SELECT 
                            hash, M, dM, Imw, Rew, delta, eta, 
                            tag, description, solvername, nkc, kcmin, kcmax, quadscheme, heirarchy, cutoff 
                            FROM points WHERE lock = FALSE and tag = ? LIMIT 1''', (tag,))

        res = c.fetchall()

        if len(res) > 1:
            raise Exception("Hash collision!")

        sample = None

        # Lock the fetched records and build ModelParams
        for r in res:
            hash_, M_, dM_, Imw_, Rew_, delta_, eta_, tag_, description_, \
                solvername_, n_kc_, kc_min_, kc_max_, quadscheme_, heirarchy_, cutoff_ = r

            # Special handling for null cutoff
            if cutoff_ == -1:
                cutoff_ = None

            c.execute('''UPDATE points SET lock = TRUE WHERE hash = ?;''', (hash_,))
            sample = Sample(M=M_, dM=dM_, Imw=Imw_, Rew=Rew_, delta=delta_, eta=eta_,
                              tag=tag_, description=description_, solvername=solvername_,
                              n_kc=n_kc_, kc_min=kc_min_, kc_max=kc_max_, quadscheme=quadscheme_,
                              cutoff=cutoff_, heirarchy=heirarchy_)

        self.conn.commit()

        return sample

    def save_result(self, sample, bau, time):
        """
        Uses the hash function on mp to update the bau and time for the relevant record.
        """
        hash = self.get_hash(sample)
        c = self.conn.cursor()

        c.execute('''UPDATE points SET bau = ?, time = ? WHERE hash = ?;''', (bau, time, hash))
        self.conn.commit()

    def save_solution(self, sample, solution):
        """
        solution should be a list of (T, bau) pairs
        """
        hash = self.get_hash(sample)
        c = self.conn.cursor()

        for T, bau in solution:
            c.execute('''INSERT INTO solutions VALUES (?,?,?)''', (hash, T, bau))

        self.conn.commit()

    def save_densities(self, sample, densities):
        """
        densities should be a list of (T, kc, entropy, <8 real components>) tuples
        """
        hash = self.get_hash(sample)
        c = self.conn.cursor()

        for d in densities:
            c.execute('''INSERT INTO densities VALUES (?,?,?,?,?,?,?,?,?,?,?,?)''',(hash, *d))

        self.conn.commit()

    def get_solution(self, sample):
        hash = self.get_hash(sample)
        c = self.conn.cursor()

        c.execute('''SELECT T, bau FROM solutions WHERE hash = ? ORDER BY T''', (hash,))
        return c.fetchall()

    def purge_hung_samples(self):
        c = self.conn.cursor()
        c.execute('''UPDATE points SET lock = FALSE WHERE bau IS NULL;''')
        self.conn.commit()

    def close_conn(self):
        self.conn.commit()
        self.conn.close()