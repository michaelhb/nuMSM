import sqlite3
import hashlib

class ScanDB:

    def __init__(self, path):
        self.conn = self.get_connection(path)

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
        c.execute('''CREATE TABLE points (hash text, tag text, M real, dM real, Imw real, Rew real, delta real, eta real, time real, bau real)''')
        self.conn.commit()

    def get_hash(self, mp, tag):
        hash_str = "M: {:.5f}, dM: {:.5e}, Imw: {:.5f}, Rew: {:.5f}, delta: {:.5f}, eta: {:.5f}, tag: ".format(
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

    def save_bau(self, mp, tag, bau, time):
        c = self.conn.cursor()
        hash = self.get_hash(mp, tag)

        # Raise exception if record exists
        c.execute('''SELECT bau FROM points WHERE hash = ? ''', (hash,))
        res = c.fetchall()

        if len(res) != 0:
            raise Exception("Record already exists!")

        # Save the BAU
        c.execute('''INSERT INTO points VALUES (?,?,?,?,?,?,?,?,?,?)''', (
            hash, tag, mp.M, mp.dM, mp.Imw, mp.Rew, mp.delta, mp.eta, time, bau
        ))

        self.conn.commit()

    def close_conn(self):
        self.conn.close()


