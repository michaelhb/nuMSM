from os import path
import sqlite3
from common import *

# Test if result is already in DB (lazy and terrible method) (true if record exists)
def check_row(conn, dM, imw):
    c = conn.cursor()
    c.execute("select * from bau_scan where ABS((dm - ?1)/?1) < 1e-5 AND ABS((imw - ?2)/?2) < 1e-5;",
              (dM, imw))
    count = len(c.fetchall())
    assert(count == 1 or count == 0)
    return count == 1

insert_qry = """INSERT INTO bau_scan(m, dm, imw, rew, delta, eta, bau) VALUES (?,?,?,?,?,?,?);"""
create_table_qry = """CREATE TABLE IF NOT EXISTS "bau_scan" """ \
                   """(m real, dm real, imw real, rew real, delta real, eta real, bau real);"""

def bau_grid_scan(name, solver_class, points_per_dim, atol_default):
    M = 0.7732
    delta = -2.199
    eta = -1.857
    Rew = 2.444

    dM_min = -14
    dM_max = -2
    dMs = [10**e for e in np.linspace(dM_min, dM_max, points_per_dim)]

    Imw_min = -6.
    Imw_max = 6.
    Imws = np.linspace(Imw_min, Imw_max, points_per_dim)

    output_dir = path.abspath(path.join(path.dirname(__file__), 'output/'))
    conn = sqlite3.connect(path.join(output_dir, "bau_{}.db".format(name)))
    c = conn.cursor()
    c.execute(create_table_qry)
    conn.commit()

    plot_no = 0
    plot_max = points_per_dim**2

    for ix, dM in enumerate(dMs):
        for Imw in Imws:
            mp = ModelParams(M, dM, Imw, Rew, delta, eta)
            print(mp)
            if check_row(conn, dM, Imw): # Skip record if already present
                print("Record already present, skipping")
            else:
                T0 = get_T0(mp)
                print("{}/{}".format(plot_no, plot_max))
                print("T0 = {}".format(T0))

                solver = solver_class(mp, T0, Tsph, 2, {'rtol' : 1e-6, 'atol' : atol_default})
                try:
                    solver.solve()
                    bau = (28. / 78.) * solver.get_final_asymmetry()
                    print("BAU: {}".format(bau))
                except Exception as e:
                    print(e)
                    bau = None

                atol_retry = 1e-20

                # Retry with smaller tolerance
                if (bau is None or not (1e-30 < bau < 1e-4)) and atol_retry > atol_default:
                    print("Trying again at atol 1e-20")
                    solver = solver_class(mp, T0, Tsph, 2,{'rtol' : 1e-6, 'atol' : 1e-20})
                    try:
                        solver.solve()
                        bau = (28. / 78.) * solver.get_final_asymmetry()
                        print("BAU: {}".format(bau))
                    except Exception as e:
                        print(e)
                        bau = None

                # If BAU still looks weird, leave it as a NULL
                if bau is None or not (1e-30 < bau < 1e-4):
                    print("BAU out of expected range, storing NULL")
                    bau = None

                # Insert record into DB
                record = (M, dM, Imw, Rew, delta, eta, bau)
                c.execute(insert_qry, record)
                conn.commit()
            plot_no += 1
    conn.close()




