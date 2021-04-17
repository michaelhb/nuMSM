from multiprocessing import Pool

def func(x):
    print("starting {}".format(x))
    return (x*x, str(x))

if __name__ == '__main__':
    with Pool(5) as p:
        print(p.map(func, [1, 2, 3]))