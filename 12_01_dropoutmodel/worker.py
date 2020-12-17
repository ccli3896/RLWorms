import numpy as np
def createandsort (n):
    rand = np.random.RandomState(42) #Give a seed to reproduce results
    a = rand.rand(n) #Generate an array of size n
    return a.sort() #Sort the array

def worker(x=2):
    print(x*x)
    return x*x