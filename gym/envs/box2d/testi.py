import numpy as np
import random

stuff = np.load('C:\\Users\\miikk\\Documents\\deep6trained.npy', allow_pickle=True)

for i in range(stuff.shape[0]):
    print(i, stuff[i].fitness)