import numpy as np
import random


a = np.array(["eka", 'toka', 'kolmas'])
print(list(a))
b = [1, 1, 1, 1, 2, 2]
print(a[b])
for i in range(20):
    print(2*random.random()-1)