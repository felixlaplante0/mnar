import numpy as np


arr = np.arange(16).reshape(4, 4)

print(arr, "\n")

idxs = [1, 2]
arr[idxs] = 0
arr[:, idxs] = 0

print(arr, "\n")

arr[idxs, idxs] = 1

print(arr)
