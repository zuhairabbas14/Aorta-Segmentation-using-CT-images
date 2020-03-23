from numba import njit
import numpy as np

@njit
def pseudo_dbscan_1d(arr, eps, neighbours):
    
    #Count each element amount
    n_arr = arr - min(arr)
    range_ = max(n_arr) + 1
    count_arr = np.zeros((range_,), dtype=np.int32)
    for elem in n_arr:
        count_arr[elem] += 1 
    
    #Determine label for each color
    current_neighbours = count_arr[:eps].sum() - 1
    label_arr = np.empty((range_,), dtype=np.int32)
    current_label = 1
    count_arr = np.concatenate((np.zeros((eps+1,), dtype=np.int32), count_arr, np.zeros((eps,), dtype=np.int32)))
    prev = False
    for i in range(range_):
        current_neighbours += count_arr[i + 2*eps + 1] - count_arr[i]
        if current_neighbours >= neighbours:
            label_arr[i] = current_label
            prev = True
        else:
            label_arr[i] = 0
            current_label += prev
            prev = False
    
    res = np.empty_like(n_arr)
    for i, elem in enumerate(n_arr):
        res[i] = label_arr[elem]
        
    return res
