import time
import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange


def profile_load_time(data_gen, num_iters=50):
    it = iter(data_gen)
    load_times = []
    for i in trange(num_iters, desc="Iter"):
        t0 = time.time()
        next(it)
        t1 = time.time()
        load_times.append(t1-t0)
    load_times = np.array(load_times)
    plt.plot(list(range(num_iters)), load_times)
    print(f"total {load_times.sum():.2f}s, average {load_times.mean():.2f}s")
