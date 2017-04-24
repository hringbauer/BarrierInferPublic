import sys
import numpy as np


import numpy as np
from scipy.integrate import quad
import psutil
import gc

xx_max = np.linspace(1,1000,num=100)
print("Memory available: %f GiB"%(psutil.virtual_memory().available/1024.**3))
[[quad(lambda x:x**2,0,x) for x in xx_max] for _ in range(3000)]
print("Memory available after quad: %f GiB"%(psutil.virtual_memory().available/1024.**3))
gc.collect()
gc.collect()
gc.collect()
print("Memory available after garbage collecting: %f GiB"%(psutil.virtual_memory().available/1024.**3))