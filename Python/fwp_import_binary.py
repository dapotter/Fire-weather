import numpy as np
import pandas as pd

file = '/home/dp/Documents/FWP/NCL/ERC_OR_WA_binary.bin'
dt = np.dtype([('a', 'float32'), ('b', 'float32'), ('c', 'float32'), ('d', 'float32')])
data = np.fromfile(file, dtype=dt)
df = pd.DataFrame(data, columns=data.dtype.names)
print('df from binary:\n', df)
