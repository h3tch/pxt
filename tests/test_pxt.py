import numpy as np
import test_cpp
# import test_cu
import test_rs
# import gc
# import objgraph

a, b = np.random.rand(256), np.random.rand(256)
# gc.set_debug(gc.DEBUG_LEAK)
# gc.collect()
print()

# objgraph.show_growth()
for _ in range(3):
    c = test_cpp.np_add(a, b)
    # objgraph.show_growth()
print(c)

for _ in range(3):
    c = test_cpp.i_add(1, 2)
    # objgraph.show_growth()
print(c)

for _ in range(3):
    tup = test_cpp.return_tuple(a, b)
    # objgraph.show_growth()
print(tup)

# rs = np.empty_like(a, dtype=np.float32)
# test_cu.multiply(rs, a.astype(np.float32), b.astype(np.float32), block=(256, 1, 1), grid=(1, 1))
# print(np.max(rs - a * b))

rs = test_rs.i_add(4, 6)
print(rs)
