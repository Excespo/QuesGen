import numpy as np

a1 = np.random.randn(5, 2, 1)
a2 = np.random.randn(5, 2, 1)

def to_tuple(*tensors):
    return tuple(tensor for tensor in tensors)

t = to_tuple(a1, a2)

print(f"a1={a1}\na2={a2}\nt={t}")
