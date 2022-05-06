import numpy as np

c = np.random.randint(10,size = (2,3,4))

print(c)

result = np.max(c,axis=0)

print(result)

idx = np.unravel_index(np.argmax(c),c.shape)
print(idx)