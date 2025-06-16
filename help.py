import numpy as np
from collections import deque

n_step = 3

np_test = np.full(n_step, fill_value=None)

np_test[0] = 1
np_test[1] = 2
np_test[2] = 3
np_test[0] = 4

deq_test = deque(maxlen=n_step)
deq_test.append(1)
deq_test.append(2)
deq_test.append(3)
deq_test.append(4)


for transition in reversed(list(np_test)[:-1]):
    print(transition)