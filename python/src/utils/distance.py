import numpy as np


class MDistance:

    def __init__(self, m):
        self._m = m

    def m(self):
        return self._m

    def set_m(self, m):
        self._m = m

    def distance(self, instance_a, instance_b):
        difference = instance_a - instance_b
        distance = np.sum(difference ** self._m) ** (1 / self._m)
        return distance
