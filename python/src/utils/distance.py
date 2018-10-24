import numpy as np


class MDistance:

    def __init__(self, m):
        self.m = m

    def set_m(self, m):
        self.m = m

    def distance(self, instance_a, instance_b):
        # vector_a = np.array(instance_a[attributes], dtype='float16')
        # vector_b = np.array(instance_b[attributes], dtype='float16')
        difference = instance_a - instance_b
        distance = np.sum(difference ** self.m) ** (1 / self.m)
        return distance
