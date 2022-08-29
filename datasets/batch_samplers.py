import numpy as np

class MyBatchSampler:

    def __init__(self, cumulative_sizes, batch_size, percentage):
        self.cumulative_sizes = cumulative_sizes
        self.batch_size_a = int(batch_size * percentage)
        self.batch_size_b = batch_size - self.batch_size_a

    def __len__(self):
        return self.cumulative_sizes[0] // self.batch_size_a

    def __iter__(self):
        indexes_a = list(np.random.permutation(range(0, self.cumulative_sizes[0])))
        indexes_b = list(np.random.permutation(range(self.cumulative_sizes[0], self.cumulative_sizes[1])))
        
        i = 0
        j = 0
        while i < len(indexes_a):
            if i + self.batch_size_a >= len(indexes_a):
                break
            if j + self.batch_size_b >= len(indexes_b):
                j = 0
            
            indexes = indexes_a[i : i + self.batch_size_a] + indexes_b[j : j + self.batch_size_b]
            yield indexes

            i += self.batch_size_a
            j += self.batch_size_b

