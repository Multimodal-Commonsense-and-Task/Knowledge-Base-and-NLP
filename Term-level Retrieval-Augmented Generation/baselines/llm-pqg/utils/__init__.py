import ray
from ray.experimental.tqdm_ray import tqdm as tqdm_ray

def get_total_line(path : str) -> int:
    total_line = 0
    with open(path, 'r', encoding='utf-8') as fp:
        for line in fp:
            total_line += 1
    return total_line

@ray.remote
class ProgressActor:
    def __init__(self, total, desc=''):
        self.progress = tqdm_ray(total=total, desc=desc)

    def update(self):
        self.progress.update(1)

    def close(self):
        self.progress.close()