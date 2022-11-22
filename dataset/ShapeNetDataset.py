from pathlib import Path

import numpy as np
import open3d
import torch.utils.data as data
import json
import os


class ShapeNetDataset(data.Dataset):
    def __init__(self, root, split, subset, length=None, pick=None):
        if not isinstance(root, Path):
            self.root = Path(root)

        self.splits_path = self.root / split
        self.subset = subset

        with self.splits_path.open() as f:
            self.splits = json.load(f)

        allowed_el = []
        for subdir, dirs, files in os.walk(self.root / self.subset / f'complete'):
            for file in files:
                allowed_el.append(file.split('.')[0])

        samples = []
        for category in self.splits:
            for el in category[subset]:
                if el in allowed_el:
                    samples += [f'{category["taxonomy_id"]}/{el}']
                
        self.samples = np.array(samples)
        
        if pick is not None:
            self.samples = self.samples[pick]

        if length is None:
            self.length = len(self.samples)
        else:
            self.length = length

    def __getitem__(self, idx):
        path = (self.root / self.subset / f'complete/{self.samples[idx]}.pcd')
        gt = np.array(open3d.io.read_point_cloud(path.as_posix()).points, dtype=np.float32)

        return gt

    def __len__(self):
        return self.length

if __name__ == '__main__':

    dataset = ShapeNetDataset(root='../pcr/data/PCN', split='PCN.json', subset='train')

    for d in dataset:
        print()
