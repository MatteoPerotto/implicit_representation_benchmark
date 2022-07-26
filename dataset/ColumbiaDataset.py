import json
from pathlib import Path

import numpy as np
import torch as tc
import tqdm

from dataset.utils import sample_point_cloud_pc

try:
    from open3d.cpu.pybind.geometry import TriangleMesh, PointCloud
    from open3d.cpu.pybind.io import read_point_cloud
    from open3d.cpu.pybind.utility import Vector3dVector, Vector3iVector
    from open3d.cpu.pybind.visualization import draw_geometries
except ImportError:
    from open3d.cuda.pybind.geometry import TriangleMesh, PointCloud
    from open3d.cuda.pybind.io import read_point_cloud
    from open3d.cuda.pybind.utility import Vector3dVector, Vector3iVector
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler

patch_size = 40


class ColumbiaDataset(Dataset):
    def __init__(self, root, split, subset='train_models_train_views',
                 length: int=None,
                 pick: list=None):
        """
        Args:
        """
        self.subset = subset
        self.file_path = split
        self.root = Path(root)
        with (self.root / split).open('r') as stream:
            split_data = json.load(stream)

        self.data = np.array(split_data[subset])

        if pick is not None:
            self.data = self.data[pick]

        if length is None:
            self.length = len(self.data)
        else:
            self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        partial_path = self.data[idx][0]
        ground_truth_path = self.data[idx][1]

        vertices = np.load((self.root / ground_truth_path / 'vertices.npy').as_posix())
        triangles = np.load((self.root / ground_truth_path / 'triangles.npy').as_posix())

        p3, p2, p1 = Path(partial_path).parts[-1].split('_')[1:4]
        y1_pose = np.load(((self.root / partial_path).parent / f'_0_0_{p1}_model_pose.npy').as_posix())
        x2_pose = np.load(((self.root / partial_path).parent / f'_0_{p2}_0_model_pose.npy').as_posix())
        y3_pose = np.load(((self.root / partial_path).parent / f'_{p3}_0_0_model_pose.npy').as_posix())

        vertices = match_mesh_to_partial(np.array(vertices), [y1_pose, x2_pose, y3_pose])
        mesh = TriangleMesh(vertices=Vector3dVector(vertices), triangles=Vector3iVector(triangles))
        complete = np.array(mesh.sample_points_uniformly(8192 * 2).points)

        center = get_bbox_center(complete)
        diameter = get_diameter(complete - center)
        complete = (complete - center) / diameter

        return complete.astype(np.float32)



def get_bbox_center(pc):
    center = pc.min(0) + (pc.max(0) - pc.min(0)) / 2.0
    return center


def get_diameter(pc):
    diameter = pc.max(0) - pc.min(0)
    return np.max(diameter)


def correct_pose(pose):
    pre = Rotation.from_euler('yz', [90, 180], degrees=True).as_matrix()
    post = Rotation.from_euler('zyz', [180, 180, 90], degrees=True).as_matrix()

    t = np.eye(4)
    t[:3, :3] = np.dot(post, np.dot(pose[:3, :3].T, pre))

    return t


def get_x_rot(pose):
    t = correct_pose(pose)

    x, y, z = Rotation.from_matrix(t[:3, :3].T).as_euler('xyz', degrees=True)

    rot_x = np.eye(4)
    rot_x[:3, :3] = Rotation.from_euler('x', x, degrees=True).as_matrix() @ Rotation.from_euler('x', 180,
                                                                                                degrees=True).as_matrix()

    return rot_x


def get_y_rot(pose):
    t = correct_pose(pose)

    x, y, z = Rotation.from_matrix(t[:3, :3].T).as_euler('xyz', degrees=True)
    if z < 0:
        t[:3, :3] = Rotation.from_euler('zxy', [-z, -x, y], degrees=True).as_matrix()
    else:
        t[:3, :3] = Rotation.from_euler('zxy', [-z, x, -y], degrees=True).as_matrix()

    theta_y = Rotation.from_matrix(t[:3, :3]).as_euler('zxy', degrees=True)[2]
    rot_y = np.eye(4)
    rot_y[:3, :3] = Rotation.from_euler('y', theta_y, degrees=True).as_matrix()

    return rot_y


def match_mesh_to_partial(vertices, pose):
    y1_pose, x2_pose, y3_pose = pose

    y_rot = get_y_rot(y1_pose)
    x2_rot = get_x_rot(x2_pose)
    y3_rot = get_y_rot(y3_pose)

    base_rot = np.eye(4)
    base_rot[:3, :3] = Rotation.from_euler('xyz', [180, 0, -90], degrees=True).as_matrix()

    t = y_rot @ x2_rot @ y3_rot @ base_rot

    pc = np.ones((np.size(vertices, 0), 4))
    pc[:, 0:3] = vertices

    pc = pc.T
    pc = t @ pc
    pc = pc.T[..., :3]

    return pc



if __name__ == '__main__':
    root = 'data/MCD'
    split = 'data/MCD/build_datasets/train_test_dataset.json'

    training_set = GraspingDataset(root, split, subset='train_models_train_views')
    for data in tqdm.tqdm(training_set):
        x, y = data

        a = PointCloud(points=Vector3dVector(x))
        a.paint_uniform_color([1, 0, 0])

        b = PointCloud(points=Vector3dVector(y))
        b.paint_uniform_color([0, 1, 0])

        draw_geometries([a, b, create_cube()])