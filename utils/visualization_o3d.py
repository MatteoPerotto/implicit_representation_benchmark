import numpy as np
import open3d as o3d
from open3d import visualization
import cv2


class GeometryViewer:

    def __init__(self, every=1):
        self.viewer = None
        self.main_pc = None
        self.setup_done = False

        self.every = every
        self.it = -1

        self.color_flag = True

    def setup(self, pc):
        points = pc[..., :3]
        main_pc = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(points))

        if pc.shape[1] == 6:
            colors = pc[..., 3:]
            main_pc.colors = o3d.utility.Vector3dVector(colors)

        viewer = visualization.Visualizer()
        viewer.create_window()

        viewer.add_geometry(main_pc)

        self.viewer = viewer
        self.main_pc = main_pc
        self.setup_done = True

    def update(self, pc: np.ndarray):
        self.read_command()

        self.it += 1
        if not (self.it % self.every == 0):
            return

        if not self.setup_done:
            self.setup(pc)

        if self.color_flag:
            pc = pc[..., :3]

        points = pc[..., :3]
        self.main_pc.points = o3d.utility.Vector3dVector(points)

        if pc.shape[1] == 6:
            colors = pc[..., 3:]
            self.main_pc.colors = o3d.utility.Vector3dVector(colors)

        self.viewer.update_geometry(self.main_pc)
        self.viewer.poll_events()
        self.viewer.update_renderer()

    def read_command(self):
        if cv2.waitKey(1) == ord('c'):
            print('switched_color')
            self.color_flag = not self.color_flag

if __name__ == '__main__':
    v = GeometryViewer()
    for _ in range(10000):
        pc = np.random.randn(1024, 3)
        v.update(pc)