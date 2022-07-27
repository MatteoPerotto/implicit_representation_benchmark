import copy
from multiprocessing import Queue, Process
from multiprocessing import Barrier
import torch
from vispy import app, scene
from vispy.scene import ViewBox
from vispy.scene.visuals import Markers
import numpy as np


class Visualizer(Process):

    def __init__(self):
        super().__init__()
        self.name = 'Visualizer'
        self.setup_f = False
        self.barrier = Barrier(2)

        self.queue_in = Queue(1)

        self.color_f = True

    def run(self):
        self.build_gui()
        self.barrier.wait()
        app.run()

    def build_gui(self):

        # logger.debug('Started gui building')
        self._timer = app.Timer('auto', connect=self.on_timer, start=True)

        canvas = scene.SceneCanvas(keys='interactive')
        canvas.size = 1200, 600
        canvas.show()

        self.grid = canvas.central_widget.add_grid()
        canvas.events.key_press.connect(self.dispatch)

        vb = ViewBox()
        vb = self.grid.add_widget(vb, row=0, col=0, row_span=1, col_span=1)


        vb.camera = scene.TurntableCamera(elevation=0, azimuth=0, distance=1)
        vb.border_color = (0.5, 0.5, 0.5, 1)

        self.scatter = Markers(parent=vb.scene)

        # logger.debug('Gui built successfully')

    def dispatch(self, event):
        if event.text == 'c':
            self.color_f = not self.color_f

    def on_timer(self, _):

        if not self.queue_in.empty():
            data = self.queue_in.get()

            points, predictions, labels = data['points'], data['predictions'], data['labels']

            positive_idxs = torch.sigmoid(predictions.squeeze(-1)) > 0.7
            false_negative_idxs = labels & ~positive_idxs
            # idxs = labels == 1

            positive = points[positive_idxs].cpu().numpy()
            false_negative = points[false_negative_idxs].cpu().numpy()

            pc = np.concatenate([positive, false_negative], axis=0)

            if pc.shape[0] != 0:
                if self.color_f:
                    p_colors = torch.where(labels[positive_idxs],
                                         torch.tensor([[0, 1, 0]]).T.repeat(1, positive.shape[0]),
                                         torch.tensor([[1, 0, 0]]).T.repeat(1, positive.shape[0])).T.cpu().numpy()
                else:
                    p_colors = np.array([[0, 0, 1]]).repeat(positive.shape[0], 0)

                fn_colors = np.array([[1, 1, 0]]).repeat(false_negative.shape[0], 0)

                colors = np.concatenate([p_colors, fn_colors], axis=0)

                self.scatter.set_data(pc * np.array([1, -1, 1]), edge_color=colors,
                                      face_color=colors, size=5)

    def update(self, data):
        if not self.setup_f:
            self.setup_f = True
            self.start()
            self.barrier.wait()

        if self.queue_in.empty():
            self.queue_in.put(copy.deepcopy(data))

    def stop(self):
        app.quit()

    def on_draw(self, event):
        pass
