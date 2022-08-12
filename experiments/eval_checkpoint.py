from pathlib import Path

# import matplotlib as mpl
# import matplotlib.pyplot as plt
# mpl.use('Qt5Agg')
# from mpl_toolkits.mplot3d import Axes3D
# import plotly.express as px
import numpy as np
import torch
from plotly.graph_objects import Scatter3d, Figure
from torch.utils.data import DataLoader
from tqdm import trange

from configs import Config
from dataset.utils import sample_point_cloud_pc

from metrics.chamfer_dist import ChamferDistanceL1
from utils.scatter import pcs_to_plotly


def main():
    """Just load a checkpoint and evaluate it on the
        first element of the dataloader.
    """

    Model = Config.Model
    Data = Config.Data
    Train = Config.Train

    model = Model.architecture(**Config.Model.Params.to_dict())
    dataset = Data.DataSet.dataset(**Config.Data.DataSet.Params.to_dict())
    dataloader = DataLoader(dataset, **Config.Data.DataLoader.Params.to_dict())

    model.train()
    model.to(Train.device)
    params = torch.load('checkpoints/39bf1b190734434ba4a943c8819fc637/latest.pth')
    model.load_state_dict(params)

    gt = next(iter(dataloader))
    gt = gt.to(Train.device)

    metric = ChamferDistanceL1()

    final_pc, prob = model.to_pc(itr=20, thr=0.7, num_points=8192*2)

    print(metric(final_pc.unsqueeze(0), gt) * 100)


if __name__ == '__main__':
    main()
