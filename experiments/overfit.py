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
from dataset.utils import sample_point_cloud_pc, check_occupancy
from utils.reproducibility import make_reproducible

if Config.Logger.active:
    from clearml import Task
else:
    from utils.noop import NoOp as Task

from metrics.chamfer_dist import ChamferDistanceL1
from utils.scatter import pcs_to_plotly


def main():
    make_reproducible(Config.seed)

    task = Task.init(project_name=Config.Logger.project, task_name=Config.Logger.task)
    task.connect(Config.to_dict())
    task.connect_configuration(Config.to_dict())
    logger = task.get_logger()

    ckpt_dir = Path(f'checkpoints/{task.id}')
    if ckpt_dir.exists():
        print('Warning: checkpoint directory already exists!')
    else:
        ckpt_dir.mkdir()

    Model = Config.Model
    Data = Config.Data
    Train = Config.Train

    model = Model.architecture(**Config.Model.Params.to_dict())
    dataset = Data.DataSet.dataset(**Config.Data.DataSet.Params.to_dict())
    dataloader = DataLoader(dataset, **Config.Data.DataLoader.Params.to_dict())

    model.train()
    model.to(Train.device)
    model.init()  # the model is re-initialized before learning a new shape

    gt = next(iter(dataloader))
    gt = gt.to(Train.device)

    # take the complete point cloud and generate the training data:
    #  positive points sampled from the complete point cloud surface and negative points sampled around it

    optimizer = Train.Optim.optim(model.parameters(), **Train.Optim.Params.to_dict())

    metric = ChamferDistanceL1()
    best = np.inf

    range = trange(Train.epochs)
    for i in range:
        x, labels = sample_point_cloud_pc(gt, n_points=Data.input_dimension,
                                          dist=Data.split,
                                          noise_rate=Data.noise_rate,
                                          tolerance=Data.tolerance)

        predictions = model(x).squeeze(-1)
        loss = Train.loss_fn(predictions, labels.float())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Logging
        if i == 0:
            fig = pcs_to_plotly([x[labels].cpu().numpy(), x[~labels].cpu().numpy()],
                                colormaps=[[0, 255, 0], [255, 0, 0]],
                                names=['positive', 'negative'])
            logger.report_plotly("Point Clouds", "input points", fig)

        if (i + 1) % 1000 == 0:
            positive_idxs = torch.sigmoid(predictions) > 0.7
            if torch.all(~positive_idxs):
                positive_idxs[:, 0] = True

            chamfer = metric(x[positive_idxs].unsqueeze(0), x[labels].unsqueeze(0))

            logger.report_scalar('Loss', 'loss', loss.item(), i)
            logger.report_scalar('Chamfer', 'chamfer', chamfer.item() * 100, i)

        if (i + 1) % 5000 == 0:
            with torch.no_grad():
                ###################################
                ### Log classification of the input

                positive_idxs = torch.sigmoid(predictions) > 0.7

                tp_idxs = positive_idxs & labels
                fp_idxs = positive_idxs & ~labels
                fn_idxs = labels & ~positive_idxs

                if torch.all(~positive_idxs):
                    positive_idxs[:, 0] = True

                chamfer = metric(x[positive_idxs].unsqueeze(0), x[labels].unsqueeze(0))

                pcs = [pc.cpu().numpy() for pc in [x[fn_idxs], x[tp_idxs], x[fp_idxs]]]
                fig = pcs_to_plotly(pcs, colormaps=[[255, 255, 0], [0, 255, 0], [255, 0, 0]],
                                    names=['false negatives', 'true_positives', 'false_positives'])

                logger.report_plotly("Point Clouds", "precision_recall", fig, i)

                ###################################
                ### Log decoded point cloud
                final_pc, prob = model.to_pc(itr=20, thr=0.85, num_points=8192 * 2)
                final_pc = final_pc.unsqueeze(0)

                labels = check_occupancy(gt, final_pc, 0.001)  # points of the reconstruction close enough to gt
                recall_labels = ~check_occupancy(final_pc, gt, 0.001) # points of gt not close to reconstruction

                pcs = [pc.squeeze().cpu().numpy() for pc in [final_pc[labels], final_pc[~labels], gt[recall_labels]]]
                fig = pcs_to_plotly(pcs, colormaps=[[0, 255, 0], [255, 0, 0], [0, 0, 255]],
                                    names=['right', 'wrong', 'missed'], colors=[prob.cpu().numpy(), None])

                logger.report_plotly("Point Clouds", "reconstruction", fig)
                logger.report_scalar('Chamfer', 'real', metric(final_pc, gt) * 100, i)

            torch.save(model.state_dict(), ckpt_dir / 'latest.pth')
            if best < chamfer:
                torch.save(model.state_dict(), ckpt_dir / 'best.pth')
                best = chamfer

        range.set_postfix(loss=loss.item())



if __name__ == '__main__':
    main()
