from matplotlib import pyplot as plt
from open3d.cpu.pybind.geometry import PointCloud
from open3d.cpu.pybind.utility import Vector3dVector
from open3d.visualization import draw
from torch.utils.data import DataLoader
from tqdm import trange

from configs import Config
from dataset.utils import sample_point_cloud_pc
from utils.visualization_vispy import Visualizer

def main():
    """This code shows how the predictions are updated
        as the model learns how to reconstruct the shape
    """
    Model = Config.Model
    Data = Config.Data
    Train = Config.Train

    model = Model.architecture(**Config.Model.Params.to_dict())
    dataset = Data.DataSet.dataset(**Config.Data.DataSet.Params.to_dict())
    dataloader = DataLoader(dataset, **Config.Data.DataLoader.Params.to_dict())

    model.train()
    model.to(Train.device)

    viewer = Visualizer()
    history = []

    for gt in dataloader:
        gt = gt.to(Train.device)

        # take the complete point cloud and generate the training data:
        #  positive points sampled from the complete point cloud surface and negative points sampled around it
        x, labels = sample_point_cloud_pc(gt, n_points=Data.input_dimension,
                                          dist=Data.split,
                                          noise_rate=Data.noise_rate,
                                          tolerance=Data.tolerance)

        x, labels, gt = x.squeeze(0), labels.squeeze(0), gt.squeeze(0)

        model.init()  # the model is re-initialized before learning a new shape
        optimizer = Train.Optim.optim(model.parameters(), **Train.Optim.Params.to_dict())

        range = trange(Train.epochs)
        for _ in range:
            predictions = model(x)
            loss = Train.loss_fn(predictions.squeeze(-1), labels.float())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Warning: visualization slows down the loop considerably
            res = {'points': x.cpu().detach(),
                   'predictions': predictions.cpu().detach(),
                   'labels': labels.cpu().detach()}
            viewer.update(res)

            range.set_postfix(loss=loss.item())
            history.append(loss.item())

        viewer.join()
        plt.plot(history)
        plt.show()
        final_pc, prob = model.to_pc(itr=20, thr=0.85, num_points=8192*2)
        draw(PointCloud(points=Vector3dVector(final_pc.cpu().numpy())))
        # viewer.stop()


if __name__ == '__main__':
    main()
