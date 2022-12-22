from matplotlib import pyplot as plt
#from open3d.cpu.pybind.geometry import PointCloud
#from open3d.cpu.pybind.utility import Vector3dVector
from open3d.visualization import draw
from torch.utils.data import DataLoader
from tqdm import trange

from configs import Config
from dataset.utils import sample_point_cloud_gpis
from utils.visualization_vispy import Visualizer
import open3d as o3d 
import torch 

import gpytorch 
from models.gpis import GPRegressionModel as GPIS
from models.mlp import MLP

def main():
    """This code shows how the predictions are updated
        as the model learns how to reconstruct the shape
    """
    Model = Config.Model
    Data = Config.Data
    Train = Config.Train
    Sched = Config.Sched
    
    dataset = Data.DataSet.dataset(**Config.Data.DataSet.Params.to_dict())
    dataloader = DataLoader(dataset, **Config.Data.DataLoader.Params.to_dict())

    for gt in dataloader:
        
        #gt = gt.to(Train.device)    
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(gt.squeeze().numpy())

        # generate training data        
        x, labels = sample_point_cloud_gpis(pcd, trainN=200, outDim=0.01)
        x, labels, gt = x.squeeze(0), labels.squeeze(0), gt.squeeze(0)

        print(type(x))
        print(x.shape)

        print(type(labels))
        print(labels.shape)

        model_arg = Config.Model.Params.to_dict()
        print(model_arg)
        model_arg.update({'train_x': x, 'train_y':labels})
        #print(model_arg)
        #model = Model.architecture(**model_arg)

        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        likelihood.noise_covar.register_constraint("raw_noise", gpytorch.constraints.Positive())

        feature_extractor = MLP(input_size = 3, output_size = 10, num_layers = 1, layers_dim = [50])

        model = GPIS(feature_extractor, x, labels, likelihood)

        hypers = {
        'likelihood.noise_covar.noise': torch.tensor(1e-4), 
        'covar_module.max_dist': torch.tensor(0.4),
        }
        model.initialize(**hypers)  # the model is re-initialized before learning a new shape

        optimizer = Train.Optim.optim(model.parameters(), **Train.Optim.Params.to_dict())
        scheduler = Sched.sched(**Sched.Params.to_dict())
        
        range = trange(Train.epochs)
        for _ in range:
            predictions = model(x)
            loss = gpytorch.mlls.ExactMarginalLogLikelihood(predictions.squeeze(-1), labels.float())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

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
                    
        # viewer.stop()


if __name__ == '__main__':
    main()
