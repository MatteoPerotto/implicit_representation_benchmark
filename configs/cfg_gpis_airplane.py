import torch.nn
from torch.optim import SGD

from dataset import ShapeNetDataset
from models.gpis import GPRegressionModel as GPIS
from models.mlp import MLP
from utils.base_config import BaseConfig


class Config(BaseConfig):

    class Model(BaseConfig):
        architecture = GPIS

        class Params(BaseConfig):

            feature_extractor = MLP(input_size = 3, output_size = 10, num_layers = 1, layers_dim = [50])
            likelihood = gpytorch.likelihoods.GaussianLikelihood()
            likelihood.noise_covar.register_constraint("raw_noise", gpytorch.constraints.Positive())
            scale_to_bounds = False

    class Data(BaseConfig):

        class DataSet(BaseConfig):
            dataset = ShapeNetDataset

            class Params(BaseConfig):
                root = '../pcr/data/PCN'
                split = 'PCN.json'
                subset = 'train'
                length = None
                pick = [0]

        class DataLoader(BaseConfig):

            class Params(BaseConfig):
                batch_size = 1
                shuffle = False
                num_workers = 0
                pin_memory = True

    class Train(BaseConfig):
        loss_fn = gpytorch.mlls.ExactMarginalLogLikelihood()
        device = 'cuda'
        epochs = 300

        class Optim(BaseConfig):
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[25,50,100,200])

            class Params(BaseConfig):
                lr = 0.05

        class Scheduler(BaseConfig):
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[25,50,100,200])

            class Params(BaseConfig):
                milestones=[25,50,100,200]


if __name__ == '__main__':
    Config.init()

    from experiments import visualize_learning
    visualize_learning.main()
