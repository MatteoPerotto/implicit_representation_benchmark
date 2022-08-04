from pathlib import Path

import torch.nn
from torch.optim import SGD

from dataset import ColumbiaDataset
from models.mlp import MLP
from utils.base_config import BaseConfig


class Config(BaseConfig):
    name = Path(__file__).parts[-1]

    class Model(BaseConfig):
        architecture = MLP

        class Params(BaseConfig):
            num_layers = 2
            layers_dim = [96] * 2

    class Data(BaseConfig):
        input_dimension = 16384
        split = [0.1, 0.4, 0.5]
        noise_rate = 0.1
        tolerance = 0.005

        class DataSet(BaseConfig):
            dataset = ColumbiaDataset

            class Params(BaseConfig):
                root = '../pcr/data/MCD'
                split = 'build_datasets/train_test_dataset.json'
                subset = 'train_models_train_views'
                length = None
                pick = [28]

        class DataLoader(BaseConfig):

            class Params(BaseConfig):
                batch_size = 1
                shuffle = False
                num_workers = 0
                pin_memory = True

    class Train(BaseConfig):
        loss_fn = torch.nn.BCEWithLogitsLoss()
        device = 'cuda'
        epochs = int(1e4)

        class Optim(BaseConfig):
            optim = SGD

            class Params(BaseConfig):
                lr = 0.01
                momentum = 0.9




if __name__ == '__main__':
    Config.init()

    from experiments import visualize_learning
    visualize_learning.main()
