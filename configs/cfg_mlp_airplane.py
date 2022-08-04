from pathlib import Path

import torch.nn
from torch.optim import SGD

from dataset import ShapeNetDataset
from models.mlp import MLP
from utils.base_config import BaseConfig


class Config(BaseConfig):
    name = Path(__file__).parts[-1]

    class Model(BaseConfig):
        architecture = MLP

        class Params(BaseConfig):
            num_layers = 16
            layers_dim = [96] * 16

    class Data(BaseConfig):
        input_dimension = 16384 * 2
        split = [0.1, 0.4, 0.5]
        noise_rate = 0.1
        tolerance = 0.001

        class DataSet(BaseConfig):
            dataset = ShapeNetDataset

            class Params(BaseConfig):
                root = './data/PCN'
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
        loss_fn = torch.nn.BCEWithLogitsLoss()
        device = 'cuda'
        epochs = int(50_000)

        class Optim(BaseConfig):
            optim = SGD

            class Params(BaseConfig):
                lr = 0.0001
                momentum = 0.9


if __name__ == '__main__':
    import json

    Config.init()
    # with open('configs/current.json', 'w') as fp:
    #     json.dump(Config.to_dict(), fp)
    #
    # print(f'Configuration {Config.name} set as active. \n'
    #       f'Re-run the script if you make any modification.')

    from experiments import overfit
    overfit.main()
