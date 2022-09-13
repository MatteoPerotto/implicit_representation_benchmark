import torch
import torch.nn as nn
import gpytorch

class LargeFeatureExtractor(nn.Sequential):
    def __init__(self, data_dim):
        super(LargeFeatureExtractor, self).__init__()
        self.add_module('linear1', nn.Linear(data_dim, 1000))
        self.add_module('relu1', nn.ReLU())
        self.add_module('linear2', nn.Linear(1000, 500))
        self.add_module('relu2', nn.ReLU())
        self.add_module('linear3', nn.Linear(500, 50))
        self.add_module('relu3', nn.ReLU())
        self.add_module('linear4', nn.Linear(50, 2))

class GPRegressionModel(gpytorch.models.ExactGP):
        def __init__(self, train_x, train_y, likelihood):
            super(GPRegressionModel, self).__init__(train_x, train_y, likelihood)
            self.mean_module = gpytorch.means.ConstantMean()
            self.covar_module = myGPs.ThinPlateRegularizer()
            self.feature_extractor = feature_extractor

            # This module will scale the NN features so that they're nice values
            self.scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(-1., 1.)

        def forward(self, x):
            # We're first putting our data through a deep net (feature extractor)
            projected_x = self.feature_extractor(x)
            projected_x = self.scale_to_bounds(projected_x)  # Make the NN values "nice"

            mean_x = self.mean_module(projected_x)
            covar_x = self.covar_module(projected_x)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

