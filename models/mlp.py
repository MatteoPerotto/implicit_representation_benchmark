import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam


class MLP(nn.Module):

    def __init__(self, num_layers, layers_dim):
        super().__init__()

        if not isinstance(layers_dim, list):
            layers_dim = [layers_dim] * num_layers

        self.input_lyr = nn.Linear(3, layers_dim[0])
        self.output_lyr = nn.Linear(layers_dim[-1], 1)

        lyr_list = []
        for i in range(len(layers_dim) - 1):
            lyr_list += [nn.Linear(layers_dim[i], layers_dim[i + 1])]

        self.hidden_lyrs = nn.Sequential(*lyr_list)

    def forward(self, x):
        x = torch.relu(self.input_lyr(x))
        for lyr in self.hidden_lyrs:
            x = torch.relu(lyr(x))
        out = self.output_lyr(x)

        return out

    def to_pc(self, itr, thr, num_points):
        device = self.input_lyr.weight.device

        with torch.enable_grad():
            old = self.training
            self.eval()

            refined_pred = torch.tensor(torch.randn(num_points * 2, 3).cpu().detach().numpy() * 1, device=device,
                                        requires_grad=True)

            loss_function = BCEWithLogitsLoss(reduction='mean')
            optim = Adam([refined_pred], lr=0.1)

            new_points = []
            # refined_pred.detach().clone()
            for step in range(itr):
                results = self(refined_pred)


                idxs = torch.sigmoid(results).squeeze() >= thr
                points = refined_pred.detach().clone()[idxs, :]
                preds = torch.sigmoid(results).detach().clone()[idxs]
                new_points += [torch.cat([points, preds], dim=1)]

                gt = torch.ones_like(results[..., 0], dtype=torch.float32)
                gt[:] = 1
                loss = loss_function(results[..., 0], gt)

                self.zero_grad()
                optim.zero_grad()
                loss.backward(inputs=[refined_pred])
                optim.step()

        res = torch.cat(new_points)

        perm = torch.randperm(res.size(0))
        res = res[perm[:num_points]]

        return res[..., :3], res[..., -1]

    def init(self):
        self.apply(init_weights)


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        m.bias.data.fill_(0)

#
# import torch
# from torch import enable_grad
# from torch.nn import BCEWithLogitsLoss
# from torch.optim import Adam
# from configs import Config
#
# class Decoder:
#     def __init__(self, sdf):
#         self.num_points = Config.Model.Decoder.num_points
#         if not isinstance(Config.Model.Decoder.thr, list):
#             self.thr = [Config.Model.Decoder.thr, 1.0]
#         else:
#             self.thr = Config.Model.Decoder.thr
#         self.itr = Config.Model.Decoder.itr
#
#
#         self.sdf = sdf
#
#     def __call__(self, fast_weights):


