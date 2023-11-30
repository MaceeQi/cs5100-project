from typing import Tuple

import torch


class FFAdamParams(object):
    def __init__(self,
                 lr: float = 0.001,
                 betas: Tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-08,
                 weight_decay: float = 0,
                 amsgrad: bool = False):
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.amsgrad = amsgrad


class FFLinear(torch.nn.Linear):
    def __init__(self,
                 tune_params: FFAdamParams,
                 in_features,
                 out_features,
                 num_epochs: int = 100,
                 thresh: float = 2.0,
                 active_func=torch.nn.ReLU(),
                 has_active_func=True,
                 bias=True,
                 device=None,
                 dtype=None):

        if not device:
            device = torch.device("cpu")

        super(FFLinear, self).__init__(in_features, out_features, bias, device, dtype)

        self.active_func = active_func
        self.has_active_func = has_active_func
        self.num_epochs = num_epochs
        self.thresh = thresh
        self.final_loss = torch.tensor([]).to(device)
        self.optimizer = torch.optim.Adam(self.parameters(),
                                          lr=tune_params.lr,
                                          betas=tune_params.betas,
                                          eps=tune_params.eps,
                                          weight_decay=tune_params.weight_decay,
                                          amsgrad=tune_params.amsgrad)

    def forward(self, data):
        orient = data / (data.norm(2, 1, keepdim=True) + 1e-4)

        if self.has_active_func:
            return self.active_func(torch.matmul(orient, self.weight.T) + self.bias.unsqueeze(0))
        else:
            return torch.matmul(orient, self.weight.T) + self.bias.unsqueeze(0)

    def train(self, d_pos, d_neg):
        for epoch in range(self.num_epochs):
            g_pos, g_neg = self.forward(d_pos).pow(2).mean(1), \
                           self.forward(d_neg).pow(2).mean(1)
            loss = (torch.log(1 + torch.exp(self.thresh - g_pos)) + torch.log(
                1 + torch.exp(g_neg - self.thresh))).mean() / 2
            self.final_loss = loss.clone().detach_()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        for param in self.parameters():
            param.grad = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.final_loss = self.final_loss.cpu() if self.final_loss.is_cuda else self.final_loss

        return self.forward(d_pos).detach_(), self.forward(d_neg).detach_()


FFLayers = (FFLinear)


class FFSupervisedVision(torch.nn.Module):
    def __init__(self, layers: list, total_classes: int, device=None):
        super().__init__()

        if not device:
            device = torch.device("cpu")

        self.total_classes = total_classes
        self.layers = torch.nn.ModuleList([layer.to(device) for layer in layers if isinstance(layer, FFLayers)])

    @property
    def final_losses(self):
        losses = []

        for layer in self.layers:
            losses.append(layer.final_loss.item())

        return losses

    def train(self, data, labels):
        g_pos, g_neg = self.__data_precondition(data, labels),\
                       self.__data_precondition(data, labels[torch.randperm(data.size(0))])

        for layer in self.layers:
            g_pos, g_neg = layer.train(g_pos, g_neg)

    def predict(self, data):
        with torch.no_grad():
            g_per_label = list()

            for label in range(self.total_classes):
                data_good = self.__data_precondition(data, label)
                goodness = []

                for i, layer in enumerate(self.layers):
                    data_good = layer(data_good)
                    goodness.append(data_good.pow(2).mean(1))

                g_per_label += [sum(goodness).unsqueeze(1)]

            preds = torch.cat(g_per_label, 1).argmax(1)

            return preds.cpu() if preds.is_cuda else preds

    def __data_precondition(self, x, y):
        x_ = x.clone()
        x_[:, :self.total_classes] *= 0.0
        x_[range(x.shape[0]), y] = 1.0
        return x_
