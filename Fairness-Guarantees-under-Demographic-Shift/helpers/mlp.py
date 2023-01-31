import numpy as np
import warnings
import torch
import torch.nn.functional as F
from torch import nn

class MLP(nn.Module):
    def __init__(self, in_dim, hidden_layers, n_class):
        super(MLP, self).__init__()

        layers = []

        layer_sizes = [in_dim] + hidden_layers + [n_class]
        for n_in, n_out in zip(layer_sizes[:-2], layer_sizes[1:-1]):
            layers.extend([nn.Linear(n_in, n_out), nn.Tanh()])

        layers.append(nn.Linear(layer_sizes[-2], layer_sizes[-1]))
        self.model = nn.Sequential(layers)
       
    def forward(self, x):
        Theta_x = self.model(x)
        out = np.sign(Theta_x)
        return out, Theta_x


class MLPOptimizer():
    def __init__(self, weight, Sen, ratio, model, train_loader, optimizer, num_epochs=10):
        self.weight = weight
        self.Sen = Sen
        self.ratio = ratio
        self.model = model
        self.train_loader = train_loader
        self.optimizer = optimizer
        self.num_epochs = num_epochs

    def minimize(self, evalf, n_iters=1000, theta0=None):
        for _ in range(n_iters):
            self.model.train()
            for i,  (data, target) in enumerate(self.train_loader):
                label = target.data.numpy()
                for i in range(len(label)):
                    label[i] = 0 if label[i] == 0.0 else 1

                label = torch.LongTensor(label)
                label = label.view(len(label))

                out, Theta_x = self.model(data)

                _, pred = torch.max(out.data, 1)
                candidate_loss = candidate_objective(self, weight, split, self.ratio) # ???
                loss = loss + candidate_loss
                running_acc += torch.sum((pred==label))

                self.optimizer.zero_grad()

                loss.backward()

                self.optimizer.step()

            running_acc = running_acc.numpy()

        return loss_vector.detach().numpy(), Theta_x.detach().numpy()
    