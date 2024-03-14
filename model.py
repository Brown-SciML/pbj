import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class ConvNet(nn.Module):
    # Model architecture code from: https://github.com/y0ast/deterministic-uncertainty-quantification/blob/master/utils/cnn_duq.py
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(128, 128, 3)
        self.bn3 = nn.BatchNorm2d(128)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2, 2)

        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2, 2)

        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, 2, 2)

        x = x.flatten(1)
        return x
    
class FCNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 256)

    def forward(self, x):
        return F.relu(self.fc1(x))
    
class DistNet(nn.Module):
    def __init__(self, latent_dim=512, num_classes=10, regular=False):
        super().__init__()
        if regular:
            self.fc1 = nn.Linear(latent_dim, num_classes, bias=False)
        else:
            self.fc1 = nn.Linear(num_classes, num_classes, bias=False)
            init_w = torch.eye(num_classes)*(num_classes/(num_classes-1))*100-(1/(num_classes-1))*100
            self.fc1.weight = nn.Parameter(init_w)

    def forward(self, x):
        return self.fc1(x)

class PredictionNet(nn.Module):
    def __init__(self, convnet, distnet, inter_dim=128, latent_dim=512, n_classes=10, regular=False):
        super().__init__()
        self.regular = regular
        self.convnet = convnet
        self.fc1 = nn.Linear(inter_dim, latent_dim, bias=False)
        self.bn1 = nn.BatchNorm1d(latent_dim)
        self.distnet = distnet
        self.n_classes = n_classes

    def calc_latent(self, x):
        x = self.convnet(x)                                         # (batch * (1 + n_examples * n_classes), o_channel, oh, ow) [-1, 512, 7, 7]   
        return self.bn1(self.fc1(x))

    def get_class_examples(self, x, y, n_ex):
        class_examples = []
        class_idxs = [np.where(y.cpu().detach().numpy() == i)[0] for i in range(0, self.n_classes)]
        for i in range(self.n_classes):
            idx = np.random.choice(class_idxs[i], n_ex, replace=True)
            class_examples.append(x[idx])
        class_examples = torch.stack(class_examples, 1)
        return class_examples

    def check_labels(self, X, y, dataset):
        missing = list(set(np.arange(dataset.n_classes)) - set(y.cpu().detach().numpy()))
        for i in missing:
            X = torch.cat([X, dataset.get_example_from_class(i).to(device)], 0)
            y = torch.cat([y, torch.tensor([i]).to(device)], 0)
        return X, y

    def get_latent_set(self, x, y=None, dataset=None, centroids=None):
        if centroids is None:
            if dataset.centroid_dl_iter is not None:
                ex, ex_y = next(dataset.centroid_dl_iter)
                ex = ex.to(device)
                ex_y = ex_y.to(device)
                ex, ex_y = self.check_labels(ex, ex_y, dataset)
                both = torch.cat([x, ex], 0)
                both = self.calc_latent(both) 
                CE = self.get_class_examples(both[len(x):], ex_y, len(x)).to(device)                               
                x = torch.concat([both[:len(x)].unsqueeze(1), CE], 1)      
            else:
                x, y = self.check_labels(x, y, dataset)
                x = self.calc_latent(x) 
                CE = self.get_class_examples(x, y, len(x)).to(device)
                x = torch.concat([x.unsqueeze(1), CE], 1)
        else:   
            if torch.sum(centroids) == 0:
                n = x.shape[1]                                            # B*C*H*W (batch, 1 + n_examples * n_classes, i_channel, ih, iw)
                x = x.reshape(-1, *x.shape[2:]) 
                x = self.calc_latent(x)                                           # (batch * (1 + n_examples * n_classes), o_channel, oh, ow) [-1, 512, 7, 7]                                        # (batch * (1 + n_examples * n_classes), latent_dim)
                x = x.view(-1, n, x.shape[-1])                                                       # (batch * (1 + n_examples * n_classes), i_channel, ih, iw)
            else:
                centroids = centroids.unsqueeze(0).repeat(x.shape[0],1,1)
                x = self.calc_latent(x)                                   # (batch * (1 + n_examples * n_classes), latent_dim)
                x = torch.concat([x.unsqueeze(1), centroids], 1)                              # (batch, 1 + n_examples * n_classes, latent_dim)
        return x, y
    
    def calc_dist(self, x):
        anchor_output = x[:,0,:].unsqueeze(1)                         # (batch, 1, output_dim)
        class_outputs = x[:,1:,:]                                     # (batch, n_classes, output_dim)
        dist = torch.cdist(anchor_output, class_outputs).squeeze()    # (batch, 1, n_classes)
        return torch.log((torch.square(dist) + 1.0) / (torch.square(dist) + 1e-10))

    def forward(self, x, y=None, dataset=None, centroids=None):
        if self.regular:
            x = self.calc_latent(x)
            return self.distnet(x), None, None, y
        else:
            ls, y = self.get_latent_set(x, y=y, dataset=dataset, centroids=centroids)
            d = self.calc_dist(ls)
            x = self.distnet(d)
            return x, d, ls, y
