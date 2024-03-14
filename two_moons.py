import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import scipy.optimize as optimize
from matplotlib import pyplot as plt

from dataset import Dataset
from model import *
from utils import *

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
nets = {'ConvNet': [ConvNet(), 512, 75], 'FCNet': [FCNet(), 256, 20]}

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def load_model(filepath, model_id):
    args = load_args(filepath+'/train_settings.npz')
    cnn_model = nets[args.NET][0]
    dist_model = DistNet(args.LATENT_DIM, args.CLASSES, args.REGULAR)
    pred_model = PredictionNet(cnn_model, dist_model, nets[args.NET][1], args.LATENT_DIM, args.CLASSES, args.REGULAR)
    pred_model.load_state_dict(torch.load(filepath+f'/model_{model_id}.pt').get('model_state_dict'))
    pred_model = pred_model.to(device)
    pred_model.eval()
    try:
        centroids = torch.tensor(np.load(path+f'/centroids_0.npy')).to(device)
    except:
        centroids = None
    return args, pred_model, centroids

def get_test_grid():
    bnds = ((0, None), (0, None))
    domain = 3
    x_lin = np.linspace(-domain+0.5, domain+0.5, 100)
    y_lin = np.linspace(-domain, domain, 100)
    xx, yy = np.meshgrid(x_lin, y_lin)
    X_grid = np.column_stack([xx.flatten(), yy.flatten()])
    n, iw = X_grid.shape
    grid_dl = DataLoader(torch.tensor(X_grid), batch_size=500, shuffle=False)
    return grid_dl, x_lin, y_lin, xx.shape

def plot_grid(z, dataset, x_lin, y_lin):
    cmap = plt.get_cmap('cividis')
    plt.clf()
    plt.set_cmap(cmap)
    cbar_min = 0
    cbar_max = 1
    plt.contourf(x_lin, y_lin, z, cmap=cmap, levels=np.linspace(cbar_min, cbar_max, num=5, endpoint=True))
    cbar = plt.colorbar()
    cbar.set_ticks(np.linspace(cbar_min, cbar_max, num=5, endpoint=True))

    X_test_vis = dataset.X_train[:200]
    y_test_vis = dataset.y_train[:200]
    mask = np.bool_(y_test_vis)
    plt.scatter(X_test_vis[mask,0], X_test_vis[mask,1], facecolors='none', edgecolors='r')
    plt.scatter(X_test_vis[~mask,0], X_test_vis[~mask,1], facecolors='none',  edgecolors='b')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.savefig('projects/twinprotonet/twomoons.png', bbox_inches='tight', pad_inches=0.1)

def get_train_stats(dataset, pred_model, centroids):
    ucd = []
    with torch.no_grad():
        for images, labels in dataset.train_dl:
        images = images.to(device).float()
        with torch.no_grad():
            score, dist, _, _ = pred_model(images, centroids=centroids)
            ucd.append(dist.cpu().detach().numpy())
    ucd = np.amax(np.concatenate(ucd, 0), 1)
    ucd_std = np.std(ucd)
    ucd_05 = np.sort(ucd)[int(1500*(1-0.95))]
  return ucd_05, ucd_std

def calculate_uc(grid_dl, pred_model, centroids, regular):
    uc = []
    with torch.no_grad():
        for images in grid_dl:
        images = images.to(device).float()
        with torch.no_grad():
            if regular:
            score, _, _, _ = pred_model(images)
            uc.append(torch.softmax(score, 1).cpu().detach().numpy())
            else:
            score, dist, _, _ = pred_model(images, centroids=centroids)
            uc.append(dist.cpu().detach().numpy())
    uc = np.concatenate(uc, 0)
    return np.amax(uc, 1)

if __name__ == "__main__":
    path = f'models/moon_twin_FCNet_256_128_0.05'
    args, pred_model, centroids = load_model(path, 0)
    dataset = Dataset(args.DATASET)
    dataset.set_train_dataloader(args.TRAIN_BATCH_SIZE)
    dataset.set_centroid_dataloader(args.TRAIN_BATCH_SIZE)

    grid_dl, x_lin, y_lin, x_shape = get_test_grid()
    uc = calculate_uc(grid_dl, pred_model, centroids, args.REGULAR)
    if not args.REGULAR:
        ucd_05, ucd_std = get_train_stats(dataset, pred_model, centroids)
        uc = sigmoid((uc-(ucd_05))/(ucd_std))
    z = uc.reshape(x_shape)
    plot_grid(z, dataset, x_lin, y_lin)