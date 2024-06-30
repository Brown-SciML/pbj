import os
os.environ['PYTHONHASHSEED']=str(0)
import numpy as np
np.random.seed(0)
import random
random.seed(0)
import torch
torch.manual_seed(0)
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
from scipy import stats
import scipy.optimize as optimize

from dataset import Dataset
from model import *
from utils import *
from resnet import *

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)
nets = {'ConvNet': [ConvNet(), 512, 75]}
    
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def get_average_auroc(filepath, train_dl, id_data_dl, ood_data_dl):
    auroc = []
    labels = ['0', '1', '2', '3', '4', '5']
    id_ucs, id_corrects, ood_ucs = get_uc_from_paths([(filepath, i) for i in range(5)], train_dl, id_data_dl, ood_data_dl)
    for id_uc, odd_uc, label in zip(*(id_ucs, ood_ucs, labels)):
        dc = [1]*len(id_uc) + [0]*len(odd_uc)
        uc = np.concatenate([id_uc, odd_uc], 0)
        auroc.append(roc_auc_score(dc, uc))
    return auroc

def get_uc_from_paths(filepaths, train_dl, id_data_dl, ood_data_dl):
    id_ucs, ood_ucs, id_corrects = [], [], []
    variability = []
    for filepath, model_id in filepaths:
        args, pred_model, centroids = load_model(filepath, model_id)

        id_y, id_pred, id_uc = get_uc(id_data_dl, pred_model, centroids)
        id_correct = np.where(id_y == id_pred, 1, 0)
        _, _, ood_uc = get_uc(ood_data_dl, pred_model, centroids)

        id_ucs.append(id_uc)
        id_corrects.append(id_correct)
        ood_ucs.append(ood_uc)

    return id_ucs, id_corrects, ood_ucs

def get_uc(dl, pred_model, centroids):
    y, pred, uc = [], [], []
    pred_model.eval()
    for images, labels in dl:
        images = images.to(device)
        y.append(labels)
        with torch.no_grad():
            if centroids is None:
                score, dist, _, _ = pred_model(images)
                uc.append(torch.amax(torch.softmax(score, 1), 1))
            else:
                score, dist, _, _ = pred_model(images, centroids=centroids)
                uc.append(torch.amax(dist, 1))
            pred.append(torch.argmax(score, 1))
    y = torch.cat(y, 0).cpu().detach().numpy()
    pred = torch.cat(pred, 0).cpu().detach().numpy()
    uc = torch.cat(uc, 0).cpu().detach().numpy()
    return y, pred, uc

if __name__ == "__main__":
    filepath = 'models/fmnist_twin_ConvNet_256_256_0.05'
    args, _, _ = load_model(filepath, 0)

    ood_dataset_name = {'mnist': 'fmnist', 'fmnist': 'mnist'}
    id_data = Dataset(args.DATASET)
    ood_data = Dataset(ood_dataset_name[args.DATASET], ood=True)
    id_data.set_train_dataloader(args.TRAIN_BATCH_SIZE, drop_last=False)
    id_data.set_test_dataloader(args.TEST_BATCH_SIZE)
    ood_data.set_test_dataloader(args.TEST_BATCH_SIZE)

    acc = np.load(filepath+'/test_accuracies.npy')
    print(acc)
    print(np.mean(acc))
    print(np.std(acc))

    auroc = get_average_auroc(filepath, id_data.train_dl, id_data.test_dl, ood_data.test_dl)
    print(auroc)
    print(np.mean(auroc))
    print(np.std(auroc))