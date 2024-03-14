import os
os.environ['PYTHONHASHSEED']=str(0)
import numpy as np
np.random.seed(0)
import random
import torch
random.seed(0)
torch.manual_seed(0)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from dataset import Dataset
from model import *
from utils import *

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
nets = {'ConvNet': [ConvNet(), 512, 75], 'FCNet': [FCNet(), 256, 20]}

def evaluate_model(images, n_samples):
    pred_model.eval()
    with torch.no_grad():
        images = images.to(device)
        scores, dist, _, _ = pred_model(images, centroids=torch.tensor(0))
        images = images.cpu().detach().numpy()
        scores = scores.cpu().detach().numpy()
        dist = dist.cpu().detach().numpy()
    pl = evaluate_with_prototypes(args, scores, n_samples)
    return scores, dist, images, pl

def evaluate_with_prototypes(args, scores, n_samples):
    probs = []
    for i in range(scores.shape[0]):
        probs.append(np.argmax(scores[i,:]))
    ci = [np.where(np.array(probs) == i)[0] for i in range(0, args.CLASSES)]
    pl = [len(idx)/n_samples for idx in ci]
    return pl

def get_class_examples(idx, n_samples):
    images, label = dataset.get_test_by_idx(idx, n_samples)
    CE = dataset.get_class_examples(len(images))
    images = torch.stack([images, *CE], 1)
    return images, label

def add_text(ax, fig, text, fontsize, weight, splines=False):
    ax.text(0.5, 0.5, text, va="center", ha="center", fontsize=fontsize, weight=weight)
    ax.set_xticks([], [])
    ax.set_yticks([], [])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    fig.add_subplot(ax)

def plot_img(ax, img):
    ax.imshow(img, cmap='gray')
    ax.set_xticks([], [])
    ax.set_yticks([], [])
    return ax

def plot_probabilities(pl, images, label, classes):
    fig1, ax1 = plt.subplots(1, 1)
    fig1.set_size_inches(4.5*1.5, 3)
    values = classes
    ax1.set_xlabel('Probability')
    ax1.barh(y=values, width=pl, color ='maroon', label="Incorrect Label")
    ax1.barh(y=values[label.item()], width=pl[label], color ='green', label="Correct Label")
    ax1.set_xlim([0,1])
    ax1.legend()
    plt.savefig('projects/twinprotonet/posterior.png', bbox_inches='tight', pad_inches=0.1)
    
    fig2, ax2 = plt.subplots(1, 1)
    ax2 = plot_img(ax2, images[0][0].squeeze(0))
    ax2.set_xlabel(f'Ground Truth: {classes[label]}', fontsize=20)
    plt.savefig('projects/twinprotonet/test_image.png', bbox_inches='tight', pad_inches=0.1)

def plot_prototypes(scores, dist, images, label, classes):
    N_PROTO = 2
    cin = [np.where(np.argmax(scores, 1) == i)[0] for i in range(10)]
    inds = [(ni, i[np.argsort(-dist[i, ni])[:N_PROTO]]) for ni, i in enumerate(cin) if len(i) > 0.1*100]
    n = N_PROTO*len(inds)
    uinds = [i[0] for i in inds]

    fig = plt.figure(figsize=(4+10*(N_PROTO), 10))
    outer = gridspec.GridSpec(2, 5, wspace=0.1, hspace=0.15, width_ratios=[0.5]+([1]*4))
    ax = plt.Subplot(fig, outer[0])
    ax.set_title('Prediction', fontsize=25, weight='bold', pad=30)
    add_text(ax, fig, f'{classes[uinds[0]]}\n Prototypes', 25, 'bold')
    if len(uinds) > 1:
        ax = plt.Subplot(fig, outer[1*(5)])
        add_text(ax, fig, f'{classes[uinds[1]]}\n Prototypes', 25, 'bold')

    count = 0
    for i, ind in inds:
        for nj, j in enumerate(ind):
            for nk, k in enumerate(uinds):
                ax = plt.Subplot(fig, outer[(n+1)*nk+(count+1)])
                if nk == 0:
                    ax.set_title(f'{classes[i]}', fontsize=25, pad=30)
                ax = plot_img(ax, images[j, k+1, 0, :, :])
                mydist = dist[j, k]
                ax.set_xlabel(f'Score: {mydist:.2f}', fontsize=25, color='orange')
                if k == i:
                    ax.set_xlabel(f'Score: {mydist:.2f}', color='blue', fontsize=25)
                fig.add_subplot(ax)
            count += 1

    # plt.show()
    plt.savefig('projects/twinprotonet/proto.png', bbox_inches='tight', pad_inches=0.1) 

def plot_prototypes_grid(n_ex, idx, fig, outer, scores, dist, images, label, classes, pl):
    N_PROTO = 4
    cin = [np.where(np.argmax(scores, 1) == i)[0] for i in range(10)]
    pred = np.argsort(-np.array(pl))[:2]
    cin = [(p, cin[p]) for p in pred]
    inds = [(ni, i[np.argsort(-dist[i, ni])[:N_PROTO]]) for ni, i in cin]
    n = N_PROTO*len(inds)
    uinds = [i[0] for i in inds]

    ax = plt.Subplot(fig, outer[idx])
    ax = plot_img(ax, images[0][0].squeeze(0))
    ax.set_title(f'Ground Truth: {classes[label]}', fontsize=20)
    fig.add_subplot(ax)

    inner = gridspec.GridSpecFromSubplotSpec(2, 3,
                    subplot_spec=outer[n_ex+idx], wspace=0.05, hspace=0.05, width_ratios=[0.2, 1, 0.005], height_ratios=[1, 0.2])
    ax = plt.Subplot(fig, inner[1])
    values = classes
    ax.set_xlabel('Probability')
    
    ax.barh(y=values, width=pl, color ='maroon', label="Incorrect Label")
    ax.barh(y=values[label.item()], width=pl[label], color ='green', label="Correct Label")
    ax.set_xlim([0,1])
    ax.legend()
    fig.add_subplot(ax)
    count = 0
    for i, ind in inds:
        inner = gridspec.GridSpecFromSubplotSpec(2, 1,
                    subplot_spec=outer[n_ex*(2+count)+idx], wspace=0, hspace=0, height_ratios=[1, 15])
        if len(ind) > 0:
            ax = plt.Subplot(fig, inner[0])
            add_text(ax, fig, f'{classes[i]} Prototypes', 20, 'normal')
        else:
            ax = plt.Subplot(fig, inner[1])
            add_text(ax, fig, 'None', fontsize=20, weight='normal', splines=True)
        inner1 = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=inner[1], wspace=0.05, hspace=0.2)
        for nj, j in enumerate(ind):
            ax = plt.Subplot(fig, inner1[nj])
            ax = plot_img(ax, images[j, i+1, 0, :, :])
            mydist = dist[j, i]
            ax.set_xlabel(f'Score: {mydist:.2f}', fontsize=16)
            fig.add_subplot(ax)
        count += 1

    plt.savefig('projects/twinprotonet/proto_grid.png', bbox_inches='tight', pad_inches=0.5) 

def visualize_single(path, dataset, classes, pred_model, n_samples=100):
    print(f'Loading {path}')
    wrong_idx = np.load(path+f"/wrong_indices_0.npy")
    print(f'Incorrect test indices: {wrong_idx}')

    response = ''
    while response != 'exit':
        response = input("Which index or 'exit': ")
        if response == 'exit':
            break
        idx = int(response)
        images, label = get_class_examples(idx, n_samples)
        scores, dist, images, pl = evaluate_model(images, n_samples)
        plot_probabilities(pl, images, label, classes)
        plot_prototypes(scores, dist, images, label, classes)

def visualize_grid(path, dataset, classes, pred_model, n_samples=100):
    n_ex = int(input("Enter number of test images: "))
    lst = list(map(int, input("Enter the indices separated by space: ").strip().split()))[:n_ex]
     
    n_ex = len(lst)
    outer = gridspec.GridSpec(4, 1+n_ex, wspace=0.2, hspace=0.2, height_ratios=[1, 0.7, 1, 1], width_ratios=[0.4]+([1]*n_ex))
    fig = plt.figure(figsize=(2+(n_ex)*5, 23))
    ax = plt.Subplot(fig, outer[0])
    add_text(ax, fig, 'Test Image', 20, 'bold')
    ax = plt.Subplot(fig, outer[1*(n_ex+1)])
    add_text(ax, fig, 'Posterior \n Distribution', 20, 'bold')
    ax = plt.Subplot(fig, outer[2*(n_ex+1)])
    add_text(ax, fig, 'Most \n Common \n Prediction', 20, 'bold')
    ax = plt.Subplot(fig, outer[3*(n_ex+1)])
    add_text(ax, fig, '2nd Most \n Common \n Prediction', 20, 'bold')
    for i, idx in enumerate(lst):
        images, label = get_class_examples(idx, n_samples)
        scores, dist, images, pl = evaluate_model(images, n_samples)
        pl = evaluate_with_prototypes(args, scores, n_samples)
        plot_prototypes_grid(n_ex+1, i+1, fig, outer, scores, dist, images, label, classes, pl)
        
if __name__ == "__main__":
    path = f'models/fmnist_twin_ConvNet_256_256_0.05'

    args, pred_model, _ = load_model(path, 0)
    dataset = Dataset(args.DATASET)
    classes = dataset.classes

    figtype = input("'single' or 'grid': ")
    if figtype == 'single':
        visualize_single(path, dataset, classes, pred_model)
    if figtype == 'grid':
        visualize_grid(path, dataset, classes, pred_model)