import numpy as np
import torch 
import argparse
from model import *
from resnet import * 

### GLOBAL VARIABLES
NUM_REPEATS        = 5
DATASET            = 'fmnist'
NET                = 'ConvNet'
CLASSES            = 10
TRAIN_BATCH_SIZE   = 256
TEST_BATCH_SIZE    = 1000
LATENT_DIM         = 256
LEARNING_RATE      = 0.05

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--REGULAR', default=False, action='store_true')
    parser.add_argument('--NUM_REPEATS', type=int, default=NUM_REPEATS)
    parser.add_argument('--DATASET', type=str, default=DATASET)
    parser.add_argument('--NET', type=str, default=NET)
    parser.add_argument('--FREEZE', default=False, action='store_true')
    parser.add_argument('--CLASSES', type=int, default=CLASSES)
    parser.add_argument('--TRAIN_BATCH_SIZE', type=int, default=TRAIN_BATCH_SIZE)
    parser.add_argument('--TEST_BATCH_SIZE', type=int, default=TEST_BATCH_SIZE)
    parser.add_argument('--LATENT_DIM', type=int, default=LATENT_DIM)
    parser.add_argument('--LEARNING_RATE', type=float, default=LEARNING_RATE)
    parser.add_argument('--CLAMP_MIN', type=float)
    parser.add_argument('--CLAMP_MAX', type=float)
    parser.add_argument('--INIT_WEIGHT', type=float, default=INIT_WEIGHT)
    return parser.parse_args()

def load_args(path):
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    npzfile = np.load(path, allow_pickle=True)
    args_dict = {}
    for i in npzfile.files:
        args_dict[i] = npzfile[i].item()
    args.__dict__ = args_dict
    return args

def load_model(filepath, model_id):
    args = load_args(filepath+'/train_settings.npz')
    nets = {'ResNet50': [ResNet50(), 2048, 200], 'ResNet34': [ResNet34(), 512, 200], 'ResNet18': [ResNet18(), 512, 200], 'ConvNet': [ConvNet(), 512, 75], 'FCNet': [FCNet(), 256, 20]}

    cnn_model = nets[args.NET][0]
    try:
        dist_model = DistNet(args.LATENT_DIM, args.CLASSES, args.INIT_WEIGHT, args.REGULAR)
    except:
        dist_model = DistNet(args.LATENT_DIM, args.CLASSES, 100, args.REGULAR)
    pred_model = PredictionNet(cnn_model, dist_model, nets[args.NET][1], args.LATENT_DIM, args.CLASSES, args.REGULAR)
    pred_model.load_state_dict(torch.load(filepath+f'/model_{model_id}.pt', map_location=torch.device('cpu')).get('model_state_dict'))
    pred_model = pred_model.to(device)
    pred_model.eval()
    if args.REGULAR:
        centroids = None
    else:
        centroids = torch.tensor(np.load(filepath+f'/centroids_{model_id}.npy')).to(device)
    return args, pred_model, centroids

def calc_accuracy(logit, target, return_idx=False):
    b = len(target)
    corrects = (torch.max(logit, 1)[1].view(target.size()).data == target.data)
    accuracy = 100.0 * corrects.sum()/b
    if return_idx:
        w_idx = torch.where(corrects==False)[0]
        return accuracy.item(), w_idx
    return accuracy.item() 

def save_model(epoch, pred_model, optimizer, scheduler, path):
    try: 
        scheduler_state = scheduler.state_dict()
    except:
        scheduler_state = None

    torch.save({'epoch': epoch,
                'model_state_dict': pred_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler_state
                }, path)
