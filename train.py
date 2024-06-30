import os
os.environ['PYTHONHASHSEED']=str(0)
import random
random.seed(0)
import numpy as np
np.random.seed(0)
import torch
import torch.nn.functional as F
torch.manual_seed(0)
import time

from dataset import Dataset
from model import *
from utils import *
from resnet import *

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

def train_step(args, epoch, dataset, pred_model, optimizer, scheduler, metric_history):
    st = time.time()
    train_acc = 0.0
    train_loss = 0.0
    sm_metric = 0
    pred_model.train()
    for images, labels in dataset.train_dl:        
        labels = labels.to(device)
        images = images.to(device)

        optimizer.zero_grad()
        scores, _, r, labels = pred_model(images, y=labels, dataset=dataset)
    
        ce_loss = F.cross_entropy(scores, labels.long()) 
        loss = ce_loss 
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_acc += calc_accuracy(scores, labels)
        sm_metric += len(torch.where(torch.amax(torch.softmax(scores, 1), 1) == 1)[0])

        if args.CLAMP_MAX is not None and args.CLAMP_MIN is not None:
            with torch.no_grad():
                pred_model.distnet.fc1.weight.clamp_(args.CLAMP_MIN, args.CLAMP_MAX)

    train_loss = train_loss/(len(dataset.train_dl))
    train_acc = train_acc/(len(dataset.train_dl))
    metric_history['train_acc'].append(train_acc)

    test_acc = 0.0
    pred_model.eval()
    wrong_idx = []
    with torch.no_grad():
        if args.REGULAR == False:
            # calculate latent space centroids 
            centroids = torch.zeros(args.CLASSES, args.LATENT_DIM)
            class_counts = torch.zeros(args.CLASSES)
            for images, labels in dataset.centroid_dl:
                images = images.to(device)
                ls = pred_model.calc_latent(images)
                class_idxs = [np.where(labels == i)[0] for i in range(0, args.CLASSES)]
                for c, idxs in enumerate(class_idxs):
                    centroids[c, :] += torch.sum(ls[idxs], 0).cpu()
                    class_counts[c] += len(idxs)
            centroids = centroids/class_counts.unsqueeze(1)
            centroids = centroids.to(device)

            # evaluation using test set 
            for i, batch in enumerate(dataset.test_dl):
                images = batch[0].to(device)
                labels = batch[1].to(device)
                scores, _, _, _ = pred_model(images, centroids=centroids)
                acc, idx = calc_accuracy(scores, labels, return_idx=True)
                wrong_idx.extend(idx+(i*args.TEST_BATCH_SIZE))
                test_acc += acc
        else:
            centroids = None
            for i, batch in enumerate(dataset.test_dl):
                images = batch[0].to(device)
                labels = batch[1].to(device)
                scores, _, _, _ = pred_model(images)
                acc, idx = calc_accuracy(scores, labels, return_idx=True)
                wrong_idx.extend(idx+(i*args.TEST_BATCH_SIZE))
                test_acc += acc

    wrong_idx = [idx.tolist() for idx in wrong_idx]
    test_acc = test_acc/(len(dataset.test_dl))
    metric_history['test_acc'].append(test_acc)

    if scheduler is not None:
        scheduler.step()

    print('Epoch: %d || Softmax Metric: %d || Loss: %.3f || Acc: %.3f || Test_Acc: %.3f ||  Time: %.3f '
            %(epoch, sm_metric, train_loss, train_acc, test_acc, time.time()-st))

    return pred_model, optimizer, scheduler, metric_history, centroids, wrong_idx

if __name__ == "__main__":
    # get settings
    args = get_args()

    # set directory
    path = 'models/{args.DATASET}'
    if args.REGULAR:
        path += '_reg'
    else:
        path += '_twin'
        if args.FREEZE:
            path += '_freeze'
    path += f'_{args.NET}'
    path += f'_{args.TRAIN_BATCH_SIZE}'
    path += f'_{args.LATENT_DIM}'
    path += f'_{args.LEARNING_RATE}'
    if not os.path.exists(path):
        os.makedirs(path)

    np.savez(path+f'/train_settings.npz', **args.__dict__)

    max_test_acc = []
    for t in range(args.NUM_REPEATS):
        # get dataset
        print('Loading dataset')
        dataset = Dataset(args.DATASET)
        args.CLASSES = dataset.n_classes
        dataset.set_train_dataloader(args.TRAIN_BATCH_SIZE, t)
        dataset.set_test_dataloader(args.TEST_BATCH_SIZE)
        dataset.set_centroid_dataloader(args.TRAIN_BATCH_SIZE, t)

        # get model
        print('Loading model')
        nets = {'ResNet50': [ResNet50(), 2048, 200], 'ResNet34': [ResNet34(), 512, 200], 'ResNet18': [ResNet18(), 512, 200], 'ConvNet': [ConvNet(), 512, 75], 'FCNet': [FCNet(), 256, 20]}
        cnn_model = nets[args.NET][0]
        dist_model = DistNet(args.LATENT_DIM, args.CLASSES, args.INIT_WEIGHT, args.REGULAR)
        pred_model = PredictionNet(cnn_model, dist_model, nets[args.NET][1], args.LATENT_DIM, args.CLASSES, args.REGULAR)
        pred_model = pred_model.to(device)

        if args.FREEZE:
            # freeze W matrix
            for param in pred_model.distnet.parameters():
                param.requires_grad = False

        # set optimizer and scheduler
        optimizer = torch.optim.SGD(pred_model.parameters(), lr=args.LEARNING_RATE, momentum=0.9, weight_decay=5e-4)
        if args.NET == 'ConvNet': 
            milestones = [25, 50]
            gamma = 0.1
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
        elif args.NET == 'ResNet18': 
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
        elif args.NET == 'ResNet34' or args.NET == 'ResNet50':
            try:
                joint_optimizer_specs = [{'params': pred_model.convnet.parameters(), 'lr': args.LEARNING_RATE, 'weight_decay': 5e-4},
                    {'params': pred_model.distnet.parameters(), 'lr': args.LEARNING_RATE*10}, {'params': pred_model.fc1.parameters(), 'lr': args.LEARNING_RATE*10}, {'params': pred_model.bn1.parameters(), 'lr': args.LEARNING_RATE*10}]
            except:
                joint_optimizer_specs = [{'params': pred_model.convnet.parameters(), 'lr': args.LEARNING_RATE, 'weight_decay': 5e-4},
                    {'params': pred_model.fc1.parameters(), 'lr': args.LEARNING_RATE*10}]
            optimizer = torch.optim.Adam(joint_optimizer_specs)
            scheduler = None
        else:
            scheduler = None

        # train model
        print('Training model')
        best_acc = 0.0 
        metric_history = {'train_acc': [], 'test_acc': []}
        for epoch in range(nets[args.NET][2]):
            dataset.reset_centroid_dataloader_iter()
            pred_model, optimizer, scheduler, metric_history, centroids, wrong_idx = train_step(args, epoch, dataset, pred_model, optimizer, scheduler, metric_history)
            current_acc = metric_history['test_acc'][-1]
            if current_acc > best_acc:
                save_model(epoch, pred_model, optimizer, scheduler, path+f'/model_{t}.pt')
                np.save(path+f"/wrong_indices_{t}.npy", wrong_idx)
                if centroids is not None:
                    np.save(path+f'/centroids_{t}.npy', centroids.cpu().detach().numpy())
                best_acc = current_acc
        
        # save metrics
        np.savez(path+f'/train_metrics_{t}.npz', **metric_history)
        max_test_acc.append(best_acc)
    np.save(path+f'/test_accuracies.npy', max_test_acc)