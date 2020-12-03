#!/usr/bin/env python
# coding: utf-8

import sys
import os
import json
import time
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import cv2

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

gpu_bool = torch.cuda.is_available()

dataset_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(size=(224, 224)),
])

dataset_aug_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
])

class KaggleDataset(Dataset):

    def __init__(self, input_dir, split, trans=None):
        self.transform = trans
        self.split = split

        try:
            if self.split == 'train':
                self.directory = os.path.join(input_dir, 'train')
                self.labels = np.load(os.path.join(self.directory, 'labels.npy'))
            elif self.split == 'valid':
                self.directory = os.path.join(input_dir, 'valid')
                self.labels = np.load(os.path.join(self.directory, 'labels.npy'))
            elif self.split == 'test':
                self.directory = os.path.join(input_dir, 'test')
                self.labels = np.load(os.path.join(self.directory, 'labels.npy'))

            self.labels = self.labels.squeeze()
        except ValueError:
            print('split should be either \'train\', \'valid\' or \'test\'')

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        img_name = f'{idx}_input.jpg'
        img = cv2.imread(os.path.join(self.directory, img_name))
        if self.transform:
            img = self.transform(img)

        return img, label


def evaluate_model(
        loader, criterion=nn.CrossEntropyLoss(), model=None,
        verbose=True, name=""):

    correct = 0
    total = 0
    loss_sum = 0

    for x, y in loader:
        if gpu_bool:
            x, y = x.cuda(), y.cuda()

        outputs = model(x).detach()
        _, predicted = torch.max(outputs, 1)

        total += predicted.size(0)
        correct += (predicted.float() == y.float()).cpu().sum().data.numpy().item()
        loss_sum += criterion(outputs, y).cpu().data.numpy().item()

    accuracy = correct / total
    avg_loss = loss_sum / total

    if verbose:
        print('%s accuracy: %f %%' % (name, 100 * accuracy))
        print('%s loss: %f' % (name, avg_loss))

    return accuracy, avg_loss


def train_model(
        model, optimizer, train_l, valid_l, verbose=True,
        criterion=nn.CrossEntropyLoss(), num_epochs=30):

    valid_loss = []
    train_loss = []
    best_loss = float('inf')
    checkpoint = deepcopy(model)

    if gpu_bool:
        model = model.cuda()

    if verbose:
        print("Starting Training...")
        sys.stdout.flush()

    # training loop
    for epoch in range(num_epochs):

        time1 = time.time() #timekeeping
        curr_loss = 0

        # train/gradient descent
        for x, y in train_l:
            if gpu_bool:
                x, y = x.cuda(), y.cuda()

            optimizer.zero_grad()
            outputs = model.forward(x)

            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            curr_loss += loss.detach().cpu().data.numpy().item()

            if gpu_bool:
                torch.cuda.empty_cache()

        # evaluate performance
        curr_loss /= (len(train_l) * 32)

        if verbose:
            print("Epoch", epoch + 1, ':')
            print("train loss:", curr_loss)

        _, v_loss = evaluate_model(valid_l, model=model, name="valid", verbose=verbose)
        train_loss.append(curr_loss)
        valid_loss.append(v_loss)

        # update best model/early stopping
        if v_loss < best_loss:
            if verbose:
                print("New best - loss of {0} compared to {1}".format(v_loss, best_loss))

            checkpoint.load_state_dict(model.state_dict())
            best_loss = v_loss

        time2 = time.time() #timekeeping
        if verbose:
            print('Elapsed time for epoch:', time2 - time1, 's')
            print('ETA of completion:', (time2 - time1)*(num_epochs - epoch - 1)/60, 'minutes')
            print()

        sys.stdout.flush()

    return train_loss, valid_loss, model, checkpoint


def fill_args():
    args = {}

    if '--help' in sys.argv:
        return None

    index = sys.argv.index('--model') + 1
    args['model'] = sys.argv[index]

    index = sys.argv.index('--model_dir') + 1
    args['model_dir'] = sys.argv[index] + '.' + args['model'] + '.model'

    index = sys.argv.index('--batch_size') + 1
    args['batch_size'] = int(sys.argv[index])

    index = sys.argv.index('--learning_rate') + 1
    args['learning_rate'] = float(sys.argv[index])

    index = sys.argv.index('--num_epochs') + 1
    args['num_epochs'] = int(sys.argv[index])

    index = sys.argv.index('--data_loc') + 1
    args['data_loc'] = sys.argv[index]

    try:
        index = sys.argv.index('--verbose') + 1
        args['verbose'] = True
    except ValueError:
        pass

    try:
        index = sys.argv.index('--data_aug') + 1
        args['data_aug'] = True
    except ValueError:
        pass

    try:
        index = sys.argv.index('--pretrained') + 1
        args['pretrained'] = True
    except ValueError:
        pass

    return args


def main():
    help_args = """
        --help :
        --model_dir : str
        --batch_size : int
        --learning_rate : float
        --num_epochs : int
        --data_aug : flag
        --data_loc : str
        --model : str
        --verbose : flag
        --pretrained : flag

        add later if neccesary
        --use_adam :
        --momentum :
    """

    args = fill_args()
    if args is None:
        print(help_args)
        return

    verbose = 'verbose' in args
    if verbose:
        print(torch.__version__)
        print(args)

    train_set = KaggleDataset(
        args['data_loc'],
        'train',
        dataset_aug_transforms if 'data_aug' in args else dataset_transforms)
    train_loader = DataLoader(dataset=train_set, batch_size=args['batch_size'], shuffle=True)

    valid_set = KaggleDataset(args['data_loc'], 'valid', dataset_transforms)
    valid_loader = DataLoader(dataset=valid_set, batch_size=args['batch_size'], shuffle=False)

    test_set = KaggleDataset(args['data_loc'], 'test', dataset_transforms)
    test_loader = DataLoader(dataset=test_set, batch_size=args['batch_size'], shuffle=False)

    net = None

    if args['model'] == 'alexnet':
        net = models.alexnet(pretrained='pretrained' in args)
        net.classifier[6] = nn.Linear(in_features=4096, out_features=7, bias=True)
    elif args['model'] == 'vgg':
        net = models.vgg11_bn(pretrained='pretrained' in args)
        net.classifier[6] = nn.Linear(in_features=4096, out_features=7, bias=True)
    elif args['model'] == 'resnet':
        net = models.resnet18(pretrained='pretrained' in args)
        net.fc = nn.Linear(in_features=512, out_features=7, bias=True)
    elif args['model'] == 'densenet':
        net = models.densenet121(pretrained='pretrained' in args)
        net.classifier = nn.Linear(in_features=1024, out_features=7, bias=True)
    else:
        raise ValueError("no such model named " + str(args['model']))

    if not os.path.exists(args['model_dir']):
        os.makedirs(args['model_dir'])

    if verbose:
        print("train size:", len(train_loader))
        print("valid size:", len(valid_loader))
        print("test size:", len(test_loader))

    sys.stdout.flush()
    opt = torch.optim.SGD(net.parameters(), lr=args['learning_rate'])
    t_loss, v_loss, net, best_net = train_model(
        model=net, optimizer=opt,
        train_l=train_loader, valid_l=valid_loader,
        num_epochs=args['num_epochs'], verbose=verbose)

    if verbose:
        _, f_loss = evaluate_model(test_loader, model=best_net.cuda(), name="test", verbose=verbose)

        plt.figure(figsize=(7, 5))
        plt.plot(list(range(args['num_epochs'])), t_loss, label='Training Loss')
        plt.plot(list(range(args['num_epochs'])), v_loss, label='Validation Loss')
        plt.plot(list(range(args['num_epochs'])), [f_loss]*args['num_epochs'], '--', label='Test Loss')

        plt.title('Loss Over Training Epochs')
        plt.ylabel('Cross Entropy Loss')
        plt.xlabel('Epoch Number')

        plt.xticks(range(0, args['num_epochs']+1, 5))
        plt.yticks(np.linspace(0, 0.3, 6))
        plt.grid(True)

        plt.legend()

        np.save(os.path.join(args['model_dir'], 'train_loss.npy'), t_loss)
        np.save(os.path.join(args['model_dir'], 'valid_loss.npy'), v_loss)
        plt.savefig(os.path.join(args['model_dir'], 'training_plot.jpg'))

    torch.save(net.state_dict(), os.path.join(args['model_dir'], 'final_model'))
    torch.save(best_net.state_dict(), os.path.join(args['model_dir'], 'best_model'))

    with open(os.path.join(args['model_dir'], 'settings.json'), 'w') as f:
        json.dump(args, f, indent=2)


if __name__ == "__main__":
    main()
