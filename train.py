import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import datasets, models, transforms
import torchvision.models as models
from torch.utils import data

from skorch import NeuralNetClassifier
from skorch.helper import predefined_split
from skorch.callbacks import LRScheduler, Checkpoint

import os
import argparse

from data.ucm_dataset import UCMDataSet

torch.manual_seed(360);
IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)
INPUT_SIZE = '321, 321'
TRAIN_DATA_DIRECTORY = '/home/amth_dg777/project/Satellite_Images'
TRAIN_DATA_LIST_PATH = '/home/amth_dg777/project/Satellite_Images/ImageSets/train.txt' # TODO: MAKE NEW TEXT FILE
TEST_DATA_DIRECTORY = '/home/amth_dg777/project/Satellite_Images'
TEST_DATA_LIST_PATH = '/home/amth_dg777/project/Satellite_Images/ImageSets/test.txt' # TODO: MAKE NEW TEXT FILE

#### Argument Parser
def get_arguments():
    parser = argparse.ArgumentParser(description="Arguments")
    parser.add_argument("--learning-rate", type=float, default=0.001,
                        help="Learning Rate")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Batch Size")
    parser.add_argument("--num-epochs", type=int, default=100,
                        help="Number of Epochs")
    parser.add_argument("--momentum", type=float, default=0.9,
                        help="Momentum")
    parser.add_argument("--device", type=str, default="cuda",
                        help="cuda/cpu")
    parser.add_argument("--random-scale", action="store_true",
                        help="Whether to randomly scale the inputs during the training.")
    parser.add_argument("--random-mirror", action="store_true",
                        help="Whether to randomly mirror the inputs during the training.")
    parser.add_argument("--input-size", type=str, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of images.")
    parser.add_argument("--train-data-dir", type=str, default=TRAIN_DATA_DIRECTORY,
                        help="Path to the directory containing the PASCAL VOC dataset.")
    parser.add_argument("--train-data-list", type=str, default=TRAIN_DATA_LIST_PATH,
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--test-data-dir", type=str, default=TEST_DATA_DIRECTORY,
                        help="Path to the directory containing the PASCAL VOC dataset.")
    parser.add_argument("--test-data-list", type=str, default=TEST_DATA_LIST_PATH,
                        help="Path to the file listing the images in the dataset.")
    return parser.parse_args()

args = get_arguments()

#### Dataloader Object

h, w = map(int, args.input_size.split(','))
input_size = (h, w)

train_dataset = UCMDataSet(args.train_data_dir, args.train_data_list, crop_size=input_size,
                scale=args.random_scale, mirror=args.random_mirror, mean=IMG_MEAN)
trainloader = data.DataLoader(train_dataset,
                batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)

test_dataset = UCMDataSet(args.test_data_dir, args.test_data_list, crop_size=input_size,
                scale=args.random_scale, mirror=args.random_mirror, mean=IMG_MEAN)
testloader = data.DataLoader(test_dataset,
                batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)

#### Model
vgg16 = models.vgg16(pretrained=True, progress=True)
print(vgg16)

#### Model Callbacks
lrscheduler = LRScheduler(policy='StepLR', step_size=7, gamma=0.1)

checkpoint = Checkpoint(f_params='best_model.pt', monitor='valid_acc_best')


#### Neural Net Classifier
net = NeuralNetClassifier(
    vgg16, 
    criterion=nn.CrossEntropyLoss,
    lr=args.learning_rate,
    batch_size=args.batch_size,
    max_epochs=args.num_epochs,
    module__output_features=2,
    optimizer=optim.SGD,
    optimizer__momentum=args.momentum,
    iterator_train=trainloader,
    iterator_train__shuffle=True,
    iterator_train__num_workers=4,
    iterator_valid__shuffle=True,
    iterator_valid__num_workers=4,
    callbacks=[lrscheduler, checkpoint],
    device=args.device # comment to train on cpu
)

#### Train the network

#net.fit(train_ds, y)
