import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import datasets, models, transforms
import torchvision.models as models

from skorch import NeuralNetClassifier
from skorch.helper import predefined_split
from skorch.callbacks import LRScheduler, Checkpoint

import os
import argparse

torch.manual_seed(360);

#### Argument Parser
parser = argparse.ArgumentParser(description="")
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

parser.parse_args()
#### Dataloader Object

#train_ds = 
#val_ds = 

#### Model
vgg16 = models.vgg16(pretrained=True, progress=True)
print(vgg16)

#### Model Callbacks
lrscheduler = LRScheduler(policy='StepLR', step_size=7, gamma=0.1)

checkpoint = Checkpoint(f_params='best_model.pt', monitor='valid_acc_best')


#### Neural Net Classifier
net = NeuralNetClassifier(
    PretrainedModel, 
    criterion=nn.CrossEntropyLoss,
    lr=args.learning_rate,
    batch_size=args.batch_size,
    max_epochs=args.num_epochs,
    module__output_features=2,
    optimizer=optim.SGD,
    optimizer__momentum=args.momentum,
    iterator_train__shuffle=True,
    iterator_train__num_workers=4,
    iterator_valid__shuffle=True,
    iterator_valid__num_workers=4,
    train_split=predefined_split(val_ds),
    callbacks=[lrscheduler, checkpoint],
    device=args.device # comment to train on cpu
)

#### Train the network

net.fit(train_ds, y)