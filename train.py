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

from modAL.models import ActiveLearner

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
    parser.add_argument("--batch-size", type=int, default=100,
                        help="Batch Size")
    parser.add_argument("--num-epochs", type=int, default=1,
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

class Vgg16Module(nn.Module):
    def __init__(self):
        super(Vgg16Module,self).__init__()
        self.net = vgg16
        self.final_layer = nn.Linear(1000,21)
        self.log_softmax=nn.LogSoftmax()      

    def forward(self,x):
        x1 = self.net(x)
        #print('Passed Thru VGG')
        y = self.final_layer(x1)
        y_pred=self.log_softmax(y)
        return y_pred

model = Vgg16Module()
#print(vgg16)

#### Model Callbacks
lrscheduler = LRScheduler(policy='StepLR', step_size=7, gamma=0.1)

checkpoint = Checkpoint(dirname = 'exp', f_params='best_model.pt', monitor='train_loss_best')


#### Neural Net Classifier
net = NeuralNetClassifier(
    module=model,
    criterion=nn.NLLLoss,
    lr=args.learning_rate,
    batch_size=args.batch_size,
    max_epochs=args.num_epochs,
    optimizer=optim.SGD,
    optimizer__momentum=args.momentum,
    train_split=None,
    #iterator_train__shuffle=True,
    #iterator_train__num_workers=1,
    #iterator_train__dataset=train_dataset,
    callbacks=[lrscheduler],
    device=args.device # comment to train on cpu
)

#### Train the network

X_train, y_train = next(iter(trainloader))
X_test, y_test = next(iter(testloader))
#net.fit(train_dataset,y)
#print(net.history)
#net.save_params(f_params='model.pkl', f_optimizer='opt.pkl', f_history='history.json')
#print(np.shape(X_train))


#### Split X and y into seed and pool
########### SMALL MEMORY ##############

X_train = X_train[:100]
y_train = y_train[:100]

# assemble initial data
n_initial = 20
initial_idx = np.random.choice(range(len(X_train)), size=n_initial, replace=False)
X_initial = X_train[initial_idx]
y_initial = y_train[initial_idx]
print(np.shape(X_initial), 'X_seed')
print(np.shape(y_initial), 'y_seed')

# generate the pool
# remove the initial data from the training dataset
X_pool = np.delete(X_train, initial_idx, axis=0)[:5000]
y_pool = np.delete(y_train, initial_idx, axis=0)[:5000]
print(np.shape(X_pool), 'X_pool')
print(np.shape(y_pool), 'y_pool')

#### Active Learner

# initialize ActiveLearner
learner = ActiveLearner(
    estimator=net,
    X_training=X_initial, y_training=y_initial,
)

# the active learning loop
n_queries = 2
for idx in range(n_queries):
    print('Query no. %d' % (idx + 1))
    query_idx, query_instance = learner.query(X_pool, n_instances=5)
    learner.teach(
        X=X_pool[query_idx], y=y_pool[query_idx], only_new=True,
    )
    # remove queried instance from pool
    X_pool = np.delete(X_pool, query_idx, axis=0)
    y_pool = np.delete(y_pool, query_idx, axis=0)

