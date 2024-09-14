from torch_geometric.data import Data, Batch, DataLoader
from torchvision.datasets import MNIST, CIFAR10, FashionMNIST
import torch

import matplotlib.pyplot as plt
import numpy as np
import time
from tqdm.auto import tqdm
import os
import multiprocessing
from itertools import repeat
# add directory above current directory to path
import sys
sys.path.insert(0, '../../graph_transformer')


from superpixel_generation import convert_numpy_img_to_superpixel_graph
from multiprocessing_istarmap import multiprocessing_istarmap
# Parameters
DATASET_NAME = "MNIST"
ADD_POSITION_TO_FEATURES = True
DESIRED_NODES = 75

## Load img dataset
print("Reading dataset...")

if DATASET_NAME == "MNIST":
    trainset = MNIST("../data", download=True, train=True)
    testset = MNIST("../data", download=True, train=False)

elif DATASET_NAME == "CIFAR10":
    trainset = CIFAR10("../data/CIFAR10", download=True, train=True)
    testset = CIFAR10("../data/CIFAR10", download=True, train=False)

elif DATASET_NAME == "FashionMNIST":
    trainset = FashionMNIST("../data", download=True, train=True)
    testset = FashionMNIST("../data", download=True, train=False)

else:
    print("Incorrect dataset name!")

print("Done.")

print(len(trainset))
print(len(testset))
num_pixels = np.prod(trainset.data.shape[1:3])
print("Img shape:", trainset.data.shape[1:])
print("Num of pixels per img:", num_pixels)
assert DESIRED_NODES < num_pixels, ('the number of superpixels cannot exceed the total number of pixels')
assert DESIRED_NODES > 1, ('the number of superpixels cannot be too small')

## Concatenate dataset
train_images = trainset.data
test_images = testset.data
print(train_images.shape)
print(test_images.shape)

images = np.concatenate((train_images, test_images))
if DATASET_NAME == "MNIST" or DATASET_NAME == "FashionMNIST":
    images = np.reshape(images, (len(images), 28, 28, 1))
print(images.shape)

train_labels = trainset.targets
test_labels = testset.targets
labels = np.concatenate((train_labels, test_labels))
print(labels.shape)

## Show one img
test_img = images[7]
plt.imshow(test_img, cmap='gray')
plt.show()

## Test conversion on one img

x, edge_index, pos = convert_numpy_img_to_superpixel_graph(
    test_img,
    desired_nodes = DESIRED_NODES,
    # add_position_to_features=ADD_POSITION_TO_FEATURES
)
print(type(x))
print(x.shape)
print(edge_index.shape)
print(pos.shape)
# print(x)
# x = x - np.mean(x) / np.std(x)
# print(x)


## Convert img dataset to superpixel graph dataset


# apply patch to enable progress bar with multiprocessing, requires python 3.8+
# see https://stackoverflow.com/questions/57354700/starmap-combined-with-tqdm/57364423#57364423


multiprocessing.pool.Pool.istarmap = multiprocessing_istarmap

# method for img -> superpixel_graph conversion


# tqdm for progress bar


print("Processing images into graphs...")
ptime = time.time()

x_list = []
edge_index_list = []
pos_list = []
slices = {
    "x": [0],
    "edge_index": [0],
    "pos": [0],
    "y": [0],
}

NUM_WORKERS = 8  # don't use too much or it will crash
with multiprocessing.Pool(NUM_WORKERS) as pool:
    args = list(zip(images, repeat(DESIRED_NODES)))
    for graph in tqdm(pool.istarmap(convert_numpy_img_to_superpixel_graph, args), total=len(args)):
        x, edge_index, pos = graph

        x = torch.as_tensor(x, dtype=torch.float32)
        edge_index = torch.as_tensor(edge_index, dtype=torch.long)
        pos = torch.as_tensor(pos, dtype=torch.float32)

        x_list.append(x)
        edge_index_list.append(edge_index)
        pos_list.append(pos)

        slices["x"].append(slices["x"][-1] + len(x))
        slices["edge_index"].append(slices["edge_index"][-1] + len(edge_index))
        slices["pos"].append(slices["pos"][-1] + len(pos))
        slices["y"].append(slices["y"][-1] + 1)

x_tensor = torch.cat(x_list, dim=0)
edge_index_tensor = torch.cat(edge_index_list, dim=0).T
pos_tensor = torch.cat(pos_list, dim=0)
y_tensor = torch.as_tensor(labels, dtype=torch.long)

slices["x"] = torch.as_tensor(slices["x"], dtype=torch.long)
slices["edge_index"] = torch.as_tensor(slices["edge_index"], dtype=torch.long)
slices["pos"] = torch.as_tensor(slices["pos"], dtype=torch.long)
slices["y"] = torch.as_tensor(slices["y"], dtype=torch.long)

del x_list
del edge_index_list
del pos_list

ptime = time.time() - ptime
print(f"Took {ptime}s.")

print(x_tensor.shape)
print(edge_index_tensor.shape)
print(pos_tensor.shape)
print(y_tensor.shape)

data = Data(x=x_tensor, edge_index=edge_index_tensor, pos=pos_tensor, y=y_tensor)
print(data)

## Save dataset in PyTorch Geometric format


path = "../data/" + DATASET_NAME + "_sp_" + str(DESIRED_NODES) + "/" + "processed" + "/"
if not os.path.exists(path):
    os.makedirs(path)

torch.save((data, slices), path + "data.pt")
