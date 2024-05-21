import gzip
import torch
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import model as md
import dataloader as dl
import pickle
from utils import lang_names
from DFA2SRN import machine_proces, dfa2srn


from torchvision.datasets import MNIST
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
from torchvision.transforms import Compose, ToTensor, Resize, Normalize
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from ax.plot.contour import plot_contour
from ax.plot.trace import optimization_trace_single_method
from ax.service.managed_loop import optimize
from ax.utils.notebook.plotting import render
from ax.utils.tutorials.cnn_utils import train, evaluate


dtype = torch.float
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

batch_size = 32
names_path = "/home/miv09159/ForLang/names.txt"
names_path = "/home/volodimir/Bureau/ForLang/names.txt" ### il faut donner le chemin dans ta machine 

names, ext = lang_names(names_path)

path = "/home/miv09159/ForLang/data/Large/"
path = "/home/volodimir/Bureau/ForLang/data/Small/"  ### il faut donner le chemin dans ta machine 


def rolling_avg(arr,window_size):


    
    window_size = 3
    
    # Convert array of integers to pandas series
    numbers_series = pd.Series(arr)
    
    # Get the window of series of
    # observations till the current time
    windows = numbers_series.expanding()
    
    # Create a series of moving averages of each window
    moving_averages = windows.mean()
    
    # Convert pandas series back to list
    moving_averages_list = moving_averages.tolist()
    return np.array(moving_averages_list)
 
with open('data_20000.pkl', 'rb') as f:  ### cette partie ouvre le pickle produit par le code cluster_main.py
    data_dict = pickle.load(f)

fig, ax = plt.subplots(9,4,figsize=(22,36)) ### cette partie affichera les données récoltées

for j in range(9):
    name = names[j]
    (plot_loss,plot_norm,plot_norm2,plot_dist) = data_dict[name]
    l_min, l_max = np.min(plot_loss), np.max(plot_loss)
    i_min, i_max = np.min(plot_norm), np.max(plot_norm)
    n_min, n_max = np.min(plot_norm2), np.max(plot_norm2)

    ax[j][0].set_yscale("log")
    # ax[j][0].set_ylim(l_min-10*l_min,l_max+10*l_max)
    ax[j][0].plot(plot_loss)
    ax[j][0].plot(rolling_avg(plot_loss,3))
    ax[j][0].set_title("Loss")

    ax[j][1].set_yscale("log")
    # ax[j][1].set_ylim(i_min-10*i_min,i_max+10*i_max)
    ax[j][1].plot(plot_norm2)
    ax[j][1].plot(rolling_avg(plot_norm2,3))
    ax[j][1].set_title("$\|\mathbf{p}_{k+1}- \mathbf{p}_k\|_2$")

    ax[j][2].set_yscale("log")
    # ax[j][2].set_ylim(n_min-10*n_min,n_max+10*n_max)
    ax[j][2].plot(plot_norm)
    ax[j][2].plot(rolling_avg(plot_norm,3))
    ax[j][2].set_title("$\|Gradient\|_\infty$")

    # ax[j][3].set_yscale("log")
    # ax[j][2].set_ylim(n_min-10*n_min,n_max+10*n_max)
    ax[j][3].plot(plot_dist)
    ax[j][3].set_title("Distence to 'the' target")


plt.show()