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
from ax.plot.trace import optimization_er_one_hot_etrace_single_method
from ax.service.managed_loop import optimize
from ax.utils.notebook.plotting import render
from ax.utils.tutorials.cnn_utils import train, evaluate


dtype = torch.float
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

batch_size = 32
names_path = "./names.txt"
# names_path = "/home/volodimir/Bureau/ForLang/names.txt"

names, ext = lang_names(names_path)

path = "./data/Small/"
# path = "/home/volodimir/Bureau/ForLang/data/Small/"


### data related functions 


def data_creator(name):
    data,_ = dl.data_proces(path,name+ext[-1]+'.txt')
    l1 = dl.get_max_length(data)
    input_length = len(dl.alphabet_extractor(data))
    data,_ = dl.data_proces(path,name+ext[0]+'.txt')
    l2 = dl.get_max_length(data)
    data,_ = dl.data_proces(path,name+ext[2]+'.txt')
    l3 = dl.get_max_length(data)
    seq_leng = np.max([l1,l2,l3])

    train_data = dl.Dataset(path,name+ext[-1]+'.txt',length = seq_leng)
    val_data   = dl.Dataset(path,name+ext[0] +'.txt',length = seq_leng)
    test_data  = dl.Dataset(path,name+ext[2] +'.txt',length = seq_leng)


    train_loader = torch.utils.data.DataLoader(dataset=train_data, 
                                            batch_size=batch_size, 
                                            shuffle=True)
    val_loader = torch.utils.data.DataLoader(dataset=val_data, 
                                            batch_size=batch_size, 
                                            shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_data, 
                                            batch_size=batch_size, 
                                            shuffle=True)
    return (train_loader, val_loader, test_loader, input_length, seq_leng)

def data_proces(path,name):
    data = []
    labels = []
    with open(path+name) as file:
        length = len(file.readlines())
    with open(path+name) as file:
        for j in range(length):
            line = file.readline()
            labeled_line = line.split()
            data.append(labeled_line[0])
            labels.append(labeled_line[1])
    return (data,labels)

def get_max_length(data):
    return np.max([len(x) for x in data])

def alphabet_extractor(data):
    alphabet = []
    for elem in data:
        for lettre in elem:
            alphabet.append(lettre)
    alphabet = set(alphabet)
    return sorted(list(alphabet))

def to_one_hot(self, x):
    x_one_hot = []
    for letter in x:
        x_one_hot.append(lettre_one_hot_encoder(self.one_hot,self.alphabet,letter))
    for _ in range(self.padding_length - len(x)):
        x_one_hot.append(np.zeros((len(self.alphabet),)))
    x_one_hot.append(len(x)*np.ones((len(self.alphabet),)))
    x_one_hot = np.array(x_one_hot)
    return torch.tensor(x_one_hot, dtype=torch.float32)

def lettre_one_hot_encoder(one_hot_list,alphabet,lettre):
    indx = integer_embeder(alphabet,lettre)
    return one_hot_list[indx]

def integer_embeder(alphabet,lettre):
    return alphabet.index(lettre)

def labeled_data_set(path, name, length):
    data = data_proces(path,name)
    alphabet = alphabet_extractor(data[0])
    target_alphabet = ['TRUE','FALSE']
    X = []
    y = []
    one_hot = []
    for j in range(len(alphabet)):
        one_hot.append(np.eye(len(alphabet))[:,j])
    
    if length == None:
        padding_length = get_max_length(data[0])
    else:
        padding_length = length
    
    for j in range(len(data[0])):
        x_one_hot = []
        x = data[0][j] 
        for letter in x:
            x_one_hot.append(lettre_one_hot_encoder(one_hot,alphabet,letter))
        for _ in range(padding_length - len(x)):
            x_one_hot.append(np.zeros((len(alphabet),)))
        x_one_hot.append(len(x)*np.ones((len(alphabet),)))
        X.append(x_one_hot)
        label = 1*(data[1][j]=='TRUE') + 0*(data[1][j]=='FALSE')
        y.append(label)
    return (X,y)

def data_creator2(name,path,ext):
    data,_ = dl.data_proces(path,name+ext[-1]+'.txt')
    l1 = dl.get_max_length(data)
    input_length = len(dl.alphabet_extractor(data))
    data,_ = dl.data_proces(path,name+ext[0]+'.txt')
    l2 = dl.get_max_length(data)
    data,_ = dl.data_proces(path,name+ext[2]+'.txt')
    l3 = dl.get_max_length(data)
    seq_leng = np.max([l1,l2,l3])

    train_data = labeled_data_set(path,name+ext[-1]+'.txt',length = seq_leng)
    val_data   = labeled_data_set(path,name+ext[0]+'.txt' ,length = seq_leng)
    test_data  = labeled_data_set(path,name+ext[2]+'.txt' ,length = seq_leng)

    
    return (train_data, val_data, test_data, input_length, seq_leng)


### model training statisyics related functions 

def stats(net,target,lr):
    ### this function returns the L2 and the L_\infty norm of the gradient
    norm_2_acc = 0
    norm_inf_acc = []
    target_dist = 0
    k=0
    for parameters in net.parameters():
        gr = parameters.grad
        parameters = parameters.detach() 
        target_dist += ((torch.linalg.norm(parameters.flatten()-target[k].flatten())).item())**2
        parameters -= lr*(gr) # lr*(target[k]) + (1-lr)*(initial[k])
        parameters.requires_grad = True
        norm_2_acc += (torch.linalg.norm(gr.flatten(),ord=2).item())**2
        norm_inf_acc.append(torch.linalg.norm(gr.flatten(),ord=float('inf')).item())
        k+=1
    target_dist = target_dist**(0.5)
    norm_2 = lr*norm_2_acc**(0.5)
    norm_inf = max(norm_inf_acc)
    return (norm_2,norm_inf,target_dist)

def target_distance(net, target):
    target_dist = 0
    k=0
    for parameters in net.parameters():
        target_dist += ((torch.linalg.norm(parameters.flatten()-target[k].flatten())).item())**2
        k+=1
    target_dist = target_dist**(0.5)
    return target_dist

def rolling_avg(arr,window_size):
    
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


#### the training function 


def train_stats(net,target, stats_data, optimizer, mini_batch, train_loader, num_epochs, lr, dtype, device):
    net.to(dtype=dtype,device=device)

    
    #train
    count = 0
    
    for _ in range(num_epochs):
        ## creating a batch
        batch_index = random.sample(list(range(len(train_loader[0]))),mini_batch)
        batch, labels = [], []
        for elem in batch_index:
            batch.append(train_loader[0][elem])
            labels.append(train_loader[1][elem])
        batch  = torch.tensor(batch)
        labels = torch.tensor(labels)
            
        # move data to proper dtype and device
        # images, labels = train_loader[i]
        optimizer.zero_grad()
        batch = batch.to(device=device)
        labels = labels.to(dtype=torch.long, device=device)
        
        # Forward pass
        
        outputs = net(batch).squeeze()
        loss = (-1)*( labels*torch.log(outputs+10**(-7)) + (1-labels)*torch.log((1-outputs)+10**(-7)) ) + torch.log(torch.tensor([1+10**(-7)])).to(dtype=torch.long, device=device)  #criterion(outputs, label)
        # loss = criterion(outputs, labels)# + reg_param*(net.get_beta()-torch.sqrt(1-1/Norm)) #reg_param*(torch.sqrt(1-1/Norm))#
        loss_plt = torch.sum(loss)/(loss).shape[0]
        # Backward and optimize
        loss_plt.backward()
        (norm_2,norm_inf,target_dist) = stats(net,target,lr)
        
       
        A = not(count > 500)
        optimizer.step()
        stats_data.push('plot_loss',loss_plt.detach().item())
        stats_data.push('plot_norm',norm_inf)
        stats_data.push('plot_norm2',norm_2)
        stats_data.push('plot_dist',target_dist)
    print("The training is done")

def test_stat(net,test_loader,stats_data):
    net.to(dtype=dtype,device=device)

    ## creating a batch
    batch_index = list(range(len(test_loader[0])))
    batch, labels = [], []
    for elem in batch_index:
        batch.append(test_loader[0][elem])
        labels.append(test_loader[1][elem])
    batch  = torch.tensor(batch)
    labels = torch.tensor(labels)
        
  
    batch = batch.to(device=device)
    labels = labels.to(dtype=torch.long, device=device)
    
    # Forward pass
    
    outputs = net(batch).squeeze()
    loss = (-1)*( labels*torch.log(outputs+10**(-7)) + (1-labels)*torch.log((1-outputs)+10**(-7)) ) + torch.log(torch.tensor([1+10**(-7)])).to(dtype=torch.long, device=device)  #criterion(outputs, label)
    loss[loss<10**(-4)] = 0
    average = torch.count_nonzero(loss)/len(test_loader[0])

    stats_data.push('average_loss',average.detach().item())
     
    print("The testing is done")



shapes = [(7,4),(9,4),(16,4),(2,4),(8,16),(14,16),(4,16),(7,64),(4,64)]


data_dict = {}
test_dict = {}
for j in range(9):

        name = names[j]
        print("In processe "+ name)
        train_loader, val_loader, test_loader, input_length, seq_leng = data_creator2(name,path,ext)
        print(len(train_loader[0]))
        print('The data is done')

        ### defining the target parameters 
        # machine_path = "/home/volodimir/Bureau/ForLang/data/machines/"
        machine_path ='/home/miv09159/ForLang/data/machines/'
        machine_name = name[:-1]+'.att'
        T_m, D_m = machine_proces(machine_path, machine_name)
        target = dfa2srn(T_m, D_m)


        print('Starting the training')
        model = md.customSRN(hidden_dim = shapes[j][0]*shapes[j][1],input_dim=input_length, output_dim=1,seq_length=seq_leng,device=device,dtyp=32)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), # or any optimizer you prefer 
                                lr= 0.01, # 0.001 is used if no lr is specified
                                momentum= 0.87)
       
        ls_names = ['plot_loss','plot_norm','plot_norm2','plot_dist']
        stats_data = md.STATS(ls_names)
        test_data  = md.STATS(['average_loss'])

        train_stats(model,target, stats_data, optimizer, 32, train_loader, 20000, 0.01, dtype, device)
        data_dict[name] = stats_data.get_data()
        test_stat(model,test_loader,test_data)
        test_dict[name] = test_data.get_data()

with open('data_20000.pkl','wb') as f:
    pickle.dump(data_dict, f)
    
with open('test_20000.pkl','wb') as f:
    pickle.dump(test_dict, f)

