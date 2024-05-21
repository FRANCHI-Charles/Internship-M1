import gzip
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import model as md
import dataloader as dl
import pickle
from utils import lang_names


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

batch_size = 32
names_path = "/home/miv09159/ForLang/names.txt"
names_path = "/home/volodimir/Bureau/ForLang/names.txt"

names, ext = lang_names(names_path)

path = "/home/miv09159/ForLang/data/Large/"
path = "/home/volodimir/Bureau/ForLang/data/Mid/"
# name = names[7]

# data,_ = dl.data_proces(path,name+ext[-1]+'.txt')
# l1 = dl.get_max_length(data)
# input_length = len(dl.alphabet_extractor(data))
# data,_ = dl.data_proces(path,name+ext[0]+'.txt')
# l2 = dl.get_max_length(data)
# data,_ = dl.data_proces(path,name+ext[2]+'.txt')
# l3 = dl.get_max_length(data)
# seq_leng = np.max([l1,l2,l3])

# train_data = dl.Dataset(path,name+ext[-1]+'.txt',length = seq_leng)
# val_data   = dl.Dataset(path,name+ext[0] +'.txt',length = seq_leng)
# test_data  = dl.Dataset(path,name+ext[2] +'.txt',length = seq_leng)

# ### test zone


# train_loader = torch.utils.data.DataLoader(dataset=train_data, 
#                                            batch_size=batch_size, 
#                                            shuffle=True)
# val_loader = torch.utils.data.DataLoader(dataset=val_data, 
#                                            batch_size=batch_size, 
#                                            shuffle=True)
# test_loader = torch.utils.data.DataLoader(dataset=test_data, 
#                                            batch_size=batch_size, 
#                                            shuffle=True)

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

    ### test zone


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



### Elements for Bayes Optimization  

def net_train(net, train_loader, parameters, dtype,  device):

    net.to(dtype=dtype,device=device)

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), # or any optimizer you prefer 
                        lr=parameters.get("lr", 0.001), # 0.001 is used if no lr is specified
                        momentum=parameters.get("momentum", 0.9)
    )

    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=int(parameters.get("step_size", 30)),
        gamma=parameters.get("gamma", 1.0),  # default is no learning rate decay
    )

    num_epochs = parameters.get("num_epochs", 3) # Play around with epoch number
    # Train Network
    reg_param  = parameters.get("reg_param",0)
    M = torch.eye(10)
    for _ in range(num_epochs):
        for _, (images, labels) in enumerate(train_loader):
            # move data to proper dtype and device
            # images, labels = train_loader[i]
            optimizer.zero_grad()
            images = images.to(device=device)
            labels = labels.to(dtype=torch.long, device=device)
            
            # Forward pass
            W = net.get_matrixW()
            Norm = torch.linalg.norm(W,ord=2)
            outputs = net(images)
            loss = criterion(outputs, labels) + reg_param*(net.get_beta()-torch.sqrt(1-1/Norm)) #reg_param*(torch.sqrt(1-1/Norm))#
            
            # Backward and optimize
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            # net.reset_beta()
    print("The training is done")
    return net

def init_net(parameters):
    model = md.customSRN(hidden_dim = parameters.get('hidden_size',128),input_dim=input_length, output_dim=2,seq_length=seq_leng,device=device,dtyp=32)
    return model 

def evaluate_saturate(
    net: nn.Module, data_loader: DataLoader, dtype: torch.dtype, device: torch.device
) -> float:
    """
    Compute classification accuracy on provided dataset.

    Args:
        net: trained model
        data_loader: DataLoader containing the evaluation set
        dtype: torch dtype
        device: torch device
    Returns:
        float: classification accuracy
    """
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in data_loader:
            # move data to proper dtype and device
            inputs = inputs.to(dtype=dtype, device=device)
            labels = labels.to(device=device)
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    res = (correct / total)
    W = net.get_matrixW()
    Norm = torch.linalg.norm(W,ord=2)
    reg  = (net.get_beta()-torch.sqrt(1-1/(Norm+10**(-4)))).item()
    # print("This net got the fo
    # llowing performance:" + str(res/(reg+10**(-4))))
    out = res + 1/(reg+10**(-4))
    return out*(out==out) - 100*(out!=out)

def train_evaluate(parameterization):

    # Get neural net
    untrained_net = init_net(parameterization) 
    
    # train
    trained_net = net_train(net=untrained_net,dtype=dtype, train_loader=train_loader, 
                            parameters=parameterization, device=device)
    
    # return the accuracy of the model as it was trained in this run
    return evaluate(
        net=trained_net,
        data_loader=test_loader,
        dtype = dtype,
        device=device,
    )




def trainer(model,train_loader,val_loader,criterion,optimizer,num_epochs,reg_param):
    # Train the model
    res = 100
    n_total_steps = len(train_loader)
    for epoch in range(num_epochs):
        ### training step
        for i, (images, labels) in enumerate(train_loader):  
        # origin shape: [N, 1, 28, 28]
        # resized: [N, 28, 28]
            # move data to proper dtype and device
            images = images.to(device)
            labels = labels.to(dtype=torch.long, device=device)
            # Forward pass
            outputs = model(images)
            W = model.get_matrixW()
            Norm = torch.linalg.norm(W,ord=2)
            loss = criterion(outputs, labels)# + torch.exp(reg_param*(model.get_beta()-torch.sqrt(1-1/Norm)))
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # scheduler.step()
            model.reset_beta()
        
        if (i+1) % 10 == 0:
            print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}, beta gap: {torch.sqrt(1-1/Norm)-model.get_beta()}')

        ### validation step
        with torch.no_grad():
            n_correct = 0
            n_samples = 0
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(dtype=torch.long, device=device)
                outputs = model(images)
                # max returns (value ,index)
                _, predicted = torch.max(outputs.data, 1)
                n_samples += labels.size(0)
                n_correct += (predicted == labels).sum().item()

            acc = 100.0 * n_correct / n_samples
            res = res*(acc>=res) + acc*(res>acc)
            print(f'Accuracy of the network on the 10000 validation words: {acc} %, beta gap: {torch.sqrt(1-1/Norm)-model.get_beta()}')
    return res

perf_dict = {}
for j in range(1,4):
    name = names[-j]
    print("In processe "+ name)
    perf_dict[name] = {}
    train_loader, val_loader, test_loader, input_length, seq_leng = data_creator(name)


    best_parameters, values, experiment, model = optimize(
        parameters=[
            {"name": "lr",          "type": "range", "bounds": [1e-6, 0.4], "log_scale": True},
            {"name": "hidden_size", "type": "range", "bounds": [16, 512],   'value_type':'int'},
            {"name": "momentum",    "type": "range", "bounds": [0.0, 1.0]},
            {"name": "max_epoch",   "type": "range", "bounds": [2, 30],     'value_type':'int'},
            {"name": "reg_param", "type": "range","bounds": [1e-4,1e-1], "log_scale": True},
            {"name": "epsilon", "type":"range", "bounds": [0.01, 0.9]}
            #{"name": "stepsize", "type": "range", "bounds": [20, 40]},         
        ],
    
        evaluation_function=train_evaluate,
        objective_name='accuracy/saturation_gap',
    )
    bp = best_parameters

    model = md.customSRN(hidden_dim = bp.get('hidden_size',128),input_dim=input_length, output_dim=2,seq_length=seq_leng,device=device,dtyp=64)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), # or any optimizer you prefer 
                            lr= bp['lr'], # 0.001 is used if no lr is specified
                            momentum= bp['momentum'])


    trainer(model,train_loader,val_loader,criterion,optimizer,bp['max_epoch'],bp['reg_param'])

    
    # model = net_train(model, train_loader, bp, dtype=torch.float32,  device=device)
    perf_dict[name]['test_acc'] = evaluate(model, test_loader,dtype=dtype, device=device)
    perf_dict[name]['train_acc'] = evaluate(model, train_loader,dtype=dtype, device=device)

    # W = model.get_matrixW()
    # Norm = torch.linalg.norm(W,ord=2)
    # # print(torch.sqrt(1-1/Norm))

    # beta = model.get_beta()
    # print(beta)
    perf_dict[name]['beta_gap'] = 0 #beta - torch.sqrt(1-1/Norm).item()
    
with open('perf_dict_saturated.pickle', 'wb') as handle:
    pickle.dump(perf_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)



# model = md.customSRN(hidden_dim = bp['hidden_size'],input_dim=input_length, output_dim=2,seq_length=seq_leng,device=device)



# criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(model.parameters(), # or any optimizer you prefer 
#                         lr= bp['lr'], # 0.001 is used if no lr is specified
#                         momentum= bp['momentum'])


# trainer(model,train_loader,val_loader,criterion,optimizer,bp['max_epoch'],bp['reg_param'])

    # print(best_parameters)
    # means, covariances = values
    # print(means)
    # print(covariances)

    # best_objectives = np.array([[trial.objective_mean*100 for trial in experiment.trials.values()]])


