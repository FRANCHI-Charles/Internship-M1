import gzip
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import model as md
import dataloader as dl
import pickle
from utils import lang_names, Loss_for_saturation
import io


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

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)


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




def trainer(model,train_loader,val_loader,criterion,optimizer,num_epochs,reg1):
    # Train the model
    res = 100
    n_total_steps = len(train_loader)
    for epoch in range(num_epochs):
        ### training step
        for i, (images, labels) in enumerate(train_loader):  
            # move data to proper dtype and device
            images = images.to(device)
            labels = labels.to(dtype=torch.long, device=device)
            # Forward pass
            outputs = model(images)
            W = model.get_matrixW()
            Norm = torch.linalg.norm(W,ord=2)
            loss = criterion(outputs, labels) + reg1*(torch.sqrt(1-1/Norm)-model.get_beta())
            # Backward and optimize

            fct_zero(model)
            loss.backward()
            fct(model, 0.001)

            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()

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
            print(f'Accuracy of the network on the validation words: {acc} %, beta gap: {torch.sqrt(1-1/Norm)-model.get_beta()}')
    return res

def reggy(tens):
    #return 0.001*(torch.sign(tens))*(((tens+1)**(1.38))/torch.log(tens+1) - 1.3)
    return (torch.sign(tens))*(1/((tens)**(0.02)))

def fct(model, lr):
    for param in model.parameters():
        param.grad = torch.clamp(param.grad,-0.025,0.025)
        # print(param.data)
        # param.data -= lr*(reggy(param.grad))
        param.data -= lr*param.grad

def fct_zero(model):
    for param in model.parameters():
        param.grad = None

def trainer_pac_prior(model,train_loader,val_loader,criterion,optimizer,num_epochs,reg1,reg2,M):
    # Train the model
    # myloss = Loss_for_saturation.apply
    res = 100
    n_total_steps = len(train_loader)
    for epoch in range(num_epochs):
        ### training step
        for i, (images, labels) in enumerate(train_loader):  
            # move data to proper dtype and device
            images = images.to(device)
            labels = labels.to(dtype=torch.long, device=device)
            # Forward pass
            outputs = model(images)
            W = model.get_matrixW()
            m = torch.nn.utils.parameters_to_vector(model.parameters())
            Norm = torch.linalg.norm(W,ord=2)
            # print(Norm)
            loss = 0.1*criterion(outputs, labels) + reg1*torch.exp(torch.sqrt(1-1/Norm)-model.get_beta()) + reg2*(torch.linalg.norm(m-M)) 
            # loss = myloss(outputs, labels)
            # Backward and optimize
            # optimizer.zero_grad()
            fct_zero(model)
            loss.backward()
            fct(model, 0.1)
            # optimizer.step()
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
            print(f'Accuracy of the network on the validation words: {acc} %, beta gap: {torch.sqrt(1-1/Norm)-model.get_beta()}, proximity with the prior: {torch.linalg.norm(m-M)}')
    return res


def Fbeta_tester(model,dataloader,n):
    j=0
    W = model.get_matrixW()
    Norm = torch.linalg.norm(W,ord=2)
    ref  = [torch.sqrt(1-1/Norm).detach().cpu().numpy()]
    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(dtype=torch.long, device=device)
        outputs = model(images)
        dic = model.get_Fbeta()
        ploting = dic['value'][-2]
        plot_po = dic['position'][-2]

        fig, axs = plt.subplots(1,2,figsize = (9,3))
        axs[0].plot(ploting, color='blue')
        axs[0].plot(ref*len(ploting), color='green')
        axs[1].plot(plot_po,color='red')
        plt.show()

        if j>n:
            break
        j+=1

name = names[-1]
print("In processe "+ name)
train_loader, val_loader, test_loader, input_length, seq_leng = data_creator(name)


model = md.SRNCell_pad(hidden_size=256,input_size=input_length,output_size=2,seq_length=seq_leng,num_layers=1,device = device, dtyp=64)
params = model.state_dict()
for key in params:
    if key!= 'fc.weight' and key!= 'fc.bias':
        print(key)
        params[key] *=1
        params[key][params[key]<=0] *= -1
    # print(params[key])


# for param in model.parameters():
#     param.data[param.data<=0] *= -1
#     print(param.data)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), # or any optimizer you prefer 
                        lr= 0.001, # 0.001 is used if no lr is specified
                        momentum= 0.9)

# test_img = []
# j = 0
# while j<1:
#     for image, label in val_loader:
#         test_img.append(image)
#     j+=1


# W = model.get_matrixW()
# Norm = torch.linalg.norm(W,ord=2)
# ref  = [torch.sqrt(1-1/Norm).detach().cpu().numpy()]
# print(ref)
# images = test_img[0].to(device)
# outputs = model(images)
# dic = model.get_Fbeta()
# fig, axs = plt.subplots(1,2)
# for k in range(len(dic['value'])):
#     ploting = dic['value'][5]
#     plot_po = dic['position'][5]
#     axs[0].plot(ploting, color='blue')
#     axs[0].plot(ref*len(ploting), color='green')
#     axs[1].plot(plot_po,color='red')
# plt.show()


trainer(model,train_loader,val_loader,criterion,optimizer,num_epochs=50,reg1=0.08)
init_parameters = model.state_dict()





# Fbeta_tester(model,val_loader,n=5)
# with open('model_clamped_gradient.pickle', 'wb') as handle:
#     pickle.dump(init_parameters, handle, protocol=pickle.HIGHEST_PROTOCOL)


# with open('model.pickle', 'rb') as handle:
#     init_parameters = CPU_Unpickler(handle).load()


# model.load_state_dict(init_parameters)

# M = torch.nn.utils.parameters_to_vector(model.parameters())
# print("##"*20)
# trainer_pac_prior(model,train_loader,val_loader,criterion,optimizer,num_epochs=10,reg1=3,reg2=0.01,M=M)
# Fbeta_tester(model,val_loader,n=5)
# print("~~"*20)
# model.load_state_dict(init_parameters)
# m = torch.nn.utils.parameters_to_vector(model.parameters())
# print(torch.linalg.norm(m-M))
# print("##"*20)
# trainer_pac_prior(model,train_loader,val_loader,criterion,optimizer,num_epochs=10,reg1=0.05,reg2=0.05,M=M)
# print("~~"*20)
# m = torch.nn.utils.parameters_to_vector(model.parameters())
# print(torch.linalg.norm(m-M))
# model.load_state_dict(init_parameters)
# print("##"*20)
# trainer_pac_prior(model,train_loader,val_loader,criterion,optimizer,num_epochs=10,reg1=0.1,reg2=0.05,M=M)
# print("~~"*20)
