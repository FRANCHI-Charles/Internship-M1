import torch
import torch.nn as nn
import numpy as np
from pandas import Series
from DFA2SRN import dfa2srn
import RNN_autmata.utils as lib

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

DTYPE = torch.float
BATCH_SIZE = 32
NAMES_PATH = "./names.txt"
DATA_PATH = "./data/Small/"


names, ext = lib.lang_names(NAMES_PATH)


#### the training function 

def train_stats(net,target, stats_data, optimizer, lossfunction, mini_batch, train_loader, num_epochs, lr, dtype, device):
    net.to(dtype=dtype,device=device)
    
    for _ in range(num_epochs):
        ## creating a batch
        
            
        # move data to proper dtype and device
        # images, labels = train_loader[i]
        optimizer.zero_grad()
        batch = batch.to(device=device)
        labels = labels.to(dtype=torch.long, device=device)
        
        # Forward pass
        
        outputs = net(batch).squeeze()
        loss = lossfunction(outputs, labels)
        # loss = criterion(outputs, labels)# + reg_param*(net.get_beta()-torch.sqrt(1-1/Norm)) #reg_param*(torch.sqrt(1-1/Norm))#
        # Backward and optimize
        loss.backward()
        optimizer.step()

        (norm_2,norm_inf,target_dist) = lib.stats(net,target,lr)
        
        stats_data.push('plot_loss',loss.detach().item())
        stats_data.push('plot_norm',norm_inf)
        stats_data.push('plot_norm2',norm_2)
        stats_data.push('plot_dist',target_dist)
    print("The training is done")


def test_stat(net,test_loader, lossfunc, stats_data, dtype):
    net.to(dtype=dtype,device=device)

    ## creating a batch
    batch_index = range(len(test_loader[0]))
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
    loss = lossfunc(outputs, labels)
    loss[loss < 10**(-4)] = 0
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
        model = lib.customSRN(hidden_dim = shapes[j][0]*shapes[j][1],input_dim=input_length, output_dim=1,seq_length=seq_leng,device=device,dtyp=32)
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

