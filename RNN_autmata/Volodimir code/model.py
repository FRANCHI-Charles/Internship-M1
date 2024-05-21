import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from torch.optim import Adam

class STATS():
    def __init__(self,lis_stats):
        super(STATS,self).__init__()
        ### this class is designed to store any kind of statistics typed as lists. It takes as input a list of strings
        self.names = lis_stats
        self.dict  = {} 
        for elem in lis_stats:
            self.dict[elem] = []
    
        
    def push(self,name,element):
        self.dict[name].append(element)

    def get_data(self):
        ls = []
        for elem in self.names:
            ls.append(self.dict[elem])
        return ls
        

class SRNCell_pad(nn.Module):
    def __init__(self, hidden_size,input_size,output_size, seq_length,num_layers=1,device='cpu',dtyp=32):
        super().__init__()
        ### network constants 
        self.dtyp        = dtyp
        self.hidden_size = hidden_size
        self.seq_len     = seq_length
        self.device      = device
        self.num_layers  = num_layers
        self.output_size = output_size
        self.beta        = 1
        ### network architecture
        self.h      = nn.Parameter(torch.ones(self.hidden_size),requires_grad = True).to(dtype=torch.float64, device=self.device)
        self.rnn    = nn.RNNCell(input_size, hidden_size, bias=True, nonlinearity='tanh', device=self.device, dtype=torch.float64)
        self.fc     = nn.Linear(hidden_size, output_size).to(dtype=torch.float64, device=self.device)
        
    def forward(self,x):
        pad_len = x[:,-1,:]
        if self.dtyp == 32:
            dtyp = torch.float32 
        else:
            dtyp = torch.float64
        x    = x[:,:-1,:].to(dtype = dtyp,device=self.device)  
        # h    = torch.zeros(x.size(0), self.hidden_size).to(dtype=dtyp,device=self.device) # bqatch x hiddendi
        outs = torch.empty(x.size(0),self.output_size,self.seq_len-1).to(dtype = dtyp,device=self.device) 
        mask = torch.empty(x.size(0),self.seq_len-1).to(dtype = dtyp,device=self.device)
        x = torch.transpose(x,2,1)
        y = torch.transpose(torch.ones(1,x.shape[0]),0,1)
        h = self.h*y
        for j in range(x.shape[2]-1): 
                mask[:,j]= (pad_len[:,0]==j*torch.ones((x.size(0),)).to(dtype = dtyp,device=self.device)) ## cration of the padding mask
                h = self.rnn(x[:,:,j],h) # bTCH X HIDDEN
                out = self.fc(h) # OUTPUT LOGITS
                outs[:,:,j] = out
                ab = torch.abs(h)
                beta = torch.min(ab)
                # beta = torch.mean(beta)
                
                A = (beta<=self.beta)
                self.beta = beta*A + self.beta*(not A)
        for k in range(outs.shape[1]):
            outs[:,k,:][mask==0] = 0        ## aplying the padding mask
        outs = torch.sum(outs,-1) ## extracting the usfull classification 
        return outs.to(dtype=torch.float64,device=self.device)
    
    def get_h(self):
        return self.h
    
    def get_matrixW(self):
        return self.state_dict()['rnn.weight_hh']

    def get_beta(self):
        return self.beta

    def reset_beta(self):
        self.beta = 1

class SRN(nn.Module):
    def __init__(self, hidden_dim,input_dim,output_dim,num_layers=1,bidirectional=False, activation= 'tanh',device='cpu', dtyp=32):
        super(SRN, self).__init__()
        self.device = device
        self.dtyp   = dtyp
        self.hidden = hidden_dim
        self.input  = input_dim
        self.output = output_dim
        self.layers = num_layers
        self.bidire = bidirectional
        self.activ  = activation
        self.beta   = 1
        self.rnn    = nn.RNN(input_size=self.input, hidden_size=self.hidden, num_layers=self.layers, nonlinearity=self.activ, batch_first=True, bidirectional=self.bidire)
        self.linear = nn.Linear(self.hidden, self.output)

    def forward(self,x):
        h = None
        h = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(dtype=self.dtyp, device=self.device)
        out,h = self.rnn(x, h)
        out = out[:, -1, :]
        beta = torch.abs(h)
        self.beta = torch.min(beta)
        output = self.linear(out)
        return output
    
    def get_beta(self):
        return self.beta

    def reset_beta(self):
        self.beta = 1
        self.feta = 0

    def get_Fbeta(self):
        return self.Fbeta
    
    def reset_Fbeta(self):
        self.Fbeta = {'value':[],'position':[]}

    def get_matrixW(self):
        return self.state_dict()['rnn.weight_hh']



class customSRN(nn.Module):
    def __init__(self, hidden_dim,input_dim,output_dim,seq_length,num_layers=1,bidirectional=False, activation= 'tanh',device='cpu',dtyp=32):
        super(customSRN, self).__init__()
        self.seqlen = seq_length
        self.hidden = hidden_dim
        self.input  = input_dim
        self.output = output_dim
        self.layers = num_layers
        self.bidire = bidirectional
        self.device = device
        self.dtyp   = dtyp
        self.beta   = 1
        self.feta   = 0
        self.Fbeta = {'value':[],'position':[]}
        if activation == 'tanh':
            self.activ = nn.Sigmoid()
        else:
            self.activ = nn.ReLU()
        self.U      = nn.Linear(self.input,  self.hidden).to(dtype=torch.float32, device = device)
        self.W      = nn.Linear(self.hidden, self.hidden).to(dtype=torch.float32, device = device)
        self.fc     = nn.Linear(self.hidden, self.output).to(dtype=torch.float32, device = device)

    def forward(self,x):
        self.Fbeta = {'value':[],'position':[]}
        pad_len = x[:,-1,:]
        if self.dtyp == 32:
            dtyp = torch.float32 
        else:
            dtyp = torch.float64
        x = x[:,:-1,:].to(dtype = dtyp,device=self.device)  
        h = torch.zeros(x.size(0), self.hidden).to(dtype=dtyp,device=self.device) # bqatch x hiddendi
        outs = torch.zeros(x.size(0),self.output,self.seqlen-1).to(dtype = dtyp,device=self.device) 
        mask = torch.zeros(x.size(0),self.seqlen-1).to(dtype = dtyp,device=self.device)
        x = torch.transpose(x,2,1)
        
        for j in range(x.shape[2]-1): 
            mask[:,j]= (pad_len[:,0]==j*torch.ones((x.size(0),)).to(dtype = dtyp,device=self.device)) ## cration of the padding mask
            y1 = self.U(x[:,:,j]) #bqtch x hidden
            y2 = self.W(h)
            h = self.activ(y1+y2) # bTCH X HIDDEN
            out = self.fc(h) # OUTPUT LOGITS
            outs[:,:,j] = self.activ(out)
            ab = torch.abs(h)
            # print(ab.shape)
            beta = torch.min(ab)
            feta = torch.max(ab).detach()
            A = (beta<=self.beta)
            B = (feta>=self.feta)
            mi, index = ab.min(dim=1)
            self.Fbeta['value'].append(mi.detach().cpu().numpy())
            self.Fbeta['position'].append(index.detach().cpu().numpy())
            self.beta = beta*A + self.beta*(not A)
            self.feta = feta*B + self.feta*(not B)
        for k in range(outs.shape[1]):
            outs[:,k,:][mask==0] = 0        ## aplying the padding mask
        outs = torch.sum(outs,-1) ## extracting the usfull classification 
        return outs.to(dtype=torch.float64,device=self.device)
    
    def get_beta(self):
        return self.beta

    def reset_beta(self):
        self.beta = 1
        self.feta = 0

    def get_Fbeta(self):
        return self.Fbeta
    
    def reset_Fbeta(self):
        self.Fbeta = {'value':[],'position':[]}

    def get_matrixW(self):
        return self.state_dict()['W.weight']
    
class SWcustomSRN(nn.Module):
    def __init__(self, hidden_dim,input_dim,output_dim,seq_length,num_layers=1,bidirectional=False, activation= 'tanh',device='cpu',dtyp=32):
        super(SWcustomSRN, self).__init__()
        self.seqlen = seq_length
        self.hidden = hidden_dim
        self.input  = input_dim
        self.output = output_dim
        self.layers = num_layers
        self.bidire = bidirectional
        self.device = device
        self.dtyp   = dtyp
        self.sat_C  = 8
        self.beta   = 1
        self.feta   = 0
        self.Fbeta  = {'value':[],'position':[]}
        if activation == 'tanh':
            self.activ = nn.Tanh()
        else:
            self.activ = nn.ReLU()
        self.U      = nn.Linear(self.input,  self.hidden).to(dtype=torch.float32, device = device)
        self.W      = nn.Linear(self.hidden, self.hidden).to(dtype=torch.float32, device = device)
        self.fc     = nn.Linear(self.hidden, self.output).to(dtype=torch.float32, device = device)

    def forward(self,x):
        self.Fbeta = {'value':[],'position':[]}
        pad_len = x[:,-1,:]
        if self.dtyp == 32:
            dtyp = torch.float32 
        else:
            dtyp = torch.float64
        x = x[:,:-1,:].to(dtype = dtyp,device=self.device)  
        h = torch.zeros(x.size(0), self.hidden).to(dtype=dtyp,device=self.device) # bqatch x hiddendi
        outs = torch.empty(x.size(0),self.output,self.seqlen-1).to(dtype = dtyp,device=self.device) 
        mask = torch.empty(x.size(0),self.seqlen-1).to(dtype = dtyp,device=self.device)
        x = torch.transpose(x,2,1)
        
        for j in range(x.shape[2]-1): 
            mask[:,j]= (pad_len[:,0]==j*torch.ones((x.size(0),)).to(dtype = dtyp,device=self.device)) ## cration of the padding mask
            y1 = self.U(x[:,:,j]) #bqtch x hidden
            y2 = self.W(h)
            h = self.activ(y1+y2) # bTCH X HIDDEN
            out = self.fc(h) # OUTPUT LOGITS
            outs[:,:,j] = out
            ab = torch.abs(h)
            beta = torch.min(ab)
            feta = torch.max(ab).detach()
            A = (beta<=self.beta)
            B = (feta>=self.feta)
            mi,index = torch.min(ab).detach()
            self.Fbeta['value'].append(mi.item())
            self.Fbeta['position'].append(index.item())
            self.beta = beta*A + self.beta*(not A)
            self.feta = feta*B + self.feta*(not B)
        for k in range(outs.shape[1]):
            outs[:,k,:][mask==0] = 0        ## aplying the padding mask
        outs = torch.sum(outs,-1) ## extracting the usfull classification 
        return outs.to(dtype=torch.float64,device=self.device)
    
    def get_beta(self):
        return self.beta

    def reset_beta(self):
        self.beta = 1
        self.feta = 0

    def get_Fbeta(self):
        return self.Fbeta
    
    def reset_Fbeta(self):
        self.Fbeta = {'value':[],'position':[]}

    def get_matrixW(self):
        return self.state_dict()['W.weight']
    


class WcustomSRN(nn.Module):
    def __init__(self, hidden_dim,input_dim,output_dim,seq_length,epsilon=0.1,num_layers=1,bidirectional=False, activation= 'tanh',device='cpu',dtyp = 32):
        super(WcustomSRN, self).__init__()
        self.dtyp   = dtyp
        self.seqlen = seq_length
        self.hidden = hidden_dim
        self.input  = input_dim
        self.output = output_dim
        self.epsilon= epsilon
        self.layers = num_layers
        self.bidire = bidirectional
        self.device = device
        if activation == 'tanh':
            self.activ = nn.Tanh()
        else:
            self.activ = nn.ReLU()
        #encoder
        self.U        = nn.Parameter(torch.randn(input_dim, hidden_dim)).to(dtype = torch.float32,device = device)
        self.W        = nn.Parameter(torch.randn(hidden_dim, hidden_dim)).to(dtype = torch.float32,device = device)
        self.b        = nn.Parameter(torch.randn(hidden_dim)).to(dtype = torch.float32,device = device)
        #encoder
        self.fc       = nn.Linear(self.hidden, self.output).to(dtype = torch.float32,device = device)

    def forward(self,x):
        normalized_W = (1-self.epsilon)*(self.W/(torch.linalg.norm(self.W,ord=2)))
        if self.dtyp == 32:
            dtyp = torch.float32 
        else:
            dtyp = torch.float64
        x = x[:,:50,:].to(dtype = dtyp,device=self.device)  
        h = torch.ones(x.size(0), self.hidden).to(dtype=dtyp,device=self.device)
        # h = torch.zeros(x.size(0), self.hidden).to(dtype=torch.float64,device=self.device) # bqatch x hiddendi
        pad_len = x[:,-1,:]
        # x = x[:,:50,:].to(dtype = torch.float64,device=self.device)  
        x = torch.transpose(x,2,1)
        outs = torch.empty(x.size(0),self.output,self.seqlen-1).to(dtype = dtyp,device=self.device)
        mask = torch.empty(x.size(0),self.seqlen-1).to(dtype = dtyp,device=self.device)
        for j in range(x.shape[2]-1):
            mask[:,j]= (pad_len[:,0]==j*torch.ones((x.size(0),)).to(dtype=dtyp, device=self.device)).to(dtype = dtyp,device=self.device)   ## cration of the padding mask
            y = x[:,:,j]@self.U + h@normalized_W + self.b              #bqtch x hidden
            h = self.activ(y)                            # bTCH X HIDDEN
            out = self.fc(h)                                        # OUTPUT LOGITS
            outs[:,:,j] = out
        for k in range(outs.shape[1]):
            outs[:,k,:][mask==0] = 0                                ## aplying the padding mask
        outs = torch.sum(outs,-1)                                   ## extracting the usfull classification 
        return outs

# model = customSRN(5,5,1,10)
# batch = 4
# data = []
# for _ in range(batch):
#     word = []
#     length = np.random.randint(3,7)
#     elems  = np.random.randint(0,4,length)
#     for k in range(length):
#         word.append(np.eye(5)[:,elems[k]])
#     for s in range(10 - length - 1):
#         word.append(np.zeros((5,)))
#     word.append(length*np.ones((5,)))
#     word = np.transpose(np.array(word))
#     data.append(word)
# data = np.array(data)
# data = torch.tensor(data)
# model(data)
